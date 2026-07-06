import os
import sqlite3
import tempfile
from unittest.mock import MagicMock

import pytest

from bayes_brain.storage import InMemoryStorage, RedisStorage, SQLiteStorage


def test_in_memory_storage():
    storage = InMemoryStorage()
    
    # Defaults
    alpha, beta = storage.get_tool_params("ctx_test", "tool_a")
    assert alpha == 1.0
    assert beta == 1.0

    # Updates
    storage.update_tool_params("ctx_test", "tool_a", 5.5, 4.2)
    alpha, beta = storage.get_tool_params("ctx_test", "tool_a")
    assert alpha == 5.5
    assert beta == 4.2

    # Decay & Update
    new_a, new_b = storage.decay_and_update("ctx_test", "tool_a", 0.5, 1.0)
    assert new_a == 5.5 * 0.5 + 1.0
    assert new_b == 4.2 * 0.5 + 0.0

    # Metadata
    storage.save_metadata("my_key", "my_val")
    assert storage.load_metadata("my_key") == "my_val"
    assert storage.load_metadata("missing") is None


def test_sqlite_storage():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name
    
    try:
        storage = SQLiteStorage(db_path)
        
        # Test defaults
        a, b = storage.get_tool_params("ctx_1", "tool_1")
        assert a == 1.0
        assert b == 1.0

        # Test updates
        storage.update_tool_params("ctx_1", "tool_1", 10.0, 2.0)
        a, b = storage.get_tool_params("ctx_1", "tool_1")
        assert a == 10.0
        assert b == 2.0

        # Test atomic decay and update
        new_a, new_b = storage.decay_and_update("ctx_1", "tool_1", 0.9, 1.0)
        assert new_a == 10.0 * 0.9 + 1.0
        assert new_b == 2.0 * 0.9 + 0.0

        # Test metadata persistence
        storage.save_metadata("vector_data", "serialized_vector_json")
        assert storage.load_metadata("vector_data") == "serialized_vector_json"
        assert storage.load_metadata("nonexistent") is None

        storage.close()
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_redis_storage():
    mock_client = MagicMock()
    mock_script = MagicMock()
    
    # Setup Lua script return
    mock_script.return_value = ["1.5", "2.5"]
    mock_client.register_script.return_value = mock_script
    
    # HGET setup
    mock_client.hget.side_effect = lambda key, field: {
        "bayes_brain:ctx_1:tool_1:alpha": b"10.0",
        "bayes_brain:ctx_1:tool_1:beta": b"5.0"
    }.get(f"{key}:{field}", None)

    # GET setup for metadata
    mock_client.get.return_value = b"meta_value"

    storage = RedisStorage(mock_client, prefix="bayes_brain:")

    # Get params
    a, b = storage.get_tool_params("ctx_1", "tool_1")
    assert a == 10.0
    assert b == 5.0
    
    # Update params
    storage.update_tool_params("ctx_1", "tool_1", 12.0, 6.0)
    mock_client.hset.assert_called_with(
        "bayes_brain:ctx_1",
        mapping={"tool_1:alpha": "12.0", "tool_1:beta": "6.0"}
    )

    # Decay & Update
    new_a, new_b = storage.decay_and_update("ctx_1", "tool_1", 0.9, 1.0)
    assert new_a == 1.5
    assert new_b == 2.5
    mock_script.assert_called_with(
        keys=["bayes_brain:ctx_1"],
        args=["tool_1:alpha", "tool_1:beta", "0.9", "1.0"]
    )

    # Metadata
    assert storage.load_metadata("some_key") == "meta_value"
    mock_client.get.assert_called_with("bayes_brain:metadata:some_key")
    
    storage.save_metadata("some_key", "new_val")
    mock_client.set.assert_called_with("bayes_brain:metadata:some_key", "new_val")


def test_sqlite_storage_incremental_and_migration():
    import json
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name
    
    try:
        # 1. Preseed with legacy metadata to simulate a legacy DB
        conn = sqlite3.connect(db_path)
        with conn:
            conn.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, val TEXT)")
            legacy_data = {"ctx_legacy_1": [0.1, 0.2], "ctx_legacy_2": [0.3, 0.4]}
            conn.execute(
                "INSERT INTO metadata (key, val) VALUES (?, ?)",
                ("vector_context_store", json.dumps(legacy_data))
            )
        conn.close()

        # 2. Instantiate SQLiteStorage and call load_all_vectors.
        # This should trigger migration and retrieve the migrated vectors.
        storage = SQLiteStorage(db_path)
        vectors = storage.load_all_vectors()
        assert vectors == legacy_data

        # 3. Test saving a new vector incrementally
        storage.save_vector("ctx_new", [0.5, 0.6])
        
        # Verify the new vector is in the loaded set
        updated_vectors = storage.load_all_vectors()
        assert updated_vectors["ctx_legacy_1"] == [0.1, 0.2]
        assert updated_vectors["ctx_legacy_2"] == [0.3, 0.4]
        assert updated_vectors["ctx_new"] == [0.5, 0.6]

        # Verify the database table 'context_vectors' actually contains the rows
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT context_key, vector FROM context_vectors WHERE context_key = ?", ("ctx_new",))
        row = cursor.fetchone()
        assert row is not None
        assert json.loads(row[1]) == [0.5, 0.6]
        conn.close()

        storage.close()
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_redis_storage_incremental_and_migration():
    import json
    mock_client = MagicMock()
    
    # Simulate empty context_vectors hash initially
    # If hgetall is called on non-existent hash, it returns empty dict
    mock_client.hgetall.return_value = {}
    
    # Setup legacy metadata return when requested
    legacy_data = {"ctx_legacy_1": [0.5, 0.5]}
    mock_client.get.side_effect = lambda key: {
        "bayes_brain:metadata:vector_context_store": json.dumps(legacy_data).encode("utf-8")
    }.get(key, None)

    storage = RedisStorage(mock_client, prefix="bayes_brain:")

    # 1. Trigger load_all_vectors, which should fallback and migrate
    vectors = storage.load_all_vectors()
    assert vectors == legacy_data
    
    # Verify migration writes to the context_vectors hash
    mock_client.hset.assert_any_call(
        "bayes_brain:context_vectors",
        mapping={"ctx_legacy_1": json.dumps([0.5, 0.5])}
    )

    # 2. Test saving vector incrementally
    storage.save_vector("ctx_new", [0.9, 0.1])
    mock_client.hset.assert_called_with(
        "bayes_brain:context_vectors",
        key="ctx_new",
        value=json.dumps([0.9, 0.1])
    )


def test_sqlite_storage_wal_and_timeout():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name
    
    try:
        storage = SQLiteStorage(db_path)
        conn = storage._get_conn()
        
        # Check journal mode is WAL
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode;")
        journal_mode = cursor.fetchone()[0]
        assert journal_mode.lower() == "wal"
        
        # Check busy_timeout is 5000 (ms)
        cursor.execute("PRAGMA busy_timeout;")
        busy_timeout = cursor.fetchone()[0]
        assert busy_timeout == 5000
        
        storage.close()
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)

