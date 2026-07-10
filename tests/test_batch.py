import os
import tempfile
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from bayesian_cortex.router import AsyncBayesianRouter, BayesianRouter
from bayesian_cortex.storage import (
    AsyncRedisStorage,
    AsyncSQLiteStorage,
    RedisStorage,
    SQLiteStorage,
)


def test_in_memory_storage_batch(mem_storage):
    storage = mem_storage

    # Batch updates
    params = {
        ("ctx_1", "tool_a"): (3.0, 4.0),
        ("ctx_1", "tool_b"): (5.0, 6.0),
        ("ctx_2", "tool_a"): (7.0, 8.0),
    }
    storage.update_candidate_params_batch(params)

    # Batch retrieval
    keys = [
        ("ctx_1", "tool_a"),
        ("ctx_1", "tool_b"),
        ("ctx_2", "tool_a"),
        ("ctx_2", "tool_b"),
    ]
    res = storage.get_candidate_params_batch(keys)
    assert res[("ctx_1", "tool_a")] == (3.0, 4.0)
    assert res[("ctx_1", "tool_b")] == (5.0, 6.0)
    assert res[("ctx_2", "tool_a")] == (7.0, 8.0)
    assert res[("ctx_2", "tool_b")] == (1.0, 1.0)  # Default fallback

    # Batch decay and update
    updates = [
        ("ctx_1", "tool_a", 0.5, 1.0),
        ("ctx_1", "tool_a", 0.5, 0.0),
    ]
    decayed = storage.decay_and_update_batch(updates)
    assert len(decayed) == 2
    # First: 3.0 * 0.5 + 1.0 = 2.5, 4.0 * 0.5 + 0.0 = 2.0
    assert decayed[0] == (2.5, 2.0)
    # Second: 2.5 * 0.5 + 0.0 = 1.25, 2.0 * 0.5 + 1.0 = 2.0
    assert decayed[1] == (1.25, 2.0)

    # Context vectors batch save
    vectors = {
        "ctx_1": [0.1, 0.2, 0.3],
        "ctx_2": [0.4, 0.5, 0.6],
    }
    storage.save_vectors(vectors)
    all_vecs = storage.load_all_vectors()
    assert all_vecs["ctx_1"] == [0.1, 0.2, 0.3]
    assert all_vecs["ctx_2"] == [0.4, 0.5, 0.6]

    # Linear parameters batch decay and update
    x_aug = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    linear_updates = [
        ("tool_a", 0.9, 1.0, x_aug, 1.0, 0.5, True),
        ("tool_a", 0.9, 0.0, x_aug, 1.0, 0.5, True),
    ]
    res_linear = storage.decay_and_update_linear_batch(linear_updates)
    assert len(res_linear) == 2
    # Verify we can fetch them
    fetched = storage.get_linear_params_batch(["tool_a"])
    assert "tool_a" in fetched
    assert fetched["tool_a"][0] is not None


def test_sqlite_storage_batch():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        storage = SQLiteStorage(db_path)

        # Batch updates
        params = {
            ("ctx_1", "tool_a"): (3.0, 4.0),
            ("ctx_1", "tool_b"): (5.0, 6.0),
        }
        storage.update_candidate_params_batch(params)

        # Batch retrieval
        res = storage.get_candidate_params_batch(
            [("ctx_1", "tool_a"), ("ctx_1", "tool_b"), ("ctx_2", "tool_a")]
        )
        assert res[("ctx_1", "tool_a")] == (3.0, 4.0)
        assert res[("ctx_1", "tool_b")] == (5.0, 6.0)
        assert res[("ctx_2", "tool_a")] == (1.0, 1.0)

        # Batch decay and update
        updates = [
            ("ctx_1", "tool_a", 0.5, 1.0),
            ("ctx_1", "tool_a", 0.5, 0.0),
        ]
        decayed = storage.decay_and_update_batch(updates)
        assert decayed[0] == (2.5, 2.0)
        assert decayed[1] == (1.25, 2.0)

        # save vectors batch
        storage.save_vectors({"ctx_1": [1.0, 2.0], "ctx_2": [3.0, 4.0]})
        all_vecs = storage.load_all_vectors()
        assert all_vecs["ctx_1"] == [1.0, 2.0]
        assert all_vecs["ctx_2"] == [3.0, 4.0]

        storage.close()
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


@pytest.mark.anyio
async def test_async_sqlite_storage_batch():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        storage = AsyncSQLiteStorage(db_path)

        # Batch updates
        params = {
            ("ctx_1", "tool_a"): (3.0, 4.0),
            ("ctx_1", "tool_b"): (5.0, 6.0),
        }
        await storage.update_candidate_params_batch(params)

        # Batch retrieval
        res = await storage.get_candidate_params_batch(
            [("ctx_1", "tool_a"), ("ctx_1", "tool_b"), ("ctx_2", "tool_a")]
        )
        assert res[("ctx_1", "tool_a")] == (3.0, 4.0)
        assert res[("ctx_1", "tool_b")] == (5.0, 6.0)
        assert res[("ctx_2", "tool_a")] == (1.0, 1.0)

        # Batch decay and update
        updates = [
            ("ctx_1", "tool_a", 0.5, 1.0),
            ("ctx_1", "tool_a", 0.5, 0.0),
        ]
        decayed = await storage.decay_and_update_batch(updates)
        assert decayed[0] == (2.5, 2.0)
        assert decayed[1] == (1.25, 2.0)

        # save vectors batch
        await storage.asave_vectors({"ctx_1": [1.0, 2.0], "ctx_2": [3.0, 4.0]})
        all_vecs = await storage.load_all_vectors()
        assert all_vecs["ctx_1"] == [1.0, 2.0]
        assert all_vecs["ctx_2"] == [3.0, 4.0]

        await storage.close()
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_redis_storage_batch():
    mock_client = MagicMock()
    mock_pipeline = MagicMock()
    mock_client.pipeline.return_value = mock_pipeline

    storage = RedisStorage(mock_client)

    # 1. get_candidate_params_batch
    mock_pipeline.execute.return_value = [b"3.5", b"4.5", None, None]
    keys = [("ctx_1", "tool_a"), ("ctx_2", "tool_b")]
    res = storage.get_candidate_params_batch(keys)
    assert res[("ctx_1", "tool_a")] == (3.5, 4.5)
    assert res[("ctx_2", "tool_b")] == (1.0, 1.0)
    assert mock_pipeline.hget.call_count == 4

    # 2. update_candidate_params_batch
    mock_pipeline.reset_mock()
    storage.update_candidate_params_batch({("ctx_1", "tool_a"): (1.5, 2.5)})
    mock_pipeline.hset.assert_called_once()
    mock_pipeline.execute.assert_called_once()

    # 3. decay_and_update_batch
    mock_pipeline.reset_mock()
    mock_script = MagicMock()
    storage._script = mock_script

    mock_pipeline.execute.return_value = [["2.5", "3.5"]]
    updates = [("ctx_1", "tool_a", 0.9, 1.0)]
    res_decay = storage.decay_and_update_batch(updates)
    assert res_decay == [(2.5, 3.5)]
    mock_script.assert_called_once()
    mock_pipeline.execute.assert_called_once()


@pytest.mark.anyio
async def test_async_redis_storage_batch():
    mock_client = MagicMock()
    mock_pipeline = AsyncMock()
    mock_client.pipeline.return_value = mock_pipeline

    # We must mock register_script as returning an AsyncMock script
    mock_script = AsyncMock()
    mock_client.register_script.return_value = mock_script

    storage = AsyncRedisStorage(mock_client)
    storage._script = mock_script

    # 1. get_candidate_params_batch
    mock_pipeline.hget = MagicMock()
    mock_pipeline.execute.return_value = [b"3.5", b"4.5", None, None]
    keys = [("ctx_1", "tool_a"), ("ctx_2", "tool_b")]
    res = await storage.get_candidate_params_batch(keys)
    assert res[("ctx_1", "tool_a")] == (3.5, 4.5)
    assert res[("ctx_2", "tool_b")] == (1.0, 1.0)
    assert mock_pipeline.hget.call_count == 4

    # 2. decay_and_update_batch
    mock_pipeline.execute.return_value = [["2.5", "3.5"]]
    updates = [("ctx_1", "tool_a", 0.9, 1.0)]
    res_decay = await storage.decay_and_update_batch(updates)
    assert res_decay == [(2.5, 3.5)]


def test_router_batch_routing_clustering(mem_storage):
    storage = mem_storage
    router = BayesianRouter(storage=storage)

    # Set tool priors to force deterministic/seeded behavior
    router.priors = {"tool_a": (10.0, 2.0), "tool_b": (1.0, 10.0)}

    contexts = ["hello world", "style clean up", "hello world"]
    candidates = ["tool_a", "tool_b"]

    # Batch route
    choices = router.route_batch(contexts, candidates)
    assert len(choices) == 3
    # Check that choices are mapped correctly: "hello world" is mapped to "tool_a" (higher prior reward)
    # and they should reuse the context keys!
    results_with_trace = router.route_batch_with_trace(contexts, candidates)
    assert len(results_with_trace) == 3

    # Check trace 0 and trace 2 use the exact same context key since they have identical context texts
    ctx_0, tool_0 = router._decode_trace_id(results_with_trace[0][1])
    ctx_2, tool_2 = router._decode_trace_id(results_with_trace[2][1])
    assert ctx_0 == ctx_2

    # Batch feedback
    feedbacks = [
        {"trace_id": results_with_trace[0][1], "success": True},
        {"trace_id": results_with_trace[1][1], "reward": 0.0},
    ]
    router.feedback_batch(feedbacks)

    # Check parameters updated
    alpha, beta = storage.get_candidate_params(ctx_0, tool_0)
    assert alpha > 1.0


@pytest.mark.anyio
async def test_async_router_batch_routing(async_mem_storage):
    storage = async_mem_storage
    router = AsyncBayesianRouter(storage=storage)

    router.priors = {"tool_a": (10.0, 2.0), "tool_b": (1.0, 10.0)}

    contexts = ["hello world", "style clean up"]
    candidates = ["tool_a", "tool_b"]

    choices = await router.aroute_batch(contexts, candidates)
    assert len(choices) == 2

    traces = await router.aroute_batch_with_trace(contexts, candidates)
    assert len(traces) == 2

    feedbacks = [
        {"trace_id": traces[0][1], "success": True},
        {
            "context_text": "style clean up",
            "candidate_name": "tool_b",
            "success": False,
        },
    ]
    await router.afeedback_batch(feedbacks)
