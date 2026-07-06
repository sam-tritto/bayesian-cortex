import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch
import urllib.error

import pytest
import numpy as np

from bayes_brain.embeddings import (
    AsyncVectorContextStore,
    AsyncSQLiteVectorStore,
    GeminiEmbedder,
    OpenAIEmbedder,
)
from bayes_brain.router import AsyncBayesianToolRouter
from bayes_brain.storage import (
    AsyncInMemoryStorage,
    AsyncSQLiteStorage,
    AsyncRedisStorage,
)


@pytest.mark.anyio
async def test_async_in_memory_storage():
    storage = AsyncInMemoryStorage()

    # Defaults
    alpha, beta = await storage.get_tool_params("ctx_test", "tool_a")
    assert alpha == 1.0
    assert beta == 1.0

    # Updates
    await storage.update_tool_params("ctx_test", "tool_a", 5.5, 4.2)
    alpha, beta = await storage.get_tool_params("ctx_test", "tool_a")
    assert alpha == 5.5
    assert beta == 4.2

    # Decay & Update
    new_a, new_b = await storage.decay_and_update("ctx_test", "tool_a", 0.5, 1.0)
    assert new_a == 5.5 * 0.5 + 1.0
    assert new_b == 4.2 * 0.5 + 0.0

    # Verify decay lower-bounding
    new_a, new_b = await storage.decay_and_update("ctx_test", "tool_a", 0.1, 0.0)
    assert new_a == max(1.0, 3.75 * 0.1 + 0.0)
    assert new_b == max(1.0, 2.1 * 0.1 + 1.0)
    assert new_a == 1.0

    # Metadata
    await storage.save_metadata("my_key", "my_val")
    assert await storage.load_metadata("my_key") == "my_val"
    assert await storage.load_metadata("missing") is None


@pytest.mark.anyio
async def test_async_sqlite_storage():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        storage = AsyncSQLiteStorage(db_path)

        # Defaults
        a, b = await storage.get_tool_params("ctx_1", "tool_1")
        assert a == 1.0
        assert b == 1.0

        # Updates
        await storage.update_tool_params("ctx_1", "tool_1", 10.0, 2.0)
        a, b = await storage.get_tool_params("ctx_1", "tool_1")
        assert a == 10.0
        assert b == 2.0

        # Atomic decay & update
        new_a, new_b = await storage.decay_and_update("ctx_1", "tool_1", 0.9, 1.0)
        assert new_a == 10.0 * 0.9 + 1.0
        assert new_b == 2.0 * 0.9 + 0.0

        # Capping at 1.0
        new_a, new_b = await storage.decay_and_update("ctx_1", "tool_1", 0.1, 0.0)
        assert new_a == 1.0
        assert new_b == max(1.0, 1.8 * 0.1 + 1.0)

        # Metadata
        await storage.save_metadata("vector_data", "serialized_vector_json")
        assert await storage.load_metadata("vector_data") == "serialized_vector_json"

        # Vectors saving / migration
        await storage.save_vector("ctx_vec", [0.1, 0.2])
        vectors = await storage.load_all_vectors()
        assert vectors["ctx_vec"] == [0.1, 0.2]

        await storage.close()
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


@pytest.mark.anyio
async def test_async_redis_storage():
    mock_client = AsyncMock()
    mock_script = AsyncMock()

    # Script registration is a synchronous client operation
    mock_client.register_script = MagicMock(return_value=mock_script)
    mock_script.return_value = ["1.5", "2.5"]

    # hget mockup
    async def mock_hget(key, field):
        lookup = {
            "bayes_brain:ctx_1:tool_1:alpha": "10.0",
            "bayes_brain:ctx_1:tool_1:beta": "5.0",
        }
        return lookup.get(f"{key}:{field}", None)

    mock_client.hget.side_effect = mock_hget
    mock_client.get.return_value = b"meta_value"

    storage = AsyncRedisStorage(mock_client, prefix="bayes_brain:")

    # Get params
    a, b = await storage.get_tool_params("ctx_1", "tool_1")
    assert a == 10.0
    assert b == 5.0

    # Update params
    await storage.update_tool_params("ctx_1", "tool_1", 12.0, 6.0)
    mock_client.hset.assert_called_with(
        "bayes_brain:ctx_1",
        mapping={"tool_1:alpha": "12.0", "tool_1:beta": "6.0"}
    )

    # Decay & Update
    new_a, new_b = await storage.decay_and_update("ctx_1", "tool_1", 0.9, 1.0)
    assert new_a == 1.5
    assert new_b == 2.5
    mock_script.assert_called_with(
        keys=["bayes_brain:ctx_1"],
        args=["tool_1:alpha", "tool_1:beta", "0.9", "1.0"]
    )

    # Metadata
    assert await storage.load_metadata("some_key") == "meta_value"


@pytest.mark.anyio
async def test_async_vector_context_store():
    store = AsyncVectorContextStore()

    await store.aadd_context("ctx_search", [1.0, 0.0, 0.0])
    await store.aadd_context("ctx_math", [0.0, 1.0, 0.0])

    assert await store.aget_nearest_context([1.0, 0.0, 0.0], 0.9) == "ctx_search"
    assert await store.aget_nearest_context([0.9, 0.1, 0.0], 0.8) == "ctx_search"
    assert await store.aget_nearest_context([0.5, 0.5, 0.0], 0.95) is None


@pytest.mark.anyio
async def test_async_sqlite_vector_store():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        store = AsyncSQLiteVectorStore(db_path, dimension=3)
        await store.aadd_context("ctx_search", [1.0, 0.0, 0.0])
        await store.aadd_context("ctx_math", [0.0, 1.0, 0.0])

        assert await store.aget_nearest_context([1.0, 0.0, 0.0], 0.9) == "ctx_search"
        assert await store.aget_nearest_context([0.9, 0.1, 0.0], 0.8) == "ctx_search"
        assert await store.aget_nearest_context([0.5, 0.5, 0.0], 0.95) is None

        await store.close()
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


class AsyncMockEmbedder:
    async def aembed_query(self, text: str):
        if "search" in text.lower():
            return [1.0, 0.0]
        return [0.0, 1.0]


class SyncMockEmbedder:
    def embed_query(self, text: str):
        if "search" in text.lower():
            return [1.0, 0.0]
        return [0.0, 1.0]


@pytest.mark.anyio
async def test_async_router_exact_match():
    storage = AsyncInMemoryStorage()
    router = AsyncBayesianToolRouter(storage=storage, decay_factor=0.95)

    tool = await router.aroute("web_search_query", ["search_api", "fallback_api"])
    assert tool in ["search_api", "fallback_api"]

    # feedback
    await router.afeedback("web_search_query", "search_api", success=True)
    key = await router._resolve_context_key("web_search_query")
    a_success, b_success = await storage.get_tool_params(key, "search_api")
    assert a_success == pytest.approx(1.95)
    assert b_success == pytest.approx(1.0)


@pytest.mark.anyio
async def test_async_router_with_async_embedder():
    storage = AsyncInMemoryStorage()
    embedder = AsyncMockEmbedder()
    router = AsyncBayesianToolRouter(storage=storage, embedder=embedder)

    tool, trace = await router.aroute_with_trace("find math help", ["tool_math", "tool_search"])
    context_key_1 = await router._resolve_context_key("find math help")
    assert context_key_1.startswith("ctx_")

    context_key_2 = await router._resolve_context_key("do some math stuff")
    assert context_key_1 == context_key_2


@pytest.mark.anyio
async def test_async_router_with_sync_embedder():
    storage = AsyncInMemoryStorage()
    embedder = SyncMockEmbedder()
    router = AsyncBayesianToolRouter(storage=storage, embedder=embedder)

    tool, trace = await router.aroute_with_trace("find math help", ["tool_math", "tool_search"])
    context_key_1 = await router._resolve_context_key("find math help")
    assert context_key_1.startswith("ctx_")


@pytest.mark.anyio
async def test_async_router_trace_feedback():
    storage = AsyncInMemoryStorage()
    router = AsyncBayesianToolRouter(storage=storage)

    chosen_tool, trace_id = await router.aroute_with_trace("context_a", ["tool_x"])
    assert chosen_tool == "tool_x"

    await router.afeedback_by_trace(trace_id, success=True)
    
    key = await router._resolve_context_key("context_a")
    alpha, beta = await storage.get_tool_params(key, "tool_x")
    assert alpha == 2.0
    assert beta == 1.0


@pytest.mark.anyio
async def test_async_router_priors():
    storage = AsyncInMemoryStorage()
    priors = {"highly_reliable": (90.0, 10.0), "unreliable": (1.0, 99.0)}
    router = AsyncBayesianToolRouter(storage=storage, priors=priors)

    chosen = await router.aroute("some_task", ["highly_reliable", "unreliable"])
    assert chosen == "highly_reliable"


@pytest.mark.anyio
async def test_async_router_fallbacks(monkeypatch):
    storage = AsyncInMemoryStorage()
    
    async def mock_get_tool_params(context_key, tool_name):
        raise RuntimeError("DB failure")
    monkeypatch.setattr(storage, "get_tool_params", mock_get_tool_params)

    router = AsyncBayesianToolRouter(storage=storage, fallback_tool="fallback")
    chosen = await router.aroute("query", ["tool_a", "fallback"])
    assert chosen == "fallback"


@pytest.mark.anyio
@patch("httpx.AsyncClient")
async def test_async_gemini_embedder_rest(mock_httpx_client):
    # Mock httpx AsyncClient behavior
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "embedding": {
            "values": [0.1, 0.2, 0.3]
        }
    }
    mock_response.raise_for_status = MagicMock()

    # Async context manager setup
    mock_client_instance = AsyncMock()
    mock_client_instance.post.return_value = mock_response
    mock_httpx_client.return_value.__aenter__.return_value = mock_client_instance

    embedder = GeminiEmbedder(api_key="fake-key")
    result = await embedder.aembed_query("hello")
    assert result == [0.1, 0.2, 0.3]
    mock_client_instance.post.assert_called_once()


@pytest.mark.anyio
@patch("httpx.AsyncClient")
async def test_async_openai_embedder_rest(mock_httpx_client):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": [
            {
                "embedding": [0.01, -0.02, 0.03]
            }
        ]
    }
    mock_response.raise_for_status = MagicMock()

    mock_client_instance = AsyncMock()
    mock_client_instance.post.return_value = mock_response
    mock_httpx_client.return_value.__aenter__.return_value = mock_client_instance

    embedder = OpenAIEmbedder(api_key="fake-key")
    result = await embedder.aembed_query("hello")
    assert result == [0.01, -0.02, 0.03]
    mock_client_instance.post.assert_called_once()


@pytest.mark.anyio
async def test_async_router_signed_trace_ids():
    import pytest
    storage = AsyncInMemoryStorage()
    
    # 1. Custom secret key (str)
    router = AsyncBayesianToolRouter(storage=storage, secret_key="my_super_secret_key")
    chosen, trace_id = await router.aroute_with_trace("query", ["tool_a"])
    assert "." in trace_id
    
    # Decode and verify it succeeds
    ctx_key, tool_name = router._decode_trace_id(trace_id)
    assert tool_name == "tool_a"
    
    # Verify with another router using the same key succeeds
    router2 = AsyncBayesianToolRouter(storage=storage, secret_key="my_super_secret_key")
    ctx_key2, tool_name2 = router2._decode_trace_id(trace_id)
    assert tool_name2 == "tool_a"
    
    # Verify with another router using a different key fails
    router3 = AsyncBayesianToolRouter(storage=storage, secret_key="different_secret_key")
    with pytest.raises(ValueError, match="Invalid or corrupted trace ID"):
        router3._decode_trace_id(trace_id)
        
    # Tampering with payload fails
    payload_part, sig_part = trace_id.split(".")
    import json
    import base64
    payload_json = json.loads(base64.urlsafe_b64decode(payload_part).decode("utf-8"))
    payload_json["tool"] = "tool_b"  # forged
    tampered_payload_b64 = base64.urlsafe_b64encode(json.dumps(payload_json).encode("utf-8")).decode("utf-8")
    tampered_trace_id = f"{tampered_payload_b64}.{sig_part}"
    
    with pytest.raises(ValueError, match="Invalid or corrupted trace ID"):
        router._decode_trace_id(tampered_trace_id)
        
    # Missing signature separator fails
    with pytest.raises(ValueError, match="Invalid or corrupted trace ID"):
        router._decode_trace_id(payload_part)
        
    # Random key auto-generation works
    router_random1 = AsyncBayesianToolRouter(storage=storage)
    router_random2 = AsyncBayesianToolRouter(storage=storage)
    
    _, trace_id_rand = await router_random1.aroute_with_trace("query", ["tool_a"])
    # Decoding with same router succeeds
    assert router_random1._decode_trace_id(trace_id_rand)[1] == "tool_a"
    # Decoding with different router (with different random key) fails
    with pytest.raises(ValueError):
        router_random2._decode_trace_id(trace_id_rand)


@pytest.mark.anyio
async def test_async_router_contextual_priors():
    storage = AsyncInMemoryStorage()
    
    contextual_priors = [
        {
            "pattern": r"math|calculator|sum",
            "priors": {
                "calculator": (99.0, 1.0),
                "search": (1.0, 99.0)
            }
        },
        {
            "reference_context": "perform general web search query",
            "priors": {
                "calculator": (1.0, 99.0),
                "search": (99.0, 1.0)
            }
        }
    ]
    
    embedder = SyncMockEmbedder()
    
    router = AsyncBayesianToolRouter(
        storage=storage,
        embedder=embedder,
        contextual_priors=contextual_priors,
        similarity_threshold=0.85
    )

    # Test Regex Match
    prior_calc_alpha, prior_calc_beta = await router.get_prior("solve a math sum", "calculator")
    assert prior_calc_alpha == 99.0
    assert prior_calc_beta == 1.0

    prior_search_alpha, prior_search_beta = await router.get_prior("solve a math sum", "search")
    assert prior_search_alpha == 1.0
    assert prior_search_beta == 99.0

    # Test Reference Context Embedding Match
    prior_search_alpha2, prior_search_beta2 = await router.get_prior("search for weather", "search")
    assert prior_search_alpha2 == 99.0
    assert prior_search_beta2 == 1.0

    # Test routing with contextual priors (Thompson sampling cold start)
    storage_clean = AsyncInMemoryStorage()
    router_clean = AsyncBayesianToolRouter(
        storage=storage_clean,
        embedder=embedder,
        contextual_priors=contextual_priors
    )
    chosen = await router_clean.aroute("solve a math sum", ["calculator", "search"])
    assert chosen == "calculator"
    
    # Verify parameter seeding in storage
    key = await router_clean._resolve_context_key("solve a math sum")
    alpha_stored, beta_stored = await storage_clean.get_tool_params(key, "calculator")
    assert alpha_stored == 99.0
    assert beta_stored == 1.0

    # Route batch
    storage_batch = AsyncInMemoryStorage()
    router_batch = AsyncBayesianToolRouter(
        storage=storage_batch,
        embedder=embedder,
        contextual_priors=contextual_priors
    )
    results = await router_batch.aroute_batch(["solve a math sum", "search for weather"], ["calculator", "search"])
    assert results == ["calculator", "search"]


