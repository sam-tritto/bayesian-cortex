from typing import Optional, Sequence

import pytest

from bayesian_cortex.embeddings import VectorContextStore
from bayesian_cortex.router import BayesianRouter
from bayesian_cortex.storage import InMemoryStorage


def test_vector_context_store():
    store = VectorContextStore()
    
    # Add contexts
    store.add_context("ctx_search", [1.0, 0.0, 0.0])
    store.add_context("ctx_math", [0.0, 1.0, 0.0])

    # Query with exact match
    assert store.get_nearest_context([1.0, 0.0, 0.0], 0.9) == "ctx_search"
    assert store.get_nearest_context([0.0, 1.0, 0.0], 0.9) == "ctx_math"

    # Query with close match
    assert store.get_nearest_context([0.9, 0.1, 0.0], 0.8) == "ctx_search"
    
    # Query with no match (below threshold)
    assert store.get_nearest_context([0.5, 0.5, 0.0], 0.95) is None


class MockEmbedder:
    def embed_query(self, text: str) -> Sequence[float]:
        # Simple rule: if "search" in text -> [1, 0], if "math" -> [0, 1]
        if "search" in text.lower():
            return [1.0, 0.0]
        return [0.0, 1.0]


def test_router_without_embeddings():
    storage = InMemoryStorage()
    router = BayesianRouter(storage=storage, decay_factor=0.95)

    # Route should select from candidate tools
    candidate_name = router.route("web_search_query", ["search_api", "fallback_api"])
    assert candidate_name in ["search_api", "fallback_api"]

    # Provide feedback
    router.feedback("web_search_query", "search_api", success=True)
    key = router._resolve_context_key("web_search_query")
    a_success, b_success = storage.get_candidate_params(key, "search_api")
    # Initial was (1, 1). Decayed: alpha = max(1.0, 1 * 0.95 + 1.0) = 1.95, beta = max(1.0, 1 * 0.95 + 0.0) = 1.0
    assert a_success == pytest.approx(1.95)
    assert b_success == pytest.approx(1.0)

    # Provide failure feedback
    router.feedback("web_search_query", "search_api", success=False)
    a_fail, b_fail = storage.get_candidate_params(key, "search_api")
    # Decayed: alpha = max(1.0, 1.95 * 0.95 + 0) = 1.8525, beta = max(1.0, 1.0 * 0.95 + 1.0) = 1.95
    assert a_fail == pytest.approx(1.8525)
    assert b_fail == pytest.approx(1.95)


def test_router_with_embeddings():
    storage = InMemoryStorage()
    embedder = MockEmbedder()
    router = BayesianRouter(
        storage=storage,
        embedder=embedder,
        decay_factor=1.0,
        similarity_threshold=0.85
    )

    # First routing creates a new cluster context key
    tool_1, trace_1 = router.route_with_trace("find math help", ["tool_math", "tool_search"])
    # Resolve context text to a context key
    context_key_1 = router._resolve_context_key("find math help")
    assert context_key_1.startswith("ctx_")

    # Routing a similar text matches the same context cluster
    context_key_2 = router._resolve_context_key("do some math stuff")
    assert context_key_1 == context_key_2

    # Routing a search text yields a different cluster context key
    context_key_search = router._resolve_context_key("do search things")
    assert context_key_1 != context_key_search


def test_router_trace_feedback():
    storage = InMemoryStorage()
    router = BayesianRouter(storage=storage)

    chosen_candidate, trace_id = router.route_with_trace("context_a", ["tool_x"])
    assert chosen_candidate == "tool_x"

    # Feedback using trace ID
    router.feedback_by_trace(trace_id, success=True)
    
    key = router._resolve_context_key("context_a")
    alpha, beta = storage.get_candidate_params(key, "tool_x")
    # (1*1 + 1) = 2.0, (1*1 + 0) = 1.0
    assert alpha == 2.0
    assert beta == 1.0


def test_router_priors_seeding():
    storage = InMemoryStorage()
    priors = {
        "highly_reliable": (90.0, 10.0),
        "unreliable": (1.0, 99.0)
    }
    router = BayesianRouter(storage=storage, priors=priors)

    # Highly reliable should be preferred
    chosen = router.route("some_task", ["highly_reliable", "unreliable"])
    assert chosen == "highly_reliable"

    # Verify storage contains the seeded priors
    key = router._resolve_context_key("some_task")
    a_rel, b_rel = storage.get_candidate_params(key, "highly_reliable")
    assert a_rel == 90.0
    assert b_rel == 10.0


class CustomMemoryVectorStore:
    def __init__(self) -> None:
        self.vectors = {}

    def add_context(self, context_key: str, vector: Sequence[float]) -> None:
        self.vectors[context_key] = vector

    def get_nearest_context(
        self, query_vector: Sequence[float], similarity_threshold: float = 0.8
    ) -> Optional[str]:
        if self.vectors:
            return list(self.vectors.keys())[0]
        return None


def test_router_with_custom_vector_store():
    storage = InMemoryStorage()
    embedder = MockEmbedder()
    custom_store = CustomMemoryVectorStore()
    
    router = BayesianRouter(
        storage=storage,
        embedder=embedder,
        vector_store=custom_store
    )
    
    # Initially empty
    assert len(custom_store.vectors) == 0
    
    # Resolving context text creates a new cluster context key
    context_key = router._resolve_context_key("some search query")
    assert context_key.startswith("ctx_")
    
    # Verify custom_store has the new context key stored
    assert context_key in custom_store.vectors
    assert custom_store.get_nearest_context([1.0, 0.0]) == context_key


def test_router_exact_match_hashing_and_normalization():
    storage = InMemoryStorage()
    router = BayesianRouter(storage=storage)

    # Clean query
    key1 = router._resolve_context_key("my context query")
    # Prefix check
    assert key1.startswith("hash_")
    # Length check: 5 for "hash_" + 64 for sha256 hex digest = 69
    assert len(key1) == 69

    # Whitespace normalization check
    key2 = router._resolve_context_key("  my context query  ")
    key3 = router._resolve_context_key("my \t context \n query")
    assert key1 == key2
    assert key1 == key3

    # Different content produces different hash
    key_other = router._resolve_context_key("different context query")
    assert key1 != key_other


def test_router_embedder_failure_fallback_hashing(caplog):
    class CrashingEmbedder:
        def embed_query(self, text: str):
            raise ValueError("Embedding engine offline")

    storage = InMemoryStorage()
    router = BayesianRouter(storage=storage, embedder=CrashingEmbedder())

    with caplog.at_level("WARNING"):
        key = router._resolve_context_key("test warning logs")
    
    assert key.startswith("hash_")
    # Assert warning was logged
    warnings = [rec.message for rec in caplog.records if rec.levelname == "WARNING"]
    assert any("Failed to generate embedding for context" in w for w in warnings)


def test_router_no_embedder_warning(caplog):
    storage = InMemoryStorage()
    with caplog.at_level("WARNING"):
        BayesianRouter(storage=storage)
    
    warnings = [rec.message for rec in caplog.records if rec.levelname == "WARNING"]
    assert any("No ContextEmbedder provided" in w for w in warnings)


def test_continuous_rewards():
    storage = InMemoryStorage()
    router = BayesianRouter(storage=storage, decay_factor=0.95)

    # Resolve context key
    key = router._resolve_context_key("continuous_task")

    # 1. Test feedback using reward float
    router.feedback("continuous_task", "tool_a", reward=0.8)
    a, b = storage.get_candidate_params(key, "tool_a")
    # alpha: max(1.0, 1 * 0.95 + 0.8) = 1.75
    # beta: max(1.0, 1 * 0.95 + 0.2) = 1.15
    assert a == pytest.approx(1.75)
    assert b == pytest.approx(1.15)

    # 2. Test feedback_by_trace using reward float
    _, trace_id = router.route_with_trace("continuous_task", ["tool_a"])
    router.feedback_by_trace(trace_id, reward=0.4)
    # alpha: max(1.0, 1.75 * 0.95 + 0.4) = 1.6625 + 0.4 = 2.0625
    # beta: max(1.0, 1.15 * 0.95 + 0.6) = 1.0925 + 0.6 = 1.6925
    a2, b2 = storage.get_candidate_params(key, "tool_a")
    assert a2 == pytest.approx(2.0625)
    assert b2 == pytest.approx(1.6925)

    # 3. Test value validations
    with pytest.raises(ValueError, match="Either 'success' or 'reward' must be provided"):
        router.feedback("continuous_task", "tool_a")

    with pytest.raises(ValueError, match="reward must be between 0.0 and 1.0 inclusive"):
        router.feedback("continuous_task", "tool_a", reward=1.5)

    with pytest.raises(ValueError, match="reward must be between 0.0 and 1.0 inclusive"):
        router.feedback("continuous_task", "tool_a", reward=-0.1)

    # 4. Test conflicting success and reward
    with pytest.raises(ValueError, match="Conflicting feedback"):
        router.feedback("continuous_task", "tool_a", success=True, reward=0.5)

    # 5. Test consistent success and reward (should succeed)
    router.feedback("continuous_task", "tool_a", success=True, reward=1.0)
    router.feedback("continuous_task", "tool_a", success=False, reward=0.0)


def test_router_fallback_on_storage_failure(monkeypatch):
    import numpy as np
    storage = InMemoryStorage()
    
    # Mock storage to fail on get_candidate_params
    def mock_get_candidate_params(context_key, candidate_name):
        raise RuntimeError("DB connection lost")
    monkeypatch.setattr(storage, "get_candidate_params", mock_get_candidate_params)
    
    telemetry_events = []
    def mock_telemetry(event, exc, ctx):
        telemetry_events.append((event, exc, ctx))
        
    router = BayesianRouter(
        storage=storage,
        fallback_candidate="fallback_candidate",
        telemetry_hook=mock_telemetry
    )
    
    # Route with fallback_candidate present in candidate list
    chosen, trace_id = router.route_with_trace("query", ["tool_a", "fallback_candidate"])
    assert chosen == "fallback_candidate"
    assert trace_id is not None
    assert len(telemetry_events) == 1
    assert telemetry_events[0][0] == "route_failure"
    assert isinstance(telemetry_events[0][1], RuntimeError)
    assert telemetry_events[0][2]["context_text"] == "query"

    # Route with fallback_candidate NOT present in candidate list (should fall back to first candidate)
    chosen_first, _ = router.route_with_trace("query", ["tool_a", "tool_b"])
    assert chosen_first == "tool_a"


def test_router_fallback_on_sampling_failure(monkeypatch):
    import numpy as np
    storage = InMemoryStorage()
    
    # Mock numpy.random.beta to fail
    def mock_beta(a, b):
        raise ValueError("Sampling failed")
    monkeypatch.setattr(np.random, "beta", mock_beta)
    
    telemetry_events = []
    def mock_telemetry(event, exc, ctx):
        telemetry_events.append((event, exc, ctx))
        
    router = BayesianRouter(
        storage=storage,
        telemetry_hook=mock_telemetry
    )
    
    chosen, trace_id = router.route_with_trace("query", ["tool_a", "tool_b"])
    assert chosen == "tool_a"
    assert len(telemetry_events) == 1
    assert telemetry_events[0][0] == "route_failure"


def test_router_fallback_on_vector_store_failure():
    class FailingVectorStore:
        def add_context(self, context_key, vector):
            pass
        def get_nearest_context(self, query_vector, similarity_threshold):
            raise RuntimeError("Index corrupted")

    storage = InMemoryStorage()
    embedder = MockEmbedder()
    
    telemetry_events = []
    def mock_telemetry(event, exc, ctx):
        telemetry_events.append((event, exc, ctx))
        
    router = BayesianRouter(
        storage=storage,
        embedder=embedder,
        vector_store=FailingVectorStore(),
        telemetry_hook=mock_telemetry
    )
    
    chosen, trace_id = router.route_with_trace("query", ["tool_a", "tool_b"])
    assert chosen == "tool_a"
    assert len(telemetry_events) == 1
    assert telemetry_events[0][0] == "route_failure"


def test_feedback_fallback_on_failure(monkeypatch):
    storage = InMemoryStorage()
    
    # Mock storage to fail on decay_and_update
    def mock_decay_and_update(context_key, candidate_name, decay_factor, reward_val):
        raise RuntimeError("Write failed")
    monkeypatch.setattr(storage, "decay_and_update", mock_decay_and_update)
    
    telemetry_events = []
    def mock_telemetry(event, exc, ctx):
        telemetry_events.append((event, exc, ctx))
        
    router = BayesianRouter(
        storage=storage,
        telemetry_hook=mock_telemetry
    )
    
    # 1. Test feedback fallback
    alpha, beta = router.feedback("query", "tool_a", success=True)
    assert alpha == 1.0
    assert beta == 1.0
    assert len(telemetry_events) == 1
    assert telemetry_events[0][0] == "feedback_failure"
    
    # 2. Test feedback_by_trace fallback
    alpha_by_trace, beta_by_trace = router.feedback_by_trace(
        "eyJjdHgiOiAiZmFsbGJhY2tfY3R4IiwgInRvb2wiOiAic29tZV90b29sIiwgIm5vbmNlIjogIjEifQ==",
        success=True
    )
    assert alpha_by_trace == 1.0
    assert beta_by_trace == 1.0
    assert len(telemetry_events) == 2
    assert telemetry_events[1][0] == "feedback_by_trace_failure"

    # 3. Test get_candidate_beliefs fallback
    def mock_get_candidate_params(context_key, candidate_name):
        raise RuntimeError("Read failed")
    monkeypatch.setattr(storage, "get_candidate_params", mock_get_candidate_params)
    
    alpha_beliefs, beta_beliefs = router.get_candidate_beliefs("query", "tool_a")
    assert alpha_beliefs == 1.0
    assert beta_beliefs == 1.0
    assert len(telemetry_events) == 3
    assert telemetry_events[2][0] == "get_candidate_beliefs_failure"


def test_router_signed_trace_ids():
    import pytest
    storage = InMemoryStorage()
    
    # 1. Custom secret key (str)
    router = BayesianRouter(storage=storage, secret_key="my_super_secret_key")
    chosen, trace_id = router.route_with_trace("query", ["tool_a"])
    assert "." in trace_id
    
    # Decode and verify it succeeds
    ctx_key, candidate_name = router._decode_trace_id(trace_id)
    assert candidate_name == "tool_a"
    
    # Verify with another router using the same key succeeds
    router2 = BayesianRouter(storage=storage, secret_key="my_super_secret_key")
    ctx_key2, candidate_name2 = router2._decode_trace_id(trace_id)
    assert candidate_name2 == "tool_a"
    
    # Verify with another router using a different key fails
    router3 = BayesianRouter(storage=storage, secret_key="different_secret_key")
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
    router_random1 = BayesianRouter(storage=storage)
    router_random2 = BayesianRouter(storage=storage)
    
    _, trace_id_rand = router_random1.route_with_trace("query", ["tool_a"])
    # Decoding with same router succeeds
    assert router_random1._decode_trace_id(trace_id_rand)[1] == "tool_a"
    # Decoding with different router (with different random key) fails
    with pytest.raises(ValueError):
        router_random2._decode_trace_id(trace_id_rand)


def test_router_contextual_priors():
    storage = InMemoryStorage()
    
    # 1. Validation test
    with pytest.raises(ValueError, match="Each contextual prior must contain a 'priors' dictionary"):
        BayesianRouter(storage=storage, contextual_priors=[{"pattern": "math"}])

    with pytest.raises(ValueError, match="Prior parameters for candidate .* must be a tuple/list"):
        BayesianRouter(storage=storage, contextual_priors=[{
            "pattern": "math",
            "priors": {"calculator": (10,)}
        }])

    with pytest.raises(ValueError, match="Each contextual prior must specify at least one of"):
        BayesianRouter(storage=storage, contextual_priors=[{
            "priors": {"calculator": (10, 1)}
        }])

    # 2. Setup router with regex-based prior
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
        },
        {
            "embedding": [1.0, 0.0],
            "priors": {
                "calculator": (1.0, 50.0),
                "search": (50.0, 1.0)
            }
        }
    ]
    
    embedder = MockEmbedder()
    router = BayesianRouter(
        storage=storage,
        embedder=embedder,
        contextual_priors=contextual_priors,
        similarity_threshold=0.85
    )

    # Test Regex Match
    prior_calc_alpha, prior_calc_beta = router.get_prior("solve a math sum", "calculator")
    assert prior_calc_alpha == 99.0
    assert prior_calc_beta == 1.0

    prior_search_alpha, prior_search_beta = router.get_prior("solve a math sum", "search")
    assert prior_search_alpha == 1.0
    assert prior_search_beta == 99.0

    # Test Reference Context Embedding Match
    prior_search_alpha2, prior_search_beta2 = router.get_prior("search for weather", "search")
    assert prior_search_alpha2 == 99.0
    assert prior_search_beta2 == 1.0

    # Test Precomputed Embedding Match
    router_embed = BayesianRouter(
        storage=InMemoryStorage(),
        embedder=embedder,
        contextual_priors=[
            {
                "embedding": [0.0, 1.0],
                "priors": {
                    "calculator": (88.0, 12.0)
                }
            }
        ]
    )
    calc_alpha, calc_beta = router_embed.get_prior("math help", "calculator")
    assert calc_alpha == 88.0
    assert calc_beta == 12.0

    # Test routing with contextual priors (Thompson sampling cold start)
    storage_clean = InMemoryStorage()
    router_clean = BayesianRouter(
        storage=storage_clean,
        embedder=embedder,
        contextual_priors=contextual_priors
    )
    chosen = router_clean.route("solve a math sum", ["calculator", "search"])
    assert chosen == "calculator"
    
    # Verify parameter seeding in storage
    key = router_clean._resolve_context_key("solve a math sum")
    alpha_stored, beta_stored = storage_clean.get_candidate_params(key, "calculator")
    assert alpha_stored == 99.0
    assert beta_stored == 1.0

    # Route batch
    storage_batch = InMemoryStorage()
    router_batch = BayesianRouter(
        storage=storage_batch,
        embedder=embedder,
        contextual_priors=contextual_priors
    )
    results = router_batch.route_batch(["solve a math sum", "search for weather"], ["calculator", "search"])
    assert results == ["calculator", "search"]




