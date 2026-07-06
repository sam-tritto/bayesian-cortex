import asyncio
import numpy as np
import pytest
from typing import Sequence

from bayesian_cortex.embeddings import VectorContextStore, AsyncVectorContextStore
from bayesian_cortex.router import BayesianRouter, AsyncBayesianRouter
from bayesian_cortex.storage import InMemoryStorage, SQLiteStorage, AsyncInMemoryStorage, AsyncSQLiteStorage


class SimpleMockEmbedder:
    """Mock embedder returning simple 2D vectors."""
    def embed_query(self, text: str) -> Sequence[float]:
        text_lower = text.lower()
        if "math" in text_lower or "calculator" in text_lower:
            return [1.0, 0.0]
        return [0.0, 1.0]

    async def aembed_query(self, text: str) -> Sequence[float]:
        return self.embed_query(text)


def test_hybrid_initialization_validation():
    embedder = SimpleMockEmbedder()
    
    # Hybrid mode must not be supported with clustering mode
    with pytest.raises(ValueError, match="Hybrid mode is only supported with linear bandit modes"):
        BayesianRouter(mode="clustering", hybrid=True, embedder=embedder)

    # Hybrid mode requires embedder
    with pytest.raises(ValueError, match="Linear bandit modes .* require a ContextEmbedder"):
        BayesianRouter(mode="lints", hybrid=True, embedder=None)

    # Valid initialization
    router = BayesianRouter(mode="lints", hybrid=True, embedder=embedder)
    assert router.hybrid is True
    assert router.mode == "lints"


def test_hybrid_routing_and_convergence():
    embedder = SimpleMockEmbedder()
    storage = InMemoryStorage()
    
    candidate_embeddings = {
        "tool_math": [1.0, 0.0],
        "tool_other": [0.0, 1.0]
    }
    
    router = BayesianRouter(
        storage=storage,
        embedder=embedder,
        mode="linucb",
        exploration_weight=0.5,
        hybrid=True,
        candidate_embeddings=candidate_embeddings,
        diagonal_covariance=False
    )

    candidates = ["tool_math", "tool_other"]

    # Initially, route multiple times and give feedback
    for _ in range(15):
        candidate_name, trace_id = router.route_with_trace("solve math equation", candidates)
        reward = 1.0 if candidate_name== "tool_math" else 0.0
        router.feedback_by_trace(trace_id, reward=reward)

    # With training, tool_math should be chosen consistently
    final_choices = [router.route("solve math equation", candidates) for _ in range(10)]
    assert all(c == "tool_math" for c in final_choices)

    # Retrieve beliefs to check
    mean_val, uncertainty_val = router.get_candidate_beliefs("solve math equation", "tool_math")
    assert mean_val > 0.8
    assert uncertainty_val < 0.5


def test_hybrid_generalization_cold_start():
    """
    Test that a newly added tool can generalize performance instantly
    based on its embedding similarity to an already trained tool.
    """
    embedder = SimpleMockEmbedder()
    storage = InMemoryStorage()
    
    # 2D tool embeddings: dimension 0 is math-related, dimension 1 is other-related
    candidate_embeddings = {
        "tool_math": [1.0, 0.0],
        "tool_other": [0.0, 1.0],
        "tool_math_v2": [0.9, 0.1],  # very similar to tool_math
        "tool_other_v2": [0.1, 0.9],  # very similar to tool_other
    }
    
    router = BayesianRouter(
        storage=storage,
        embedder=embedder,
        mode="linucb",
        exploration_weight=0.1,
        hybrid=True,
        candidate_embeddings=candidate_embeddings,
        diagonal_covariance=False
    )

    # 1. Train on original tools (tool_math vs tool_other)
    candidate_train = ["tool_math", "tool_other"]
    for _ in range(20):
        candidate_name, trace_id = router.route_with_trace("solve math equation", candidate_train)
        reward = 1.0 if candidate_name== "tool_math" else 0.0
        router.feedback_by_trace(trace_id, reward=reward)

    # 2. Route between two NEW tools (tool_math_v2 vs tool_other_v2) without any training on them
    candidate_new = ["tool_math_v2", "tool_other_v2"]
    
    # Since tool_math_v2 is highly similar to tool_math, it should be selected immediately!
    choices = [router.route("solve math equation", candidate_new) for _ in range(5)]
    assert all(c == "tool_math_v2" for c in choices)


def test_hybrid_metadata_string_embedding():
    """Verify tool metadata string dynamic embedding support."""
    embedder = SimpleMockEmbedder()
    storage = InMemoryStorage()
    
    candidate_metadata = {
        "tool_math": "solve calculus math",
        "tool_other": "send email notifications"
    }
    
    router = BayesianRouter(
        storage=storage,
        embedder=embedder,
        mode="linucb",
        hybrid=True,
        candidate_metadata=candidate_metadata
    )

    # Verify we can route and fetch beliefs successfully
    candidate_name, trace_id = router.route_with_trace("do calculus equations", ["tool_math", "tool_other"])
    assert candidate_name in ["tool_math", "tool_other"]
    
    router.feedback_by_trace(trace_id, reward=1.0)
    
    # Check if embedding cache was populated
    assert "tool_math" in router._candidate_embedding_cache
    assert "tool_other" in router._candidate_embedding_cache


def test_hybrid_batch_routing_and_feedback():
    embedder = SimpleMockEmbedder()
    storage = InMemoryStorage()
    
    candidate_embeddings = {
        "tool_math": [1.0, 0.0],
        "tool_other": [0.0, 1.0]
    }
    
    router = BayesianRouter(
        storage=storage,
        embedder=embedder,
        mode="linucb",
        hybrid=True,
        candidate_embeddings=candidate_embeddings
    )

    # Batch route
    contexts = ["math equation", "another calculus problem"]
    candidates = ["tool_math", "tool_other"]
    routes = router.route_batch_with_trace(contexts, candidates)
    assert len(routes) == 2
    
    # Batch feedback
    feedbacks = [
        {"trace_id": routes[0][1], "success": True},
        {"trace_id": routes[1][1], "success": True}
    ]
    router.feedback_batch(feedbacks)


@pytest.mark.anyio
async def test_async_hybrid_routing_and_feedback(tmp_path):
    db_file = tmp_path / "test_async_hybrid.db"
    storage = AsyncSQLiteStorage(db_path=str(db_file))
    embedder = SimpleMockEmbedder()
    
    candidate_embeddings = {
        "tool_math": [1.0, 0.0],
        "tool_other": [0.0, 1.0]
    }
    
    router = AsyncBayesianRouter(
        storage=storage,
        embedder=embedder,
        mode="linucb",
        hybrid=True,
        candidate_embeddings=candidate_embeddings,
        diagonal_covariance=True
    )

    candidates = ["tool_math", "tool_other"]

    # Route and feedback asynchronous
    candidate_name, trace_id = await router.aroute_with_trace("do math algebra", candidates)
    await router.afeedback_by_trace(trace_id, reward=1.0)

    # Retrieve async beliefs
    mean_val, uncertainty_val = await router.aget_candidate_beliefs("do math algebra", "tool_math")
    assert mean_val > 0.5

    # Async batch route and feedback
    contexts = ["do algebra", "do trigonometry"]
    routes = await router.aroute_batch_with_trace(contexts, candidates)
    assert len(routes) == 2

    feedbacks = [
        {"trace_id": routes[0][1], "reward": 1.0},
        {"trace_id": routes[1][1], "reward": 1.0}
    ]
    await router.afeedback_batch(feedbacks)

    await storage.close()
