import asyncio
import numpy as np
import pytest
from typing import Sequence

from bayes_brain.embeddings import VectorContextStore, AsyncVectorContextStore
from bayes_brain.router import BayesianToolRouter, AsyncBayesianToolRouter
from bayes_brain.storage import InMemoryStorage, SQLiteStorage, AsyncInMemoryStorage, AsyncSQLiteStorage


class SimpleMockEmbedder:
    """Mock embedder returning simple 2D vectors."""
    def embed_query(self, text: str) -> Sequence[float]:
        if "math" in text.lower():
            return [1.0, 0.0]
        return [0.0, 1.0]

    async def aembed_query(self, text: str) -> Sequence[float]:
        return self.embed_query(text)


def test_linear_initialization_validation():
    # Linear modes require an embedder
    with pytest.raises(ValueError, match="Linear bandit modes .* require a ContextEmbedder"):
        BayesianToolRouter(mode="lints", embedder=None)

    with pytest.raises(ValueError, match="Linear bandit modes .* require a ContextEmbedder"):
        BayesianToolRouter(mode="linucb", embedder=None)

    # Valid initialization
    embedder = SimpleMockEmbedder()
    router = BayesianToolRouter(mode="lints", embedder=embedder)
    assert router.mode == "lints"

    router_ucb = BayesianToolRouter(mode="linucb", embedder=embedder, diagonal_covariance=True)
    assert router_ucb.mode == "linucb"
    assert router_ucb.diagonal_covariance is True


def test_lints_routing_and_convergence():
    embedder = SimpleMockEmbedder()
    storage = InMemoryStorage()
    router = BayesianToolRouter(
        storage=storage,
        embedder=embedder,
        mode="lints",
        exploration_weight=0.1,
        diagonal_covariance=False
    )

    candidate_tools = ["tool_math", "tool_other"]

    # Initially, both tools should be routed occasionally
    choices = [router.route("solve math equation", candidate_tools) for _ in range(10)]
    assert len(choices) == 10

    # Simulate feedback: tool_math always succeeds (reward=1.0) on math queries, tool_other always fails (reward=0.0)
    for _ in range(15):
        tool, trace_id = router.route_with_trace("solve math equation", candidate_tools)
        reward = 1.0 if tool == "tool_math" else 0.0
        router.feedback_by_trace(trace_id, reward=reward)

    # After feedback, tool_math should be consistently chosen for math queries
    final_choices = [router.route("solve math equation", candidate_tools) for _ in range(10)]
    assert all(c == "tool_math" for c in final_choices)

    # Expected reward for tool_math under math context should be close to 1.0, uncertainty should be small
    mean_val, uncertainty_val = router.get_tool_beliefs("solve math equation", "tool_math")
    assert mean_val > 0.8
    assert uncertainty_val < 0.5


def test_lints_diagonal_covariance_routing():
    embedder = SimpleMockEmbedder()
    storage = InMemoryStorage()
    router = BayesianToolRouter(
        storage=storage,
        embedder=embedder,
        mode="lints",
        exploration_weight=0.1,
        diagonal_covariance=True
    )

    candidate_tools = ["tool_math", "tool_other"]

    # Train diagonal covariance model
    for _ in range(15):
        tool, trace_id = router.route_with_trace("solve math equation", candidate_tools)
        reward = 1.0 if tool == "tool_math" else 0.0
        router.feedback_by_trace(trace_id, reward=reward)

    final_choices = [router.route("solve math equation", candidate_tools) for _ in range(10)]
    assert all(c == "tool_math" for c in final_choices)


def test_linucb_routing_and_convergence():
    embedder = SimpleMockEmbedder()
    storage = InMemoryStorage()
    router = BayesianToolRouter(
        storage=storage,
        embedder=embedder,
        mode="linucb",
        exploration_weight=0.5,
        diagonal_covariance=False
    )

    candidate_tools = ["tool_math", "tool_other"]

    # Train LinUCB
    for _ in range(15):
        tool, trace_id = router.route_with_trace("solve math equation", candidate_tools)
        reward = 1.0 if tool == "tool_math" else 0.0
        router.feedback_by_trace(trace_id, reward=reward)

    # LinUCB is deterministic given parameters, so it should consistently select the best tool now
    final_choices = [router.route("solve math equation", candidate_tools) for _ in range(5)]
    assert all(c == "tool_math" for c in final_choices)


def test_linucb_diagonal_covariance():
    embedder = SimpleMockEmbedder()
    storage = InMemoryStorage()
    router = BayesianToolRouter(
        storage=storage,
        embedder=embedder,
        mode="linucb",
        exploration_weight=0.5,
        diagonal_covariance=True
    )

    candidate_tools = ["tool_math", "tool_other"]

    # Train LinUCB with diagonal covariance
    for _ in range(15):
        tool, trace_id = router.route_with_trace("solve math equation", candidate_tools)
        reward = 1.0 if tool == "tool_math" else 0.0
        router.feedback_by_trace(trace_id, reward=reward)

    final_choices = [router.route("solve math equation", candidate_tools) for _ in range(5)]
    assert all(c == "tool_math" for c in final_choices)


def test_linear_sqlite_storage(tmp_path):
    db_file = tmp_path / "test_linear.db"
    storage = SQLiteStorage(db_path=str(db_file))
    embedder = SimpleMockEmbedder()
    
    router = BayesianToolRouter(
        storage=storage,
        embedder=embedder,
        mode="lints",
        exploration_weight=0.2,
    )
    
    # Run route and feedback to trigger updates in SQLite
    tool, trace_id = router.route_with_trace("solve math equation", ["t1", "t2"])
    router.feedback_by_trace(trace_id, reward=1.0)
    
    # Reload from same DB file to ensure storage persistence
    storage2 = SQLiteStorage(db_path=str(db_file))
    router2 = BayesianToolRouter(
        storage=storage2,
        embedder=embedder,
        mode="lints",
        exploration_weight=0.2,
    )
    
    p1, r1 = storage2.get_linear_params("t1")
    p2, r2 = storage2.get_linear_params("t2")
    
    assert p1 is not None or p2 is not None
    
    storage.close()
    storage2.close()


@pytest.mark.anyio
async def test_async_lints_routing_and_feedback(tmp_path):
    db_file = tmp_path / "test_async_linear.db"
    storage = AsyncSQLiteStorage(db_path=str(db_file))
    embedder = SimpleMockEmbedder()
    
    router = AsyncBayesianToolRouter(
        storage=storage,
        embedder=embedder,
        mode="lints",
        exploration_weight=0.1,
    )
    
    candidate_tools = ["tool_math", "tool_other"]
    
    # Route and feedback multiple times asynchronously
    for _ in range(10):
        tool, trace_id = await router.aroute_with_trace("solve math equation", candidate_tools)
        reward = 1.0 if tool == "tool_math" else 0.0
        await router.afeedback_by_trace(trace_id, reward=reward)
        
    final_choices = []
    for _ in range(5):
        tool = await router.aroute("solve math equation", candidate_tools)
        final_choices.append(tool)
        
    # Since tool_math always succeeds, it should start dominating
    assert "tool_math" in final_choices
    
    await storage.close()
