import pytest
import os
from unittest.mock import MagicMock

from bayesian_cortex import (
    BayesianRouter,
    AsyncBayesianRouter,
)
from bayesian_cortex.storage import (
    InMemoryStorage,
    SQLiteStorage,
    AsyncInMemoryStorage,
    AsyncSQLiteStorage,
)

def test_initialization_by_backend():
    # 1. Test Sync memory backend
    router_mem = BayesianRouter(storage_backend="memory")
    assert isinstance(router_mem.storage, InMemoryStorage)

    # 2. Test Sync SQLite backend
    db_path = "test_unified_sync.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    try:
        router_sql = BayesianRouter(storage_backend="sqlite", storage_path=db_path)
        assert isinstance(router_sql.storage, SQLiteStorage)
        assert router_sql.storage.db_path == db_path
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)

    # 3. Test Async memory backend
    router_async_mem = AsyncBayesianRouter(storage_backend="memory")
    assert isinstance(router_async_mem.storage, AsyncInMemoryStorage)

    # 4. Test Async SQLite backend
    db_path_async = "test_unified_async.db"
    if os.path.exists(db_path_async):
        os.remove(db_path_async)
    try:
        router_async_sql = AsyncBayesianRouter(storage_backend="sqlite", storage_path=db_path_async)
        assert isinstance(router_async_sql.storage, AsyncSQLiteStorage)
        assert router_async_sql.storage.db_path == db_path_async
    finally:
        if os.path.exists(db_path_async):
            os.remove(db_path_async)


def test_conflict_raise():
    # Specifying both storage and storage_backend should raise ValueError
    storage = InMemoryStorage()
    with pytest.raises(ValueError, match="Cannot specify both"):
        BayesianRouter(storage=storage, storage_backend="memory")


def test_flexible_parameters_routing_sync():
    router = BayesianRouter(storage_backend="memory")

    # Scenario A: Tool routing with old parameter names
    chosen_candidate = router.route(
        context_text="Fetch user profile from PostgreSQL",
        candidates=["sql_tool", "vector_tool", "graphql_tool"],
    )
    assert chosen_candidate in ["sql_tool", "vector_tool", "graphql_tool"]

    # Scenario B: Skill routing with new parameter names
    chosen_skill = router.route(
        context_key="Refactor this legacy asyncio network loop",
        candidates=["skills/async-expert", "skills/naive-coder", "skills/strict-defensive"],
    )
    assert chosen_skill in ["skills/async-expert", "skills/naive-coder", "skills/strict-defensive"]

    # Route with trace with aliases
    tool, trace = router.route_with_trace(
        context_key="Test query key",
        candidates=["toolA", "toolB"],
    )
    assert tool in ["toolA", "toolB"]
    assert trace is not None

    # Feedback with aliases: context_key, and candidate_name / skill_name / candidate
    router.feedback(context_key="Test query key", candidate_name=tool, success=True)
    router.feedback(context_key="Test query key", candidate_name=tool, reward=1.0)
    router.feedback(context_key="Test query key", candidate_name=tool, success=True)

    # Beliefs retrieval with aliases
    a1, b1 = router.get_candidate_beliefs(context_key="Test query key", candidate_name=tool)
    a2, b2 = router.get_candidate_beliefs(context_key="Test query key", candidate_name=tool)
    a3, b3 = router.get_candidate_beliefs(context_key="Test query key", candidate_name=tool)
    assert a1 == a2 == a3
    assert b1 == b2 == b3


@pytest.mark.anyio
async def test_flexible_parameters_routing_async():
    router = AsyncBayesianRouter(storage_backend="memory")

    # aroute and aroute_with_trace with aliases
    chosen_skill = await router.aroute(
        context_key="Build a Next.js application",
        candidates=["skills/frontend-wizard", "skills/basic-html"],
    )
    assert chosen_skill in ["skills/frontend-wizard", "skills/basic-html"]

    skill, trace = await router.aroute_with_trace(
        context_key="Build a Next.js application",
        candidates=["skills/frontend-wizard", "skills/basic-html"],
    )
    assert skill in ["skills/frontend-wizard", "skills/basic-html"]
    assert trace is not None

    # afeedback with aliases
    await router.afeedback(context_key="Build a Next.js application", candidate_name=skill, success=True)
    await router.afeedback(context_key="Build a Next.js application", candidate_name=skill, reward=1.0)

    # aget_candidate_beliefs with aliases
    a, b = await router.aget_candidate_beliefs(context_key="Build a Next.js application", candidate_name=skill)
    assert a >= 1.0
    assert b >= 1.0


def test_batch_aliases_sync():
    router = BayesianRouter(storage_backend="memory")

    contexts = ["Context one", "Context two"]
    candidates = ["skill_A", "skill_B"]

    # route_batch and route_batch_with_trace
    selected = router.route_batch(contexts=contexts, candidates=candidates)
    assert len(selected) == 2

    selected_traces = router.route_batch_with_trace(contexts=contexts, candidates=candidates)
    assert len(selected_traces) == 2

    # feedback_batch with dict aliases
    feedbacks = [
        {"context_key": "Context one", "skill_name": "skill_A", "success": True},
        {"context_text": "Context two", "candidate": "skill_B", "reward": 0.0},
    ]
    router.feedback_batch(feedbacks)


@pytest.mark.anyio
async def test_batch_aliases_async():
    router = AsyncBayesianRouter(storage_backend="memory")

    contexts = ["Context one", "Context two"]
    candidates = ["skill_A", "skill_B"]

    # aroute_batch and aroute_batch_with_trace
    selected = await router.aroute_batch(contexts=contexts, candidates=candidates)
    assert len(selected) == 2

    selected_traces = await router.aroute_batch_with_trace(contexts=contexts, candidates=candidates)
    assert len(selected_traces) == 2

    # afeedback_batch with dict aliases
    feedbacks = [
        {"context_key": "Context one", "skill_name": "skill_A", "success": True},
        {"context_text": "Context two", "candidate": "skill_B", "reward": 0.0},
    ]
    await router.afeedback_batch(feedbacks)
