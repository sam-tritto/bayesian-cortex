"""
Shared pytest fixtures for BayesianCortex tests.
"""

from typing import Sequence

import pytest

from bayesian_cortex.router import AsyncBayesianRouter, BayesianRouter
from bayesian_cortex.storage import AsyncInMemoryStorage, InMemoryStorage

# ---------------------------------------------------------------------------
# Mock embedders
# ---------------------------------------------------------------------------


class DeterministicEmbedder:
    """
    Returns a 2-D unit vector based on keyword in text.
    'math'/'calc'/'sum'/'equation' -> [1, 0], everything else -> [0, 1].
    """

    def embed_query(self, text: str) -> Sequence[float]:
        if any(kw in text.lower() for kw in ("math", "calc", "sum", "equation")):
            return [1.0, 0.0]
        return [0.0, 1.0]

    async def aembed_query(self, text: str) -> Sequence[float]:
        return self.embed_query(text)

    def embed_queries(self, texts: list) -> list:
        return [self.embed_query(t) for t in texts]

    async def aembed_queries(self, texts: list) -> list:
        return self.embed_queries(texts)


class CrashingEmbedder:
    """Always raises – used to test fallback paths."""

    def embed_query(self, text: str) -> Sequence[float]:
        raise RuntimeError("Embedder offline")

    async def aembed_query(self, text: str) -> Sequence[float]:
        raise RuntimeError("Embedder offline")


# ---------------------------------------------------------------------------
# Fixtures – storage
# ---------------------------------------------------------------------------


@pytest.fixture
def mem_storage() -> InMemoryStorage:
    return InMemoryStorage()


@pytest.fixture
def async_mem_storage() -> AsyncInMemoryStorage:
    return AsyncInMemoryStorage()


@pytest.fixture
def sqlite_storage(tmp_path):
    from bayesian_cortex.storage import SQLiteStorage

    db = SQLiteStorage(db_path=str(tmp_path / "test.db"))
    yield db
    db.close()


@pytest.fixture
async def async_sqlite_storage(tmp_path):
    from bayesian_cortex.storage import AsyncSQLiteStorage

    db = AsyncSQLiteStorage(db_path=str(tmp_path / "test_async.db"))
    yield db
    await db.close()


# ---------------------------------------------------------------------------
# Fixtures – embedders
# ---------------------------------------------------------------------------


@pytest.fixture
def det_embedder() -> DeterministicEmbedder:
    return DeterministicEmbedder()


@pytest.fixture
def crashing_embedder() -> CrashingEmbedder:
    return CrashingEmbedder()


# ---------------------------------------------------------------------------
# Fixtures – pre-built routers
# ---------------------------------------------------------------------------


@pytest.fixture
def clustering_router(mem_storage) -> BayesianRouter:
    return BayesianRouter(storage=mem_storage, decay_factor=0.95)


@pytest.fixture
def clustering_router_with_embedder(mem_storage, det_embedder) -> BayesianRouter:
    return BayesianRouter(
        storage=mem_storage,
        embedder=det_embedder,
        decay_factor=1.0,
        similarity_threshold=0.85,
    )


@pytest.fixture
def lints_router(mem_storage, det_embedder) -> BayesianRouter:
    return BayesianRouter(
        storage=mem_storage,
        embedder=det_embedder,
        mode="lints",
        exploration_weight=0.1,
        diagonal_covariance=False,
    )


@pytest.fixture
def linucb_router(mem_storage, det_embedder) -> BayesianRouter:
    return BayesianRouter(
        storage=mem_storage,
        embedder=det_embedder,
        mode="linucb",
        exploration_weight=0.5,
        diagonal_covariance=False,
    )


@pytest.fixture
def hybrid_router(mem_storage, det_embedder) -> BayesianRouter:
    return BayesianRouter(
        storage=mem_storage,
        embedder=det_embedder,
        mode="linucb",
        hybrid=True,
        exploration_weight=0.5,
        candidate_embeddings={"tool_math": [1.0, 0.0], "tool_other": [0.0, 1.0]},
        diagonal_covariance=False,
    )


@pytest.fixture
def async_clustering_router(async_mem_storage) -> AsyncBayesianRouter:
    return AsyncBayesianRouter(storage=async_mem_storage, decay_factor=0.95)


# ---------------------------------------------------------------------------
# anyio backend
# ---------------------------------------------------------------------------

pytest_plugins = ("anyio",)
