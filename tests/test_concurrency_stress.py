"""
Infrastructure Concurrency & Stress Testing for Bayesian Cortex.

Tests that AsyncSQLiteStorage connection pooling, WAL mode, and randomized
exponential backoff retry decorator successfully handle extreme concurrency
(1,000 simultaneous asynchronous operations) without losing updates or corrupting data.
"""

import asyncio
import os
import tempfile
from typing import List

import numpy as np
import pytest

from bayesian_cortex.router import AsyncBayesianRouter
from bayesian_cortex.storage import AsyncSQLiteStorage


class SimpleMockEmbedder:
    """Mock embedder returning a deterministic 3-dimensional vector."""

    def __init__(self, dimension: int = 3):
        self.dimension = dimension

    def embed_query(self, text: str) -> List[float]:
        # Generate simple distinct vectors based on hash of text
        val = float(hash(text) % 100) / 100.0
        return [val, val * 0.5, 1.0 - val]

    async def aembed_query(self, text: str) -> List[float]:
        return self.embed_query(text)


@pytest.mark.anyio
async def test_sqlite_concurrency_stress_clustering():
    """
    Fire 1,000 concurrent clustering (Beta-Binomial) route and feedback requests
    against a local SQLite database to force write lock conflicts.
    Asserts that zero transactions are lost or failed.
    """
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    try:
        # Max connections = 5 and a short timeout to aggressively trigger locks and exercise retry logic
        storage = AsyncSQLiteStorage(db_path=db_path, max_connections=5, timeout=2.0)

        router = AsyncBayesianRouter(
            storage=storage,
            mode="clustering",
            decay_factor=1.0,  # 1.0 to ensure exact summing of updates
            embedder=None,
        )

        candidates = ["cand_0", "cand_1", "cand_2"]

        async def run_worker(worker_id: int):
            # Alternate contexts to test concurrent updates on multiple rows
            context_text = f"context_{worker_id % 3}"
            chosen, trace_id = await router.aroute_with_trace(context_text, candidates)
            await router.afeedback_by_trace(trace_id, reward=1.0)

        # Launch 1,000 concurrent operations
        workers = [run_worker(i) for i in range(1000)]
        await asyncio.gather(*workers)

        # Verify math correctness: total increment across all alphas must be exactly 1000
        total_alpha_diff = 0.0
        for ctx_id in range(3):
            context_key = f"context_{ctx_id}"
            resolved_key = router._hash_context_text(context_key)
            for cand in candidates:
                alpha, beta = await storage.get_candidate_params(resolved_key, cand)
                total_alpha_diff += alpha - 1.0

        assert (
            total_alpha_diff == 1000.0
        ), f"Expected total alpha difference of 1000.0, got {total_alpha_diff}"

        await storage.close()

    finally:
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except Exception:
                pass
        for suffix in ["-wal", "-shm"]:
            wal_file = db_path + suffix
            if os.path.exists(wal_file):
                try:
                    os.remove(wal_file)
                except Exception:
                    pass


@pytest.mark.anyio
async def test_sqlite_concurrency_stress_linear():
    """
    Fire 1,000 concurrent linear (LinTS) route and feedback requests
    against a local SQLite database to force write lock conflicts.
    Asserts that covariance matrices and regression vectors remain stable.
    """
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    try:
        storage = AsyncSQLiteStorage(db_path=db_path, max_connections=5, timeout=2.0)
        embedder = SimpleMockEmbedder(dimension=3)

        router = AsyncBayesianRouter(
            storage=storage,
            embedder=embedder,
            mode="lints",
            decay_factor=1.0,
            diagonal_covariance=True,
            exploration_weight=0.5,
            lambda_val=1.0,
            similarity_threshold=0.0,  # Maps all contexts to the same context key
        )

        candidates = ["cand_0", "cand_1", "cand_2"]

        async def run_worker(worker_id: int):
            context_text = f"context_{worker_id % 3}"
            chosen, trace_id = await router.aroute_with_trace(context_text, candidates)
            await router.afeedback_by_trace(trace_id, reward=1.0)

        # Launch 1,000 concurrent operations
        workers = [run_worker(i) for i in range(1000)]
        await asyncio.gather(*workers)

        # Verify final parameters
        for cand in candidates:
            precision, reward_vector = await storage.aget_linear_params(cand)
            if precision is not None:
                assert precision.shape == (4,)  # 3 features + 1 intercept
                assert reward_vector.shape == (4,)
                assert np.all(np.isfinite(precision))
                assert np.all(np.isfinite(reward_vector))
                assert np.all(precision >= 1.0)

        await storage.close()

    finally:
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except Exception:
                pass
        for suffix in ["-wal", "-shm"]:
            wal_file = db_path + suffix
            if os.path.exists(wal_file):
                try:
                    os.remove(wal_file)
                except Exception:
                    pass
