"""
Targeted tests for previously-uncovered paths in router.py and storage.py.

Focuses on:
  - Router init validation edge cases
  - storage_backend shorthand (sqlite / memory / invalid)
  - Clustering decay math precision (exact formulas)
  - feedback() with reward= vs success= branches
  - feedback_by_trace() on linear modes using context_store vector lookup
  - route / feedback on lints/linucb with diagonal_covariance=True
  - get_candidate_beliefs() on all three modes
  - route_batch / feedback_batch – clustering & linear
  - VectorContextStore edge cases (zero vector, empty store)
  - InMemoryStorage linear params (decay_and_update_linear, batch variant)
  - SQLiteStorage linear params round-trip
  - BaseStorage default batch methods (fallback implementations)
  - save_vector / load_all_vectors / save_vectors on InMemory and SQLite
  - storage_backend="memory" / "in-memory" shorthand
  - invalid mode and invalid decay_factor
  - reset_candidate_beliefs
  - process_ui_feedback + evaluate_rag_success edge cases
"""
import math
import numpy as np
import pytest
from typing import Sequence

from bayesian_cortex.router import BayesianRouter, AsyncBayesianRouter
from bayesian_cortex.storage import InMemoryStorage, AsyncInMemoryStorage, SQLiteStorage
from bayesian_cortex.embeddings import VectorContextStore
from bayesian_cortex.rag import (
    check_citation,
    calculate_faithfulness,
    evaluate_rag_success,
    process_ui_feedback,
    _map_ui_feedback_to_reward,
)


CANDIDATES = ["tool_a", "tool_b"]


# ===========================================================================
# Router init validation
# ===========================================================================

class TestRouterInitValidation:
    def test_invalid_mode_raises(self, mem_storage):
        with pytest.raises(ValueError, match="mode must be"):
            BayesianRouter(storage=mem_storage, mode="banana")

    def test_invalid_decay_factor_raises(self, mem_storage):
        with pytest.raises(ValueError, match="decay_factor"):
            BayesianRouter(storage=mem_storage, decay_factor=0.0)
        with pytest.raises(ValueError, match="decay_factor"):
            BayesianRouter(storage=mem_storage, decay_factor=1.1)

    def test_storage_backend_and_storage_conflict_raises(self, mem_storage):
        with pytest.raises(ValueError, match="Cannot specify both"):
            BayesianRouter(storage=mem_storage, storage_backend="memory")

    def test_storage_backend_memory_shorthand(self):
        router = BayesianRouter(storage_backend="memory")
        assert isinstance(router.storage, InMemoryStorage)

    def test_storage_backend_in_memory_shorthand(self):
        router = BayesianRouter(storage_backend="in-memory")
        assert isinstance(router.storage, InMemoryStorage)

    def test_storage_backend_sqlite_shorthand(self, tmp_path):
        router = BayesianRouter(
            storage_backend="sqlite",
            storage_path=str(tmp_path / "shorthand.db"),
        )
        from bayesian_cortex.storage import SQLiteStorage
        assert isinstance(router.storage, SQLiteStorage)
        router.storage.close()

    def test_storage_backend_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown storage_backend"):
            BayesianRouter(storage_backend="duckdb")

    def test_hybrid_requires_linear_mode(self, det_embedder):
        with pytest.raises(ValueError, match="Hybrid mode is only supported"):
            BayesianRouter(mode="clustering", hybrid=True, embedder=det_embedder)

    def test_lints_requires_embedder(self):
        with pytest.raises(ValueError, match="Linear bandit modes"):
            BayesianRouter(mode="lints")

    def test_secret_key_bytes_accepted(self, mem_storage):
        router = BayesianRouter(storage=mem_storage, secret_key=b"bytekey12345678")
        assert router.secret_key == b"bytekey12345678"


# ===========================================================================
# Clustering mode – decay math precision
# ===========================================================================

class TestClusteringDecayMath:
    """Verify the exact beta-binomial posterior update formula."""

    def test_success_increments_alpha(self, clustering_router, mem_storage):
        ctx = "test-task"
        clustering_router.feedback(ctx, "tool_a", success=True)
        key = clustering_router._resolve_context_key(ctx)
        alpha, beta = mem_storage.get_candidate_params(key, "tool_a")
        # Initial (1, 1), decay=0.95: alpha = max(1, 1*0.95 + 1.0) = 1.95
        assert alpha == pytest.approx(1.95)
        assert beta == pytest.approx(1.0)

    def test_failure_increments_beta(self, clustering_router, mem_storage):
        ctx = "test-task-fail"
        clustering_router.feedback(ctx, "tool_a", success=False)
        key = clustering_router._resolve_context_key(ctx)
        alpha, beta = mem_storage.get_candidate_params(key, "tool_a")
        # Initial (1, 1): alpha = max(1, 1*0.95 + 0) = 1.0, beta = max(1, 1*0.95 + 1.0) = 1.95
        assert alpha == pytest.approx(1.0)
        assert beta == pytest.approx(1.95)

    def test_partial_reward(self, clustering_router, mem_storage):
        ctx = "reward-task"
        clustering_router.feedback(ctx, "tool_a", reward=0.7)
        key = clustering_router._resolve_context_key(ctx)
        alpha, beta = mem_storage.get_candidate_params(key, "tool_a")
        # alpha = max(1, 1*0.95 + 0.7) = 1.65
        # beta  = max(1, 1*0.95 + 0.3) = 1.25
        assert alpha == pytest.approx(1.65)
        assert beta == pytest.approx(1.25)

    def test_decay_clamps_to_one(self, mem_storage):
        """Under aggressive decay the clamp prevents U-shaped Beta distribution."""
        router = BayesianRouter(storage=mem_storage, decay_factor=0.01)
        ctx = "clamp-task"
        # Seed a high alpha
        router.feedback(ctx, "tool_a", reward=1.0)
        router.feedback(ctx, "tool_a", reward=1.0)
        # Now apply aggressive decay with reward=0 – alpha must not drop below 1
        for _ in range(5):
            router.feedback(ctx, "tool_a", reward=0.0)
        key = router._resolve_context_key(ctx)
        alpha, beta = mem_storage.get_candidate_params(key, "tool_a")
        assert alpha >= 1.0
        assert beta >= 1.0

    def test_route_missing_context_or_candidates_raises(self, clustering_router):
        with pytest.raises(ValueError, match="Must provide either"):
            clustering_router.route(candidates=["a"])
        with pytest.raises(ValueError, match="Must provide 'candidates'"):
            clustering_router.route(context_text="ctx")
        with pytest.raises(ValueError, match="Candidates list cannot be empty"):
            clustering_router.route(context_text="ctx", candidates=[])

    def test_feedback_missing_args_raises(self, clustering_router):
        with pytest.raises(ValueError, match="Must provide either"):
            clustering_router.feedback(candidate_name="tool_a", success=True)
        with pytest.raises(ValueError, match="Must provide a candidate_name"):
            clustering_router.feedback(context_text="ctx", success=True)
        with pytest.raises(ValueError, match="Either 'success' or 'reward'"):
            clustering_router.feedback(context_text="ctx", candidate_name="tool_a")


# ===========================================================================
# feedback_by_trace – linear modes context-vector lookup
# ===========================================================================

class TestFeedbackByTraceLookup:
    """
    When feedback_by_trace is called on a linear bandit, the router must
    retrieve the stored context embedding from the vector store.
    """

    def test_lints_feedback_by_trace_uses_stored_vector(self, lints_router):
        _, trace_id = lints_router.route_with_trace("solve math equation", CANDIDATES)
        # Should not raise even though we're now in linear mode
        result = lints_router.feedback_by_trace(trace_id, reward=1.0)
        mean, uncertainty = result
        assert isinstance(mean, float)
        assert isinstance(uncertainty, float)

    def test_linucb_feedback_by_trace_uses_stored_vector(self, linucb_router):
        _, trace_id = linucb_router.route_with_trace("solve math equation", CANDIDATES)
        mean, uncertainty = linucb_router.feedback_by_trace(trace_id, reward=1.0)
        assert isinstance(mean, float)
        assert isinstance(uncertainty, float)

    def test_feedback_by_trace_strict_raises_on_bad_trace(self, clustering_router):
        with pytest.raises(ValueError):
            clustering_router.feedback_by_trace("bad.trace", reward=1.0, strict=True)


# ===========================================================================
# Linear modes – diagonal covariance path
# ===========================================================================

class TestLinearDiagonalCovariance:
    def test_lints_diagonal_route_and_feedback(self, mem_storage, det_embedder):
        router = BayesianRouter(
            storage=mem_storage,
            embedder=det_embedder,
            mode="lints",
            exploration_weight=0.1,
            diagonal_covariance=True,
        )
        for _ in range(8):
            name, trace = router.route_with_trace("math equation", CANDIDATES)
            router.feedback_by_trace(trace, reward=1.0 if name == "tool_a" else 0.0)
        choices = [router.route("math equation", CANDIDATES) for _ in range(5)]
        assert "tool_a" in choices

    def test_linucb_diagonal_route_and_feedback(self, mem_storage, det_embedder):
        router = BayesianRouter(
            storage=mem_storage,
            embedder=det_embedder,
            mode="linucb",
            exploration_weight=0.5,
            diagonal_covariance=True,
        )
        for _ in range(8):
            name, trace = router.route_with_trace("math equation", CANDIDATES)
            router.feedback_by_trace(trace, reward=1.0 if name == "tool_a" else 0.0)
        choices = [router.route("math equation", CANDIDATES) for _ in range(5)]
        assert "tool_a" in choices

    def test_hybrid_lints_diagonal(self, mem_storage, det_embedder):
        router = BayesianRouter(
            storage=mem_storage,
            embedder=det_embedder,
            mode="lints",
            hybrid=True,
            exploration_weight=0.1,
            diagonal_covariance=True,
            candidate_embeddings={"tool_a": [1.0, 0.0], "tool_b": [0.0, 1.0]},
        )
        name, trace = router.route_with_trace("math equation", CANDIDATES)
        mean, unc = router.feedback_by_trace(trace, reward=1.0)
        assert isinstance(mean, float)


# ===========================================================================
# get_candidate_beliefs on all modes
# ===========================================================================

class TestGetCandidateBeliefs:
    def test_clustering_beliefs_returns_alpha_beta(self, clustering_router):
        clustering_router.feedback("ctx", "tool_a", success=True)
        alpha, beta = clustering_router.get_candidate_beliefs("ctx", "tool_a")
        assert alpha > 1.0   # updated
        assert beta == pytest.approx(1.0)

    def test_lints_beliefs_returns_mean_and_uncertainty(self, lints_router):
        _, trace = lints_router.route_with_trace("math equation", CANDIDATES)
        lints_router.feedback_by_trace(trace, reward=1.0)
        mean, unc = lints_router.get_candidate_beliefs("math equation", "tool_a")
        assert isinstance(mean, float)
        assert isinstance(unc, float)
        assert unc >= 0.0

    def test_hybrid_beliefs_cold_start(self, hybrid_router):
        mean, unc = hybrid_router.get_candidate_beliefs("math equation", "tool_math")
        assert isinstance(mean, float)
        assert isinstance(unc, float)

    def test_beliefs_missing_args_raises(self, clustering_router):
        with pytest.raises(ValueError):
            clustering_router.get_candidate_beliefs(candidate_name="tool_a")


# ===========================================================================
# Batch routing – clustering mode
# ===========================================================================

class TestBatchRoutingClustering:
    def test_route_batch_returns_correct_length(self, clustering_router):
        contexts = ["ctx1", "ctx2", "ctx3"]
        results = clustering_router.route_batch(contexts, CANDIDATES)
        assert len(results) == 3
        assert all(r in CANDIDATES for r in results)

    def test_route_batch_empty_contexts(self, clustering_router):
        assert clustering_router.route_batch([], CANDIDATES) == []

    def test_feedback_batch_trace_and_text_mixed(self, clustering_router, mem_storage):
        contexts = ["hello world", "style cleanup"]
        traces = clustering_router.route_batch_with_trace(contexts, CANDIDATES)
        feedbacks = [
            {"trace_id": traces[0][1], "success": True},
            {
                "context_text": "style cleanup",
                "candidate_name": traces[1][0],
                "reward": 0.0,
            },
        ]
        clustering_router.feedback_batch(feedbacks)
        # Verify trace 0's params were updated
        ctx_key, cand = clustering_router._decode_trace_id(traces[0][1])
        alpha, beta = mem_storage.get_candidate_params(ctx_key, cand)
        assert alpha > 1.0


# ===========================================================================
# Batch routing – linear modes
# ===========================================================================

class TestBatchRoutingLinear:
    def test_lints_batch_route_and_feedback(self, lints_router):
        contexts = ["math problem", "general query"]
        traces = lints_router.route_batch_with_trace(contexts, CANDIDATES)
        assert len(traces) == 2
        feedbacks = [
            {"trace_id": traces[0][1], "reward": 1.0},
            {"trace_id": traces[1][1], "reward": 0.0},
        ]
        lints_router.feedback_batch(feedbacks)

    def test_linucb_batch_route_and_text_feedback(self, linucb_router):
        contexts = ["equation solving", "web search"]
        traces = linucb_router.route_batch_with_trace(contexts, CANDIDATES)
        assert len(traces) == 2
        feedbacks = [
            {"context_text": "equation solving", "candidate_name": traces[0][0], "reward": 1.0},
            {"context_text": "web search", "candidate_name": traces[1][0], "reward": 0.0},
        ]
        linucb_router.feedback_batch(feedbacks)


# ===========================================================================
# VectorContextStore edge cases
# ===========================================================================

class TestVectorContextStoreEdgeCases:
    def test_empty_store_returns_none(self):
        store = VectorContextStore()
        assert store.get_nearest_context([1.0, 0.0]) is None

    def test_zero_query_vector_returns_none(self):
        store = VectorContextStore()
        store.add_context("ctx_a", [1.0, 0.0])
        assert store.get_nearest_context([0.0, 0.0]) is None

    def test_zero_stored_vector_skipped(self):
        store = VectorContextStore()
        store.add_context("ctx_zero", [0.0, 0.0])
        store.add_context("ctx_real", [1.0, 0.0])
        result = store.get_nearest_context([1.0, 0.0], similarity_threshold=0.9)
        assert result == "ctx_real"

    def test_below_threshold_returns_none(self):
        store = VectorContextStore()
        store.add_context("ctx_a", [1.0, 0.0])
        # Orthogonal vector – similarity = 0.0
        assert store.get_nearest_context([0.0, 1.0], similarity_threshold=0.5) is None

    def test_json_round_trip(self):
        store = VectorContextStore()
        store.add_context("ctx_a", [0.5, 0.5])
        store.add_context("ctx_b", [1.0, 0.0])
        restored = VectorContextStore.from_json(store.to_json())
        assert restored.get_nearest_context([1.0, 0.0], 0.9) == "ctx_b"

    def test_get_context_vector_missing_returns_none(self):
        store = VectorContextStore()
        assert store.get_context_vector("missing_key") is None


# ===========================================================================
# InMemoryStorage – linear param updates
# ===========================================================================

class TestInMemoryStorageLinear:
    def test_decay_and_update_linear_diagonal(self):
        storage = InMemoryStorage()
        x = np.array([1.0, 0.0, 1.0], dtype=np.float32)
        p1, r1 = storage.decay_and_update_linear("cand_a", 1.0, 1.0, x, 1.0, 0.5, diagonal=True)
        assert p1 is not None
        # Second update should decay and accumulate
        p2, r2 = storage.decay_and_update_linear("cand_a", 0.9, 0.0, x, 1.0, 0.5, diagonal=True)
        assert np.all(p2 > 0)

    def test_decay_and_update_linear_full_matrix(self):
        storage = InMemoryStorage()
        x = np.array([1.0, 0.0, 1.0], dtype=np.float32)
        p1, r1 = storage.decay_and_update_linear("cand_b", 1.0, 1.0, x, 1.0, 0.5, diagonal=False)
        assert p1.shape == (3, 3)
        # Verify it's positive definite (all eigenvalues > 0)
        eigvals = np.linalg.eigvalsh(p1)
        assert np.all(eigvals > 0)

    def test_get_linear_params_returns_none_when_missing(self):
        storage = InMemoryStorage()
        p, r = storage.get_linear_params("nonexistent")
        assert p is None
        assert r is None

    def test_get_linear_params_returns_copy_not_reference(self):
        storage = InMemoryStorage()
        x = np.array([1.0, 0.0, 1.0], dtype=np.float32)
        storage.decay_and_update_linear("cand", 1.0, 1.0, x, 1.0, 0.5, diagonal=True)
        p1, _ = storage.get_linear_params("cand")
        p1[:] = 999.0  # mutate returned copy
        p2, _ = storage.get_linear_params("cand")
        assert not np.all(p2 == 999.0)  # internal state should be unaffected

    def test_decay_and_update_linear_batch(self):
        storage = InMemoryStorage()
        x = np.array([1.0, 0.0, 1.0], dtype=np.float32)
        updates = [
            ("cand_x", 1.0, 1.0, x, 1.0, 0.5, True),
            ("cand_x", 0.9, 0.0, x, 1.0, 0.5, True),
        ]
        results = storage.decay_and_update_linear_batch(updates)
        assert len(results) == 2
        for p, r in results:
            assert p is not None


# ===========================================================================
# SQLiteStorage – linear params round-trip
# ===========================================================================

class TestSQLiteStorageLinear:
    def test_linear_params_persist_and_reload(self, sqlite_storage, tmp_path):
        from bayesian_cortex.storage import SQLiteStorage
        x = np.array([1.0, 0.5, 1.0], dtype=np.float32)
        sqlite_storage.decay_and_update_linear("tool_x", 1.0, 1.0, x, 1.0, 0.5, diagonal=True)
        # Open a fresh connection to the same file
        db2 = SQLiteStorage(db_path=str(tmp_path / "test.db"))
        p, r = db2.get_linear_params("tool_x")
        assert p is not None
        assert r is not None
        db2.close()

    def test_linear_params_full_matrix_persist(self, sqlite_storage, tmp_path):
        from bayesian_cortex.storage import SQLiteStorage
        x = np.array([1.0, 0.0, 1.0], dtype=np.float32)
        sqlite_storage.decay_and_update_linear("tool_y", 1.0, 1.0, x, 1.0, 0.5, diagonal=False)
        db2 = SQLiteStorage(db_path=str(tmp_path / "test.db"))
        p, r = db2.get_linear_params("tool_y")
        assert p.shape == (3, 3)
        db2.close()


# ===========================================================================
# BaseStorage fallback batch implementations
# ===========================================================================

class TestBaseStorageFallbackBatch:
    """
    BaseStorage provides default batch methods that call the single-item
    methods in a loop. Verify they work correctly on InMemoryStorage
    (which doesn't override them at the base level for all paths).
    """

    def test_base_get_candidate_params_batch_fallback(self):
        # Create a minimal storage that doesn't override the batch methods
        from bayesian_cortex.storage import BaseStorage
        storage = InMemoryStorage()
        storage.update_candidate_params("ctx_x", "tool_a", 5.0, 3.0)
        keys = [("ctx_x", "tool_a"), ("ctx_x", "tool_b")]
        # Call the BaseStorage-level batch method directly
        result = BaseStorage.get_candidate_params_batch(storage, keys)
        assert result[("ctx_x", "tool_a")] == (5.0, 3.0)
        assert result[("ctx_x", "tool_b")] == (1.0, 1.0)

    def test_base_save_vectors_fallback(self):
        from bayesian_cortex.storage import BaseStorage
        storage = InMemoryStorage()
        BaseStorage.save_vectors(storage, {"k1": [1.0, 2.0], "k2": [3.0, 4.0]})
        vecs = storage.load_all_vectors()
        assert vecs["k1"] == [1.0, 2.0]
        assert vecs["k2"] == [3.0, 4.0]


# ===========================================================================
# RAG helpers – edge cases
# ===========================================================================

class TestRAGHelpers:
    def test_check_citation_empty_response_false(self):
        assert check_citation("") is False

    def test_check_citation_no_fallback_true(self):
        assert check_citation("Our policy allows 5 days rollover.") is True

    def test_check_citation_matches_pattern_false(self):
        assert check_citation("Sorry, I could not find the documentation.") is False

    def test_calculate_faithfulness_empty_response(self):
        assert calculate_faithfulness("", ["some source text"]) == 0.0

    def test_calculate_faithfulness_perfect_overlap(self):
        source = "The employee gets four weeks vacation annually."
        response = "Employees receive four weeks of vacation annually."
        score = calculate_faithfulness(response, [source])
        assert score > 0.5

    def test_calculate_faithfulness_no_overlap(self):
        source = "The policy covers parental leave."
        response = "Quantum physics describes subatomic particles."
        score = calculate_faithfulness(response, [source])
        assert score < 0.5

    def test_calculate_faithfulness_string_source(self):
        score = calculate_faithfulness("four weeks vacation", "four weeks vacation allowed")
        assert score > 0.0

    def test_evaluate_rag_success_combined(self):
        source = "Employees get 4 weeks paid vacation yearly."
        good_response = "Employees receive four weeks of paid vacation annually."
        bad_response = "I don't know the answer."
        assert evaluate_rag_success(good_response, [source]) is True
        assert evaluate_rag_success(bad_response, [source]) is False

    def test_map_ui_feedback_valid_strings(self):
        assert _map_ui_feedback_to_reward("thumbs_up") == 1.0
        assert _map_ui_feedback_to_reward("thumbs_down") == 0.0
        assert _map_ui_feedback_to_reward("like") == 1.0
        assert _map_ui_feedback_to_reward("dislike") == 0.0
        assert _map_ui_feedback_to_reward("yes") == 1.0
        assert _map_ui_feedback_to_reward("no") == 0.0

    def test_map_ui_feedback_bool(self):
        assert _map_ui_feedback_to_reward(True) == 1.0
        assert _map_ui_feedback_to_reward(False) == 0.0

    def test_map_ui_feedback_float_clamped(self):
        assert _map_ui_feedback_to_reward(0.75) == 0.75

    def test_map_ui_feedback_out_of_range_raises(self):
        with pytest.raises(ValueError):
            _map_ui_feedback_to_reward(1.5)

    def test_map_ui_feedback_unsupported_string_raises(self):
        with pytest.raises(ValueError):
            _map_ui_feedback_to_reward("maybe")

    def test_process_ui_feedback_routes_to_router(self, clustering_router):
        _, trace_id = clustering_router.route_with_trace("ctx", ["tool_a"])
        reward = process_ui_feedback(clustering_router, trace_id, "thumbs_up")
        assert reward == 1.0


# ===========================================================================
# Async clustering router – basic smoke tests
# ===========================================================================

@pytest.mark.anyio
async def test_async_router_route_and_feedback(async_clustering_router):
    name, trace = await async_clustering_router.aroute_with_trace("ctx", ["tool_a", "tool_b"])
    assert name in ["tool_a", "tool_b"]
    await async_clustering_router.afeedback_by_trace(trace, reward=1.0)


@pytest.mark.anyio
async def test_async_router_route_shorthand(async_clustering_router):
    name = await async_clustering_router.aroute("ctx", ["tool_a", "tool_b"])
    assert name in ["tool_a", "tool_b"]


@pytest.mark.anyio
async def test_async_router_feedback_direct(async_clustering_router):
    result = await async_clustering_router.afeedback("ctx", "tool_a", success=True)
    assert len(result) == 2


@pytest.mark.anyio
async def test_async_router_get_beliefs(async_clustering_router):
    await async_clustering_router.afeedback("ctx", "tool_a", success=True)
    beliefs = await async_clustering_router.aget_candidate_beliefs("ctx", "tool_a")
    assert len(beliefs) == 2
    assert beliefs[0] > 1.0  # alpha was updated


# ===========================================================================
# reset_candidate_beliefs
# ===========================================================================

class TestManualBeliefReset:
    def test_manual_reset_via_update_candidate_params(self, clustering_router, mem_storage):
        # Build up state
        for _ in range(5):
            clustering_router.feedback("ctx", "tool_a", success=True)
        key = clustering_router._resolve_context_key("ctx")
        alpha_before, _ = mem_storage.get_candidate_params(key, "tool_a")
        assert alpha_before > 2.0

        key = clustering_router._resolve_context_key("ctx")
        mem_storage.update_candidate_params(key, "tool_a", 1.0, 1.0)
        alpha_after, beta_after = mem_storage.get_candidate_params(key, "tool_a")
        assert alpha_after == 1.0
        assert beta_after == 1.0
