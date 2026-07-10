# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- GitHub Actions CI workflow: runs the full test suite on Python 3.11 and 3.12
  on every push and pull request, with coverage uploaded to Codecov.
- GitHub Actions publish workflow: automatically builds and publishes to PyPI
  on version tags (`v*.*.*`) using OIDC trusted publishing (no API token needed).
- `conftest.py` with shared fixtures (`DeterministicEmbedder`, `CrashingEmbedder`,
  storage fixtures, and pre-built router fixtures) to eliminate copy-pasted setup
  code across test modules.
- `test_core_coverage.py`: 65 new targeted tests covering previously untested
  critical paths — decay math precision, linear feedback vector lookup,
  diagonal covariance paths, `get_candidate_beliefs` on all modes,
  batch routing/feedback, `VectorContextStore` edge cases, and linear param
  persistence round-trips through SQLite.
- Dynamic CI status and Codecov coverage badges in README (replacing the
  hardcoded `61%` badge).
- `[project.urls]` table in `pyproject.toml` (Homepage, Repository, Issues, Changelog).

### Changed
- `[dependency-groups] dev` now includes `pytest-cov>=5.0.0` and
  `coverage[toml]>=7.0.0` so coverage runs without extra setup.
- `[tool.coverage.report]` gate raised from 55% → 63% to lock in gains and
  prevent future regressions.
- `rag.py` reaches **100%** test coverage; `router.py` improved from 61% → 66%.

---

## [0.1.2] — 2026-07-07

### Changed
- Fixed installation commands in README to use hyphens (`bayesian-cortex`)
  instead of underscores in `pip install` / `uv add` examples.
- Updated `pyproject.toml` with correct dependency versions and project metadata.
- Pinned `uv_build>=0.8.17,<0.9.0` as the build backend.
- README asset links changed to absolute GitHub raw URLs so they render
  correctly on PyPI.

---

## [0.1.1] — 2026-07-06

This release constitutes the first full feature-complete version of
BayesianCortex, expanding the original Thompson Sampling prototype into a
production-grade contextual bandit library.

### Added

#### Core routing engine
- **`AsyncBayesianRouter`** — fully async counterpart to `BayesianRouter`,
  backed by `AsyncSQLiteStorage` with connection pooling and retry logic.
- **Linear Contextual Bandits** — `mode="lints"` (Thompson Sampling) and
  `mode="linucb"` (Upper Confidence Bound) with full and diagonal covariance
  matrix support via `diagonal_covariance` flag.
- **Hybrid bandit mode** (`hybrid=True`) — shared parameter space where
  candidate embeddings are concatenated with context embeddings, enabling
  zero-shot generalization to new candidates at inference time.
- **Continuous reward support** — `feedback(reward=0.73)` accepts any `float`
  in `[0.0, 1.0]`, not just binary success/failure.
- **Contextual priors** — `contextual_priors` parameter seeds Beta distribution
  parameters based on regex patterns, reference-text embedding similarity, or
  precomputed embedding vectors.
- **Batch APIs** — `route_batch`, `route_batch_with_trace`, `feedback_batch`,
  and their async counterparts (`aroute_batch`, `afeedback_batch`).
- **HMAC-signed trace IDs** — `route_with_trace` returns tamper-evident trace
  IDs; `feedback_by_trace` verifies the HMAC signature before applying reward.
  `strict=True` flag makes verification failures raise `ValueError`.
- **Fallback routing and telemetry hooks** — `fallback_candidate` and
  `telemetry_hook` parameters for graceful degradation and observability
  without hard crashes.
- **`storage_backend` shorthand** — `BayesianRouter(storage_backend="sqlite",
  storage_path="app.db")` without manually importing `SQLiteStorage`.

#### Storage backends
- **`AsyncSQLiteStorage`** — async SQLite backend with `aiosqlite`,
  WAL journal mode, 5000 ms busy timeout, per-connection pooling, and
  exponential-backoff retry decorator.
- **`AsyncInMemoryStorage`** — async in-memory backend for testing.
- **`AsyncRedisStorage`** — async Redis backend.
- **`VectorStoreProtocol`** — formal protocol for pluggable vector index
  implementations.
- **`SQLiteVectorStore`** — persistent vector store using `sqlite-vec` for
  native ANN search in SQLite.
- **Incremental vector persistence** — `save_vector` / `save_vectors` write
  individual context vectors without re-serializing the entire store; automatic
  migration from legacy JSON metadata blob to the `context_vectors` table.
- **Selection logging** — `log_selection` / `log_feedback` / `get_selection_logs`
  for full audit trail of routing decisions.

#### Embedders
- **`GeminiEmbedder`** — supports the Gemini Generative Language API via
  raw HTTP (`urllib`) for sync and `httpx` for async; accepts an optional
  pre-constructed SDK client.
- **`OpenAIEmbedder`** — supports the OpenAI Embeddings API via raw HTTP
  for sync and `httpx` for async; respects `OPENAI_BASE_URL` env var for
  OpenAI-compatible endpoints.
- **`AsyncVectorContextStore`** — async wrapper around `VectorContextStore`.

#### RAG helpers (`rag.py`)
- `check_citation(response)` — detects standard "I don't know" fallback phrases
  via configurable regex patterns.
- `calculate_faithfulness(response, source_chunks)` — token-overlap metric
  returning a `[0, 1]` score of how much of the response is grounded in sources.
- `evaluate_rag_success(response, source_chunks)` — combined check: citation
  pass **and** faithfulness above threshold.
- `process_ui_feedback` / `aprocess_ui_feedback` — maps thumbs-up/down,
  boolean, or numerical UI signals to a `reward` and calls the router.

#### MCP server (`mcp_server.py`)
- MCP tools for `route_candidate`, `submit_feedback`, `get_beliefs`,
  `route_with_context`, and `reset_beliefs`.
- SVG metrics dashboard served as an MCP resource.
- Conditional tool registration: register only the subset of tools your agent
  needs to reduce context window bloat.

### Fixed
- **Beta bimodality bug** — decayed `alpha` and `beta` parameters are now
  clamped to a minimum of `1.0`. Without the clamp, aggressive decay could
  push them below `1.0`, turning the Beta PDF into a bimodal U-shape and
  causing completely erratic routing.

### Changed
- Renamed `BayesianToolRouter` → `BayesianRouter` and
  `AsyncBayesianToolRouter` → `AsyncBayesianRouter`.
- Renamed `tool_name` → `candidate_name` throughout the public API, storage
  schema, and MCP tools for domain-agnostic naming.
- Project renamed from `bayes-brain` → `bayesian-cortex`.

---

## [0.1.0] — 2026-07-05

Initial release of `bayes-brain` (now `bayesian-cortex`).

### Added
- `BayesianRouter` — synchronous Thompson Sampling router using Beta-Binomial
  posteriors stored in `InMemoryStorage` or `SQLiteStorage`.
- `InMemoryStorage` — thread-safe in-memory storage backend.
- `SQLiteStorage` — thread-safe SQLite backend with WAL journal mode.
- `RedisStorage` — Redis backend using Lua scripts for atomic decay-and-update.
- `LocalSentenceTransformerEmbedder` — lazy-loaded local embedding model via
  `sentence-transformers`.
- `VectorContextStore` — in-memory cosine-similarity vector index for
  clustering routing contexts into named keys.
- Exact-match SHA-256 context hashing as fallback when no embedder is provided.
- `decay_factor` parameter for temporal forgetting of historical feedback.
- `priors` parameter for seeding Beta distribution parameters per candidate.

---

[Unreleased]: https://github.com/sam-tritto/bayesian-cortex/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/sam-tritto/bayesian-cortex/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/sam-tritto/bayesian-cortex/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/sam-tritto/bayesian-cortex/releases/tag/v0.1.0
