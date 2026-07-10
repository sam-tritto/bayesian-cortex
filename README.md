<p align="center">
  <img src="https://raw.githubusercontent.com/sam-tritto/bayesian-cortex/main/assets/logo.png" alt="BayesianCortex Logo" width="400">
  <br>
  <br>
  <a href="https://pypi.org/project/bayesian_cortex/"><img src="https://img.shields.io/pypi/v/bayesian_cortex.svg" alt="PyPI version"></a>
  <a href="https://github.com/sam-tritto/bayesian-cortex/actions/workflows/ci.yml"><img src="https://github.com/sam-tritto/bayesian-cortex/actions/workflows/ci.yml/badge.svg" alt="CI Status"></a>
  <a href="https://codecov.io/gh/sam-tritto/bayesian-cortex"><img src="https://codecov.io/gh/sam-tritto/bayesian-cortex/graph/badge.svg" alt="Coverage"></a>
  <a href="https://docs.astral.sh/uv/"><img src="https://img.shields.io/badge/uv-%23DE5FE9.svg?style=flat&logo=uv&logoColor=white" alt="uv"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
</p>

# BayesianCortex: Dynamic Agent Routing via Contextual Bayesian Bandits

## **What is BayesianCortex? 🧠**

AI agents are built with rigid utility belts. When you ask them to do a task, they guess which tool or prompt to use based on static logic. They don't learn from their mistakes, and they easily fall apart when things fail silently.
BayesianCortex changes that by introducing a dynamic learning loop and treats routing as a **Contextual Multi-Armed Bandit** using **Thompson Sampling** with exact conjugate updates. 

It manages what AI architects call **"The Golden Triad"** of how an agent interacts with the world, adapting on the fly:
* 🛠️ **Tools (The Hands)**: It learns which specific action (like a SQL database query vs. a web search API) successfully retrieves the data you need without throwing errors.
* 💡 **Skills (The Mind)**: It tracks which specialized prompt or workflow gets the best results, letting the agent ditch underperforming instructions that cause hallucinations.
* 📚 **Memory (The Records)**: It figures out exactly which document vault or retrieval strategy holds the true answer to a user's question, cutting down on token waste.

In short: It stops your agents from repeating the same mistakes, turning raw AI loops into self-optimizing systems that get smarter with every single run, and adapts in real time to failures, hallucinations, or shifting environment dynamics.

Stop burning tokens and install **BayesianCortex** today!

Works with [OpenAI](https://openai.com/), [Anthropic](https://www.anthropic.com/), [Google Gemini](https://gemini.google.com/), [FastMCP](https://gofastmcp.com/getting-started/welcome), and many more! You just have to plug in your own **Transport Layer**, and you're good to go.

---

## Why BayesianCortex? (The Value Proposition)

At the core of production-grade AI engineering is a fundamental architectural trade-off: **should you use a Frontier LLM as a router, or should you use a Contextual Multi-Armed Bandit?**

While adding a routing step (like embedding generation) might seem like extra overhead, the engineering and economic benefits of local contextual bandits are massive compared to alternative approaches.

---

### Comparison: Three Ways to Route an Agent

| Routing Approach | How it Works | The Core Problem |
| :--- | :--- | :--- |
| **1. The Cloud LLM Router** *(Legacy)* | Pass the user prompt, system descriptions, and schemas of all tools (e.g., 30+ tools) to Claude or GPT to select the candidate. | **Slow and Comically Expensive.** High "context tax" on every single turn. If a tool fails, it triggers a costly recursive "self-correction loop" burning frontier tokens. |
| **2. Semantic Vector Search** *(Naive)* | Embed the user prompt and query a vector database for the candidate with the closest cosine similarity. | **Blind to Real-World Performance.** Measures only if a tool *sounds* right, not if it *works*. If a tool is a perfect semantic match but its API is down, it locks the agent into an infinite failure loop. |
| **3. BayesianCortex Router** *(Optimized)* | Embed the user prompt, pass it as context ($x_t$) to a Thompson Sampling / UCB bandit math engine, and compute the candidate with the highest expected success rate. | **Decouples routing from LLMs.** Adapts dynamically using actual performance feedback, routing around failures instantly. |

---

### Core Pillars of Value

#### 1. Zero-Cost Local Embeddings
By leveraging lightweight, local embedding models (like a 100MB `bge-small-en-v1.5` or `all-MiniLM-L6-v2` via the `local-ml` extra package), developers can compute context vectors in **under 5 milliseconds for $0 API cost**. 

The bandit doesn't need a multi-billion-parameter model to understand the deep philosophy of a prompt; it just needs the embedding model to be *spatially consistent*. As long as similar user requests map to the same neighborhood in vector space, the linear bandit math (`LinUCB` or `LinTS`) will cluster and map those neighborhoods to the tools/skills that actually succeed.

#### 2. Eliminating the Multi-Turn "Retry Loop" Tax
When an agent fails in production, it usually costs a fortune due to recursive LLM self-correction loops:
1. **Turn 1**: Call LLM to route to Tool A ($0.01) --> Tool A times out.
2. **Turn 2**: Call LLM to diagnose error and try Tool B ($0.015).
3. **Turn 3**: Call LLM to process Tool B ($0.01) --> Success.
* **Total Cost**: ~$0.035 for one successful execution.

With **BayesianCortex**, if Tool A fails, the environment returns a reward of `0`. The bandit instantly mutates its internal covariance matrix for that context neighborhood. The next time a similar prompt comes in, it routes directly to Tool B *before* calling the Cloud LLM, bypassing the retry loop tax.

#### 3. True Runtime Self-Healing
If a specialized prompt template (Skill) starts causing hallucinations because a foundational model updated its weights, or if a specific RAG index gets corrupted, developers usually have to monitor logs and manually redeploy code. 

**BayesianCortex** provides automated, mathematical self-healing. It naturally shifts probability distributions away from underperforming assets based on live feedback data, optimizing itself in production without human intervention.

---

### Executive Value Proposition Summary

1. **Massive Token & Cost Reduction**: Replaces expensive cloud LLM routing calls with ultra-fast, zero-cost local embedding matrix multiplications.
2. **Deterministic Reliability**: Stops agents from getting trapped in infinite error/retry loops by mathematically routing *around* failures, rate limits, and hallucinations in real time.
3. **Contextual Optimization**: Unlike static vector search which only looks at semantic similarity, it maps context directly to **proven execution success**.

This turns your routing setup from a static heuristic into an indispensable, money-saving framework for production-grade agent architectures.

---

## Architectural Overview

BayesianCortex is decoupled from your execution layer, acting as a lightweight interceptor/middleware:

```
                       [ User Prompt ]
                              │
                              ▼
                 [ Vector Index / Embedder ]
                              │ (Retrieve Context Key)
                              ▼
                  [ Bayesian Candidate Router ]
                   ├─ Thompson Sampling / UCB
                   ├─ Fallback Key Hashing
                   └─ Fetch (α, β) or (B_a, f_a)
                              │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
     [ Selected Candidate A ]               [ Selected Candidate B ]
            │                                 │
            └────────────────┬────────────────┘
                             ▼
                     [ Execution Trace ]
                             │ (Success / Fail / Reward)
                             ▼
                 [ Decoupled Telemetry Hook ]
                             │ (Update parameters in DB)
                             ▼
                      [ Storage Cache ] (Redis, SQLite, In-Memory)
```

---

## Installation

Install using `uv` or standard pip:

```bash
# Core package (In-memory, SQLite, and Redis support)
uv pip install bayesian-cortex

# Install with local embedding support
uv pip install "bayesian-cortex[local-ml]"
```

For advanced features, ensure the following database dependencies are satisfied:
* `sqlite-vec` (Required for SQLite vector stores)
* `aiosqlite` (Required for asynchronous SQLite operations)
* `redis` (Required for Redis cache storage)
* `httpx` (Required for API-based embedders)

---

## Quick Start

### Synchronous API

By supporting both Candidates and Skills, `bayesian_cortex` manages routing uncertainty under a single unified class:

```python
from bayesian_cortex import BayesianRouter
from bayesian_cortex.embeddings import GeminiEmbedder

# 1. Initialize router using auto-configured SQLite backend
embedder = GeminiEmbedder(model_name="models/text-embedding-004")
router = BayesianRouter(
    storage_backend="sqlite",
    storage_path="bayes_cache.db",
    embedder=embedder,
    decay_factor=0.95
)

# Scenario A: Candidate Routing (deterministic, input-output bound)
chosen_candidate = router.route(
    context_key="Fetch user profile from PostgreSQL", 
    candidates=["sql_tool", "vector_tool", "graphql_tool"]
)
print(f"Routed to candidate: {chosen_candidate}")

# Provide feedback
router.feedback(
    context_key="Fetch user profile from PostgreSQL",
    candidate=chosen_candidate,
    success=True
)

# Scenario B: Skill / Prompt Routing (heuristic, workflow-bound)
chosen_skill = router.route(
    context_key="Refactor this legacy asyncio network loop", 
    candidates=["skills/async-expert", "skills/naive-coder", "skills/strict-defensive"]
)
print(f"Routed to skill prompt: {chosen_skill}")

# Provide feedback (e.g. if generated code compiles/passes unit tests)
router.feedback(
    context_key="Refactor this legacy asyncio network loop",
    candidate=chosen_skill,
    success=True
)

# Scenario C: RAG Routing (Memory: dynamic, context-dependent knowledge retrieval)
chosen_kb = router.route(
    context_key="What is our policy on parental leave rollover?",
    candidates=["rag/hr_handbook", "rag/benefits_v2_draft", "rag/general_faq"]
)
print(f"Routed to RAG index: {chosen_kb}")

# Retrieve text chunks from chosen_kb, run the LLM, and evaluate success:
from bayesian_cortex import evaluate_rag_success
retrieved_chunks = ["Parental leave rollover allows up to 5 days rollover..."]
generated_response = "Our policy allows you to roll over up to 5 days of parental leave."

success = evaluate_rag_success(
    response=generated_response,
    source_chunks=retrieved_chunks,
    faithfulness_threshold=0.5
)
router.feedback(
    context_key="What is our policy on parental leave rollover?",
    candidate=chosen_kb,
    success=success
)
```

### Asynchronous API

For asynchronous, non-blocking workflows in web applications (FastAPI, FastMCP) or multi-agent environments:

```python
import asyncio
from bayesian_cortex import AsyncBayesianRouter
from bayesian_cortex.embeddings import GeminiEmbedder

async def main():
    embedder = GeminiEmbedder(model_name="models/text-embedding-004")
    router = AsyncBayesianRouter(
        storage_backend="sqlite",
        storage_path="bayes_cache.db",
        embedder=embedder,
        decay_factor=0.95
    )

    # Async Route
    chosen_skill = await router.aroute(
        context_key="Refactor this legacy asyncio network loop", 
        candidates=["skills/async-expert", "skills/naive-coder", "skills/strict-defensive"]
    )
    print(f"Routed to: {chosen_skill}")

    # Async Feedback
    await router.afeedback(
        context_key="Refactor this legacy asyncio network loop",
        candidate=chosen_skill,
        success=True
    )

    # Async RAG Route
    chosen_kb = await router.aroute(
        context_key="What is our policy on parental leave rollover?",
        candidates=["rag/hr_handbook", "rag/benefits_v2_draft", "rag/general_faq"]
    )
    print(f"Routed to: {chosen_kb}")

    # Async RAG Feedback (using citation check & token overlap)
    from bayesian_cortex import evaluate_rag_success
    success = evaluate_rag_success(
        response="Our policy allows 5 days rollover.",
        source_chunks=["Parental leave rollover allows up to 5 days rollover..."]
    )
    await router.afeedback(
        context_key="What is our policy on parental leave rollover?",
        candidate=chosen_kb,
        success=success
    )

asyncio.run(main())
```
---

## 🧪 Testing with the Interactive Demo

To witness Thompson Sampling adapt to drifting API failure rates in real time, run the built-in simulation script:

```bash
uv run python demo.py
```
This script initializes a local SQLite bandit database, generates simulated query clusters (e.g., coding tasks, web search queries), routes them, simulates execution, updates priors, and prints ASCII sparklines showing the learning process.

<p align="center">
  <img src="https://raw.githubusercontent.com/sam-tritto/bayesian-cortex/main/assets/demo_interactive.png" alt="Manual Interactive Routing Loop" width="700"/>
</p>

---

## The Core Math Engine

To prevent runtime latency, BayesianCortex avoids heavy Markov Chain Monte Carlo (MCMC) sampling (e.g., PyMC or Stan). Instead, it uses exact closed-form updates and supports two main mathematical modes: **Context Clustering** (Beta-Binomial) and **Linear Contextual Bandits** (LinTS / LinUCB).

### 1. Context Clustering Mode (Beta-Binomial Conjugate Pair)
Each candidate $i$ in a context cluster is modeled as a Beta distribution representing the belief of its success probability:

1. **Belief Representation**: $\theta_i \sim \text{Beta}(\alpha_i, \beta_i)$.
2. **Prior (Initial State)**: $\alpha_i = 1.0, \beta_i = 1.0$ (Uniform flat prior representing total uncertainty).
3. **Thompson Sampling**: For each candidate candidate, sample a success probability:
   $$\theta_i \sim \text{Beta}(\alpha_i, \beta_i)$$
   Select the candidate with the highest sampled probability:
   $$i^* = \arg\max_{i} \theta_i$$
4. **Posterior Update (Telemetry)**:
   - **Success**: $\alpha_i \leftarrow \alpha_i + \text{reward}$
   - **Failure**: $\beta_i \leftarrow \beta_i + (1 - \text{reward})$

#### Handling Non-Stationary Environments (Drifting APIs)
If an API starts failing or degrades over time, historical successes should not dominate the routing indefinitely. BayesianCortex applies an exponential decay factor $\gamma \in (0, 1]$ on historical updates prior to adding new rewards:
$$\alpha_t = \max(1.0, \gamma \alpha_{t-1} + \text{reward})$$
$$\beta_t = \max(1.0, \gamma \beta_{t-1} + (1 - \text{reward}))$$

Both parameters are strictly clamped to a lower-bound of `1.0` to prevent the distribution from becoming U-shaped/bimodal, which stabilizes Thompson Sampling exploration.

---

### 2. Linear Contextual Bandits Mode (LinTS & LinUCB)
Instead of partitioning tasks into discrete clusters, the linear modes learn a linear relationship between the continuous task embedding space and the expected reward. Let the text embedding vector be $x \in \mathbb{R}^d$. We augment it with a bias term $x' = [x, 1.0]$ to learn prior success rates as linear offsets.

* **Linear Thompson Sampling (LinTS)**: Models the success probability parameter $\theta_a$ for candidate $a$ as a linear combination of features, $\theta_a = x'^T w_a$, where weights $w_a$ are sampled from the posterior distribution $\mathcal{N}(\hat{w}_a, v^2 B_a^{-1})$.
* **Linear UCB (LinUCB)**: Selects the candidate maximizing the upper confidence bound of the expected reward:
  $$a^* = \arg\max_a \left(x'^T \hat{w}_a + \alpha \sqrt{x'^T B_a^{-1} x'}\right)$$
  where $\hat{w}_a$ is the ridge regression estimate, $B_a$ is the precision matrix, and $\alpha$ (or $v$) is the exploration weight.
* **L2 Regularization ($\lambda$)**: Performs ridge regression shrinkage on parameters.
* **Diagonal Covariance Approximation**: Optional diagonal approximation ($O(d)$ runtime/storage) to avoid full matrix inversion ($O(d^3)$) during high-throughput execution.
* **Shared-Parameter (Hybrid) Contextual Bandits**: Maps the task context $x_c$ and the candidate's embedding $t_a$ into a single joint feature space, $x_{\text{augmented}} = [x_c, t_a, 1.0]$, and maintains a single unified weight vector $w \in \mathbb{R}^{d_{ctx} + d_{candidate} + 1}$ across all candidates. This eliminates disjoint parameter spaces and enables zero-shot generalization.

---

## Core Features & Advanced Operations

### 🔌 Persistent, Native Vector Storage (`sqlite-vec`)
To avoid loading all context vectors into memory, BayesianCortex supports native database-level vector indexing and search via the `sqlite-vec` extension:
* **Sync Store**: [SQLiteVectorStore](file:///Users/sam/Locals%20Only/bayesian-cortex/src/bayesian_cortex/embeddings.py#L670-L789)
* **Async Store**: [AsyncSQLiteVectorStore](file:///Users/sam/Locals%20Only/bayesian-cortex/src/bayesian_cortex/embeddings.py#L877-L980)

```python
from bayesian_cortex.embeddings import SQLiteVectorStore

# Creates a vec0 virtual table for cosine-distance vector matches
vector_store = SQLiteVectorStore(
    db_path="vectors.db",
    dimension=384,
    table_name="vec_context_store"
)

router = BayesianRouter(
    storage=storage,
    embedder=embedder,
    vector_store=vector_store
)
```

### 📈 Contextual Bandit Configuration (LinTS / LinUCB)
Switch from discrete clustering to linear regression-based generalization to handle continuous feature spaces:

```python
router = BayesianRouter(
    storage=storage,
    embedder=embedder,
    mode="lints",                 # "clustering", "lints", or "linucb"
    exploration_weight=0.5,       # v in LinTS, alpha in UCB
    lambda_val=1.0,               # L2 regularization parameter
    diagonal_covariance=True,      # O(d) diagonal approximation (highly recommended for performance)
)
```

### 🤝 Shared-Parameter (Hybrid) Contextual Bandits
For setups where you want to generalize learning across candidates (e.g., in a cold-start situation where a new candidate is introduced), BayesianCortex implements a **Shared-Parameter (Hybrid) Contextual Bandit**.

Instead of maintaining disjoint parameter matrices for each individual candidate, the hybrid mode learns a single unified parameter set $w \in \mathbb{R}^{d_{ctx} + d_{candidate} + 1}$ stored under a shared database key (`__shared_hybrid__`). For each routing decision:
1. Task context $x_c$ and candidate candidate embeddings $t_a$ are resolved.
2. The router builds an augmented feature vector: $x_{\text{augmented}} = [x_c, t_a, 1.0]$.
3. The routing score is computed by taking the dot product of $x_{\text{augmented}}$ with the unified shared weight vector (sampled from the posterior in LinTS, or the ridge regression estimate plus exploration bonus in LinUCB).

#### Enabling Hybrid Mode:
```python
# Define or resolve embeddings for your candidates
candidate_embeddings = {
    "math_calculator": [1.0, 0.0, 0.1],
    "python_interpreter": [0.9, 0.1, 0.2],
    "web_search": [0.0, 1.0, 0.0]
}

# Or provide string descriptions for dynamic metadata embedding
candidate_metadata = {
    "math_calculator": "Execute mathematical equations and numeric calculations",
    "python_interpreter": "Run custom Python script blocks for data analysis",
    "web_search": "Search the web for real-time information"
}

router = BayesianRouter(
    storage=storage,
    embedder=embedder,
    mode="linucb",
    hybrid=True,                       # Enable hybrid mode
    candidate_embeddings=candidate_embeddings,   # Direct embedding vectors (Optional)
    candidate_metadata=candidate_metadata,       # String descriptions to embed dynamically (Optional)
    diagonal_covariance=True
)
```

#### Dynamic Candidate Embedding Resolution
When `hybrid=True`, candidate embeddings are resolved dynamically in order of preference:
1. **Direct Vector Lookup**: Uses vectors provided in `candidate_embeddings`.
2. **Metadata Embedding**: Embeds string descriptions provided in `candidate_metadata` using the active `ContextEmbedder` / `AsyncContextEmbedder`.
3. **Fallback Embedding**: Embeds the candidate's name (`candidate_name`) as a fallback via the active embedder.

### 📦 Batch/Bulk API Support
Avoid the N-roundtrip database/network bottleneck when processing large telemetry bundles or task lists:

```python
contexts = ["Compile source code", "Format file contents"]
candidates = ["compiler_tool", "linter_tool"]

# Batch Routing
chosen_candidates = router.route_batch(contexts, candidates)

# Batch Feedback
feedbacks = [
    {"context_text": "Compile source code", "candidate_name": "compiler_tool", "success": True, "reward": 1.0},
    {"context_text": "Format file contents", "candidate_name": "linter_tool", "success": False, "reward": 0.0}
]
router.feedback_batch(feedbacks)
```
* **Database Optimization**: SQLite backends chunk parameters into sizes of 200 and wrap requests in an immediate transaction (`executemany`). Redis backends execute Lua scripts inside pipeline blocks.

### 🛡️ Tamper-Proof Signed Trace IDs
To prevent client-side reward-poisoning and tampering attacks in decoupled or asynchronous setups, BayesianCortex signs trace IDs using an HMAC-SHA256 signature.

```python
router = BayesianRouter(
    storage=storage,
    embedder=embedder,
    secret_key="my-app-secure-hmac-key" # Auto-generates random 32-byte key if omitted
)

chosen_candidate, trace_id = router.route_with_trace(context_text, candidates)
# trace_id is formatted as 'payload_b64..signature_hex'

# Automatically validates signature before updating parameters; raises ValueError if tampered
router.feedback_by_trace(trace_id=trace_id, reward=1.0)
```

### 🎯 Context-Specific Priors (Warm Starts)
Developers can override global priors and seed prior beliefs tailored to specific tasks or domains using prompt regexes, reference contexts, or precomputed embedding vectors.

```python
contextual_priors = [
    {
        "pattern": r"(?i)compile|build|code",
        "priors": {"compiler_tool": (10.0, 1.0), "search_tool": (1.0, 5.0)}
    },
    {
        "reference_context": "Retrieve medical and scientific paper abstracts",
        "similarity_threshold": 0.85,
        "priors": {"pubmed_rag": (20.0, 1.0)}
    }
]

router = BayesianRouter(
    storage=storage,
    embedder=embedder,
    contextual_priors=contextual_priors
)
```

### ⚡ High-Concurrency & High-Performance SQLite Backend
For production use-cases, the SQLite storage backends ([SQLiteStorage](file:///Users/sam/Locals%20Only/bayesian-cortex/src/bayesian_cortex/storage.py#L387) and [AsyncSQLiteStorage](file:///Users/sam/Locals%20Only/bayesian-cortex/src/bayesian_cortex/storage.py#L1642)) are built for concurrent, lock-free performance:
* **Write-Ahead Logging (WAL)**: Initialized with `PRAGMA journal_mode=WAL;` to dramatically improve read/write concurrency.
* **Busy Timeout**: Initialized with `PRAGMA busy_timeout=5000;` to handle transient write contentions.
* **Connection Pooling (`AsyncSQLiteConnectionPool`)**: `AsyncSQLiteStorage` manages an elastic pool of up to 10 concurrent database connections, returning them to the pool after operations finish, and rolling back uncommitted transactions automatically.
* **Exponential Backoff & Jitter**: All database operations in `AsyncSQLiteStorage` are wrapped in an `_execute_with_retry` decorator. If a `sqlite3.OperationalError` (such as `"database is locked"`) is encountered, the operation retries with a randomized exponential backoff to ensure thread, task, and process safety.

### 🔏 Robust Hashed Exact Matching Fallbacks
When operating without an embedder, or if API embedder requests fail, the router normalizes the context (stripping whitespace) and hashes the string using SHA-256 (prefixed with `hash_`). This guarantees a short, fixed-length context key and prevents key matching fragility due to whitespace differences.

### 🧠 Automated RAG Routing & Feedback Loops (Memory)

RAG routing requires evaluating whether a given knowledge base or retrieval strategy succeeded. Because RAG fails silently (returning irrelevant noise or hallucinations rather than throwing errors), BayesianCortex provides helper utilities to automate feedback loops and handle direct user UI feedback (such as Thumbs Up/Down components).

#### 1. Automated RAG Success Metrics

Combine citation checks (checking if the LLM returned a standard fallback phrase) and token-overlap faithfulness metrics:

```python
from bayesian_cortex import evaluate_rag_success

retrieved_sources = [
    "Employees get 4 weeks of paid vacation yearly.",
    "Unused vacation days do not roll over."
]
llm_response = "Employees receive four weeks of paid vacation annually, but they do not roll over."

# Returns True if the response contains no fallback phrases (e.g., "I don't know")
# and has sufficient unique non-stopword token overlap with the sources (threshold defaults to 0.5)
success = evaluate_rag_success(
    response=llm_response,
    source_chunks=retrieved_sources,
    faithfulness_threshold=0.5
)

# Provide feedback to the router
router.feedback(
    context_key="How much vacation time do we get?",
    candidate="rag/hr_policies",
    success=success
)
```

#### 2. UI / Human-in-the-Loop Feedback (Thumbs Up / Down)

For scenarios where success is determined by the end user clicking thumbs up/down buttons on a chat UI, you can route the feedback payload directly back to the router using the `process_ui_feedback` (sync) or `aprocess_ui_feedback` (async) helpers. 

These helpers map diverse UI states (`"thumbs_up"`, `"like"`, `"dislike"`, `True`, `False`, `1`, `0`) to a `1.0` or `0.0` reward and verify the HMAC signature of the trace ID.

```python
from bayesian_cortex import process_ui_feedback

# 1. Route the query and obtain a signed trace ID (safeguards against client-side reward poisoning)
chosen_source, trace_id = router.route_with_trace(
    "How to reset credentials?", 
    candidates=["rag/it_support", "rag/security_protocols"]
)

# ... backend delivers LLM response containing trace_id to client UI ...

# 2. Receive thumbs-up click from client-side UI and process it:
process_ui_feedback(
    router=router,
    trace_id=trace_id,       # Signed trace ID from route_with_trace
    feedback_value="thumbs_up"  # Acceptable values: "thumbs_up", "thumbs_down", True, False, etc.
)
```

---

## Integrations & FastMCP Server

Optimize candidate/skill selection in Claude Code or other MCP hosts by registering a Meta-Candidate to handle dynamic routing, alongside administrative candidates to manage and monitor bandit beliefs.

You can configure and expose these endpoints using [create_mcp_server](file:///Users/sam/Locals%20Only/bayesian-cortex/src/bayesian_cortex/mcp_server.py):

```python
from bayesian_cortex.mcp_server import create_mcp_server

# Build the FastMCP server with dynamic component toggles
mcp = create_mcp_server(
    server_name="BayesianCortex",
    db_path="mcp_bandit.db",
    enable_tools=True,       # Registers execute_adaptive_action
    enable_skills=True,      # Registers get_candidate_beliefs & reset_candidate_beliefs
    enable_rag=False,        # Registers route_knowledge_base (off by default to save tokens)
    sub_tools=["local_pytest", "docker_sandbox", "fallback_api"] # Can also use 'candidates'
)
```

### 🧠 Optimizing the MCP "Context Tax"

Under the MCP specification, whenever an AI client (like Claude Code or Cursor) connects to your server, it executes a handshake that lists all tools and resources. The client then injects the full JSON Schema definition of every registered endpoint into the LLM's system prompt on every turn of the conversation.

A well-documented tool schema can consume **300 to 1,500 tokens of system context**. By using `enable_tools`, `enable_skills`, and `enable_rag` toggles, you can selectively disable capabilities you aren't using to prevent token bloat, reduce API costs, and minimize model routing confusion.

You can launch the server over `stdio` by executing:
```bash
python -m bayesian_cortex.mcp_server
```

### Registered Candidates, Skills & Resources

| Endpoint | Type | Configuration Toggle | Description |
| :--- | :--- | :--- | :--- |
| `execute_adaptive_action` | `Tool` | `enable_tools=True` | Thompson sampling/UCB routes incoming tasks to the best sub-candidate/skill candidate and automatically applies execution feedback. |
| `get_candidate_beliefs` | `Tool` | `enable_skills=True` | Retrieve current posterior $\alpha$ and $\beta$ beliefs for all candidate candidates/skills under a given context (resolving context-specific priors). |
| `reset_candidate_beliefs` | `Tool` | `enable_skills=True` | Reset the beliefs back to the default prior for a candidate/skill under a context. |
| `route_knowledge_base` | `Tool` | `enable_rag=True` | Selects the highest-yielding RAG index source/strategy for a semantic query. |
| `cortex://metrics` | `Resource` | Always Active | Exposes a Markdown Dashboard with context clusters, expected success rates, and raw telemetry metrics (dynamically filtered based on active flags). |

### 🛠️ Host Integration Configuration

Configure your agent or host to run the MCP server. Below are standard configuration profiles for popular clients:

#### 💻 Claude Code (CLI Agent)
Claude Code supports MCP dynamically from your shell. 

* **Automatic CLI Setup:**
  ```bash
  claude mcp add bayesian-cortex python3 /path/to/bayesian-cortex/src/bayesian_cortex/mcp_server.py
  ```
  *(Make sure to replace `/path/to/bayesian-cortex` with the absolute path to your cloned directory).*

* **Manual Configuration:**
  Edit your global configuration file (usually located at `~/.claude.json`) and add:
  ```json
  {
    "mcpServers": {
      "bayesian-cortex": {
        "command": "python3",
        "args": ["/path/to/bayesian-cortex/src/bayesian_cortex/mcp_server.py"],
        "env": {
          "BAYES_DB_PATH": "/path/to/bayesian-cortex/mcp_bandit.db"
        }
      }
    }
  }
  ```

#### 👾 Antigravity & VS Code (Cursor / IDE Extensions)
If using Antigravity or a similar IDE-integrated assistant (e.g., Cursor, Roo Code, VS Code MCP settings), add the following server configuration in your extension settings:

- **Type / Transport:** `command`
- **Name:** `bayesian-cortex`
- **Command:** `python3`
- **Arguments:** `/path/to/bayesian-cortex/src/bayesian_cortex/mcp_server.py`
- **Environment Variables:** `BAYES_DB_PATH=/path/to/bayesian-cortex/mcp_bandit.db`

#### 🖥️ Claude Desktop
On macOS, Claude Desktop configures its MCP servers via `~/Library/Application Support/Claude/claude_desktop_config.json` (on Windows: `%APPDATA%/Claude/claude_desktop_config.json`).

Add the following to your config:
```json
{
  "mcpServers": {
    "bayesian-cortex": {
      "command": "python3",
      "args": [
        "/path/to/bayesian-cortex/src/bayesian_cortex/mcp_server.py"
      ],
      "env": {
        "BAYES_DB_PATH": "/path/to/bayesian-cortex/mcp_bandit.db"
      }
    }
  }
}
```

### 📈 Visual Diagnostics on the Metrics Dashboard (`cortex://metrics`)
The `cortex://metrics` dashboard exposes rich, live visuals to monitor routing decisions and distributions in real time:
* **ASCII Sparklines**: Displays inline unicode block characters (e.g. ` ▂▃▅▇█▆▄▂`) representing the shape of the $\text{Beta}(\alpha, \beta)$ probability distribution next to each candidate/skill in the context clusters table.
* **Beta PDF SVG Charts**: Renders custom inline SVG charts mapping probability density curves for all candidate candidates/skills under each context cluster (utilizing SciPy's Beta stats model), complete with colors, legends, labels, and coordinate grids.
* **Recent Executions Log**: Lists the 20 most recent routing executions chronologically, detailing the Trace ID, Timestamp, Context Cluster, Selected Candidate/Skill, and Reward feedback outcome.
* **History MA10 SVG Line Chart**: Renders a chronological line plot tracking the running moving average success rates of candidate candidates/skills over time.

<p align="center">
  <img src="https://raw.githubusercontent.com/sam-tritto/bayesian-cortex/main/assets/demo_dashboard.png" alt="Bayesian Cortex Beliefs Dashboard" width="700"/>
</p>

#### How to open the dashboard:
* **Using your Agent:** Ask your agent: *"Read the resource `cortex://metrics`"*
* **Using a GUI Client (e.g., Cursor/Claude Desktop):** Look at the **Resources** pane or icon in the chat interface and click on `cortex://metrics` to open the live view.

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a full history of releases.

---

## License

MIT
