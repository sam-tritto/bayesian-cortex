<p align="center">
  <img src="assets/logo.png" alt="BayesBrain Logo" width="400">
</p>

# BayesBrain: Dynamic Tool Routing via Bayesian Bandits

In multi-agent systems, a supervisor agent often needs to decide which specialized sub-agent or API tool to invoke to solve a specific user prompt. Traditionally, this is done using hardcoded heuristics, prompt engineering, or raw LLM classification logits. None of these handle real-time uncertainty or feedback loops well.

**BayesBrain** treats tool routing as a **Contextual Multi-Armed Bandit** using **Thompson Sampling** with exact Beta-Binomial conjugate updates. It dynamically learns the most reliable tool or configuration under semantic context clusters, adapting in real time to API failures, drift, or changing developer requirements.

---

## The Core Math Engine (Beta-Binomial Conjugate Pair)

To prevent runtime latency, BayesBrain avoids heavy Markov Chain Monte Carlo (MCMC) sampling (e.g., PyMC or Stan). Instead, it uses exact closed-form Beta-Binomial updates:

1. **Belief Representation**: Each tool $i$ is modeled as a Beta distribution: $\text{Beta}(\alpha_i, \beta_i)$.
2. **Prior (Initial State)**: $\alpha_i = 1.0, \beta_i = 1.0$ (Uniform flat distribution representing total uncertainty).
3. **Thompson Sampling**: For each candidate tool, sample a success probability:
   $$\theta_i \sim \text{Beta}(\alpha_i, \beta_i)$$
   Select the tool with the highest sampled probability:
   $$i^* = \arg\max_{i} \theta_i$$
4. **Posterior Update (Telemetry)**:
   - **Success**: $\alpha_i \leftarrow \alpha_i + \text{reward}$
   - **Failure**: $\beta_i \leftarrow \beta_i + (1 - \text{reward})$

### Handling Non-Stationary Environments (Drifting APIs)
If an API starts failing or degrades over time, historical successes should not dominate the routing indefinitely. BayesBrain applies an exponential decay factor $\gamma \in (0, 1]$ on historical updates prior to adding new rewards:
$$\alpha_t = \gamma \alpha_{t-1} + \text{reward}$$
$$\beta_t = \gamma \beta_{t-1} + (1 - \text{reward})$$

This ensures the router rapidly adapts to outages, API updates, or regressions.

### Preventing Parameter Over-Decay (Beta Bimodality)
To prevent the Beta parameters from decaying indefinitely under continuous discount cycles (which could cause the distribution parameters to drop below $1.0$, resulting in a U-shaped bimodal distribution that breaks Thompson Sampling exploration), both $\alpha$ and $\beta$ are strictly clamped to a lower-bound of `1.0`:
$$\alpha_t = \max(1.0, \gamma \alpha_{t-1} + \text{reward})$$
$$\beta_t = \max(1.0, \gamma \beta_{t-1} + (1 - \text{reward}))$$

This regularizes the bandit beliefs and is natively implemented across all storage backends (including within Redis Lua scripts).

---

## Architectural Overview

BayesBrain is decoupled from your execution layer, acting as a lightweight interceptor/middleware:

```
                       [ User Prompt ]
                              │
                              ▼
                 [ Vector Index / Embedder ]
                              │ (Retrieve Context Key)
                              ▼
                  [ Bayesian Tool Router ]
                   ├─ Thompson Sampling
                   ├─ Fallback Key Hashing
                   └─ Fetch (α, β) from Cache/DB
                              │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
     [ Selected Tool A ]               [ Selected Tool B ]
            │                                 │
            └────────────────┬────────────────┘
                             ▼
                     [ Execution Trace ]
                             │ (Success / Fail / Reward)
                             ▼
                 [ Decoupled Telemetry Hook ]
                             │ (Update α, β in DB)
                             ▼
                      [ Storage Cache ] (Redis, SQLite, In-Memory)
```

---

## Installation

Install using `uv` or standard pip:

```bash
# Core package (In-memory, SQLite, and Redis support)
uv pip install bayes-brain

# Install with local embedding support
uv pip install "bayes-brain[local-ml]"
```

---

## Quick Start

### 1. Initialize the Router with a Storage Backend and Embedder

You can use the built-in [GeminiEmbedder](file:///Users/sam/Locals%20Only/bayes-brain/src/bayes_brain/embeddings.py#L62-L138) or [OpenAIEmbedder](file:///Users/sam/Locals%20Only/bayes-brain/src/bayes_brain/embeddings.py#L140-L245) for lightweight, API-driven embeddings without downloading heavy local models.

```python
from bayes_brain.router import BayesianToolRouter
from bayes_brain.storage import SQLiteStorage
from bayes_brain.embeddings import GeminiEmbedder

# SQLiteStorage supports concurrent WAL mode and busy timeouts out of the box
storage = SQLiteStorage("bayes_cache.db")

# Automatically loads API key from GEMINI_API_KEY environment variable.
# Falls back to standard urllib requests if no SDK client is provided.
embedder = GeminiEmbedder(model_name="models/text-embedding-004")

router = BayesianToolRouter(
    storage=storage,
    embedder=embedder,
    decay_factor=0.95,
    fallback_tool="fallback_llm"
)
```

### 2. Route and Feedback

BayesBrain supports binary feedback (`success=True/False`) as well as continuous rewards (e.g. utility rates in `[0.0, 1.0]`) to support fine-grained feedback loops.

```python
context_prompt = "Summarize the latest AI research papers"
candidates = ["arxiv_rag", "google_search", "fallback_llm"]

# 1. Route the request (Thompson Sampling)
chosen_tool = router.route(
    context_text=context_prompt,
    candidate_tools=candidates
)
print(f"Routed task to: {chosen_tool}")

# 2. Execute and collect utility feedback
try:
    # Run your tool logic ...
    utility_score = 0.85  # e.g., success rate or response quality score
    success = True
except Exception:
    utility_score = 0.0
    success = False

# 3. Provide feedback (accepts success and/or continuous reward)
router.feedback(
    context_text=context_prompt,
    tool_name=chosen_tool,
    success=success,
    reward=utility_score
)
```

### 3. Asymmetric Telemetry with Trace IDs

For asynchronous non-blocking workflows, generate trace IDs during routing and apply feedback later:

```python
# Route task and receive a unique session trace identifier
chosen_tool, trace_id = router.route_with_trace(
    context_text=context_prompt,
    candidate_tools=candidates
)

# ... dispatch execution to background workers ...

# Update parameters asynchronously using the trace identifier
router.feedback_by_trace(trace_id=trace_id, reward=1.0)
```

---

## Core Features & Advanced Operations

### 🔌 Pluggable Vector Stores (`VectorStoreProtocol`)
You can inject custom lightweight vector stores (like Chroma or FAISS) by implementing the [VectorStoreProtocol](file:///Users/sam/Locals%20Only/bayes-brain/src/bayes_brain/embeddings.py#L17-L32):

```python
from typing import Sequence, Optional
from bayes_brain.embeddings import VectorStoreProtocol

class MyChromaStore(VectorStoreProtocol):
    def add_context(self, context_key: str, vector: Sequence[float]) -> None:
        # Write to your index
        pass

    def get_nearest_context(
        self, query_vector: Sequence[float], similarity_threshold: float = 0.8
    ) -> Optional[str]:
        # Return nearest context key from your index
        return "some_context_key"

router = BayesianToolRouter(
    storage=storage,
    embedder=embedder,
    vector_store=MyChromaStore()
)
```
* **Incremental Caching**: Storage backends ([InMemoryStorage](file:///Users/sam/Locals%20Only/bayes-brain/src/bayes_brain/storage.py#L85-L145), [SQLiteStorage](file:///Users/sam/Locals%20Only/bayes-brain/src/bayes_brain/storage.py#L146-L322), and [RedisStorage](file:///Users/sam/Locals%20Only/bayes-brain/src/bayes_brain/storage.py#L323-L448)) implement incremental `load_all_vectors` and `save_vector` methods to perform fine-grained writes.
* **Auto-Migration**: Upon first load, legacy monolithic `"vector_context_store"` JSON structures are automatically migrated to incremental, query-friendly database columns.

### 🛡️ Fail-Safe Routing & Telemetry Hooks
All entry points in [BayesianToolRouter](file:///Users/sam/Locals%20Only/bayes-brain/src/bayes_brain/router.py#L16-L32) are wrapped in fail-safe try-except blocks. If storage connections fail, vector indexes corrupt, or sampling throws exceptions, the router silently handles the issue by returning the candidate default or `fallback_tool`.

You can configure a `telemetry_hook` to alert your developer channels or monitoring tools on failures:

```python
def my_telemetry_callback(context_text: str, error: Exception, metadata: dict):
    print(f"Telemetry Alert: Routing error on context '{context_text}': {error}")
    # Forward warning to Sentry, Datadog, etc.

router = BayesianToolRouter(
    storage=storage,
    embedder=embedder,
    fallback_tool="fallback_llm",
    telemetry_hook=my_telemetry_callback
)
```

### ⚡ High-Performance SQLite WAL Mode
The [SQLiteStorage](file:///Users/sam/Locals%20Only/bayes-brain/src/bayes_brain/storage.py#L146-L322) backend runs with:
* **Write-Ahead Logging (WAL)**: `PRAGMA journal_mode=WAL;` to dramatically improve read/write concurrency.
* **Busy Timeout**: `PRAGMA busy_timeout=5000;` to gracefully handle concurrent write contentions.

### 🔏 Robust Hashed Exact Matching Fallbacks
When operating without an embedder, or if API embedder requests fail, the router normalizes the context (stripping whitespace) and hashes the string using SHA-256 (prefixed with `hash_`). This guarantees a short, fixed-length context key and prevents key matching fragility due to whitespace differences.

---

## Integrations & FastMCP Server

Optimize tool selection in Claude Code or other MCP hosts by registering a Meta-Tool to handle dynamic routing, alongside administrative tools to manage and monitor bandit beliefs.

You can configure and expose these endpoints using [create_mcp_server](file:///Users/sam/Locals%20Only/bayes-brain/src/bayes_brain/mcp_server.py#L67-L249):

```python
from bayes_brain.mcp_server import create_mcp_server

# Build the FastMCP server
mcp = create_mcp_server(
    server_name="BayesBrain",
    db_path="mcp_bandit.db",
    sub_tools=["local_pytest", "docker_sandbox", "fallback_api"]
)
```

### Registered Tools & Resources

| Endpoint | Type | Description |
| :--- | :--- | :--- |
| `execute_adaptive_action` | `Tool` | Thompson sampling routes incoming tasks to the best sub-tool and automatically applies execution feedback. |
| `get_tool_beliefs` | `Tool` | Retrieve current posterior $\alpha$ and $\beta$ beliefs for all tools under a given context. |
| `reset_beliefs` | `Tool` | Reset the beliefs back to the default prior `(1.0, 1.0)` for a tool under a context. |
| `bayes://metrics` | `Resource` | Exposes a Markdown Dashboard with context clusters, expected success rates, and raw JSON data. |

---

## License

MIT
