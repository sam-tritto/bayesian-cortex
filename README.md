<p align="center">
  <img src="assets/logo.png" alt="BayesBrain Logo" width="400">
</p>

# BayesBrain: Dynamic Tool Routing via Bayesian Bandits


In multi-agent systems, a supervisor agent often needs to decide which specialized sub-agent or API tool to invoke to solve a specific user prompt. Traditionally, this is done using hardcoded heuristics, prompt engineering, or raw LLM classification logits. None of these handle real-time uncertainty or feedback loops well.

**BayesBrain** treats tool routing as a **Contextual Multi-Armed Bandit** using **Thompson Sampling** with exact Beta-Binomial conjugate updates.

---

## The Core Math Engine (Beta-Binomial Conjugate Pair)

To prevent runtime latency, BayesBrain avoids heavy Markov Chain Monte Carlo (MCMC) sampling (e.g., PyMC or Stan). Instead, it uses exact closed-form Beta-Binomial updates:

1. **Belief Representation**: Each tool $i$ is modeled as a Beta distribution: $\text{Beta}(\alpha_i, \beta_i)$.
2. **Prior (Initial State)**: $\alpha_i = 1, \beta_i = 1$ (Uniform flat distribution representing total uncertainty).
3. **Thompson Sampling**: For each candidate tool, sample a success probability:
   $$\theta_i \sim \text{Beta}(\alpha_i, \beta_i)$$
   Select the tool with the highest sampled probability:
   $$i^* = \arg\max_{i} \theta_i$$
4. **Posterior Update (Telemetry)**:
   - **Success**: $\alpha_i \leftarrow \alpha_i + 1$
   - **Failure**: $\beta_i \leftarrow \beta_i + 1$

### Handling Non-Stationary Environments (Drifting APIs)
If an API starts failing or degrades over time, historical successes should not dominate the routing indefinitely. BayesBrain applies an exponential decay factor $\gamma \in (0, 1)$ on historical updates prior to adding new rewards:
$$\alpha_t = \gamma \alpha_{t-1} + \text{reward}$$
$$\beta_t = \gamma \beta_{t-1} + (1 - \text{reward})$$

This ensures the router rapidly adapts to outages, API updates, or regressions.

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
                  └─ Fetch (α, β) from Cache/DB
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
     [ Selected Tool A ]               [ Selected Tool B ]
            │                                 │
            └────────────────┬────────────────┘
                             ▼
                     [ Execution Trace ]
                             │ (Success / Fail)
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

### 1. Define custom storage backend and ContextEmbedder

```python
from typing import Sequence
from bayes_brain.router import BayesianToolRouter
from bayes_brain.storage import SQLiteStorage

# Use SQLite for lightweight local caching
storage = SQLiteStorage("bayes_cache.db")

# Define a custom ContextEmbedder or use standard ones
class SimpleEmbedder:
    def embed_query(self, text: str) -> Sequence[float]:
        # Connect to OpenAI, Ollama, or local model here
        # Return a list of floats representing the embedding vector
        return [0.1, 0.2, 0.3] # Placeholder

router = BayesianToolRouter(
    storage=storage,
    embedder=SimpleEmbedder(),
    decay_factor=0.95
)
```

### 2. Route and Feedback

```python
# Route the tool based on the user prompt
context_prompt = "Find recent articles about climate change"
candidates = ["web_search", "vector_rag", "fallback_llm"]

chosen_tool = router.route(
    context_text=context_prompt,
    candidate_tools=candidates
)

print(f"Routing to: {chosen_tool}")

# Execute the tool ...
try:
    # Tool execution code
    success = True
except Exception:
    success = False

# Send feedback asynchronously
router.feedback(
    context_text=context_prompt,
    tool_name=chosen_tool,
    success=success
)
```

### 3. Asymmetric Telemetry with Trace IDs

To avoid stalling immediate response loops, track sessions with unique trace IDs and log feedback asynchronously:

```python
# Get chosen tool and trace token
chosen_tool, trace_id = router.route_with_trace(
    context_text=context_prompt,
    candidate_tools=candidates
)

# ... execution happens asynchronously ...

# Update later using the trace identifier
router.feedback_by_trace(trace_id=trace_id, success=True)
```

---

## Integrations

### LangGraph (Conditional Routing)
```python
def bayesian_routing_node(state):
    user_intent = state["messages"][-1].content
    available_tools = ["search_api", "vector_rag", "fallback_llm"]
    
    chosen_node = router.route(
        context_text=user_intent,
        candidate_tools=available_tools
    )
    return chosen_node

workflow.add_conditional_edges("supervisor_node", bayesian_routing_node)
```

### LangChain RunnableLambda
```python
from langchain_core.runnables import RunnableLambda

def route_payload(inputs):
    tool = router.route(inputs["context"], inputs["tools"])
    return {"selected_tool": tool, "original_input": inputs["input"]}

chain = RunnableLambda(route_payload) | tool_executor_chain
```

### FastMCP Server Integration
Optimize Claude Code tool invocation by registering a single Meta-Tool to handle tool selection under the hood:

```python
from mcp.server.fastmcp import FastMCP
from bayes_brain.router import BayesianToolRouter
from bayes_brain.storage import SQLiteStorage

mcp = FastMCP("BayesBrain")
router = BayesianToolRouter(storage=SQLiteStorage("mcp_bandit.db"))

@mcp.tool()
async def execute_adaptive_action(task_description: str) -> str:
    """Dynamically routes tasks to the most reliable sub-tool."""
    sub_tools = ["local_pytest", "docker_sandbox", "fallback_api"]
    
    chosen_tool, trace_id = router.route_with_trace(
        context_text=task_description, 
        candidate_tools=sub_tools
    )
    
    result, success = await run_tool_logic(chosen_tool, task_description)
    
    router.feedback_by_trace(trace_id=trace_id, success=success)
    return result
```

---

## License

MIT
