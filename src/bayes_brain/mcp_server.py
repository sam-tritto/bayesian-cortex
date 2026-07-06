import json
import os
import sqlite3
from typing import Callable, List, Optional, Tuple, Union

from mcp.server.fastmcp import FastMCP

from bayes_brain.router import BayesianToolRouter
from bayes_brain.storage import SQLiteStorage
def _get_all_beliefs(router: BayesianToolRouter) -> dict:
    """
    Retrieve all beliefs from the storage backend of the router.
    """
    from bayes_brain.storage import InMemoryStorage, SQLiteStorage, RedisStorage

    storage = router.storage
    beliefs = {}

    if isinstance(storage, InMemoryStorage):
        with storage._lock:
            for (ctx_key, tool), (alpha, beta) in storage._data.items():
                if ctx_key not in beliefs:
                    beliefs[ctx_key] = {}
                beliefs[ctx_key][tool] = {"alpha": alpha, "beta": beta}

    elif isinstance(storage, SQLiteStorage):
        conn = sqlite3.connect(storage.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT context_key, tool_name, alpha, beta FROM tool_params")
            for ctx_key, tool, alpha, beta in cursor.fetchall():
                if ctx_key not in beliefs:
                    beliefs[ctx_key] = {}
                beliefs[ctx_key][tool] = {"alpha": float(alpha), "beta": float(beta)}
        except Exception:
            pass
        finally:
            conn.close()

    elif isinstance(storage, RedisStorage):
        try:
            keys = storage.client.keys(f"{storage.prefix}*")
            for k in keys:
                k_str = k.decode("utf-8") if isinstance(k, bytes) else str(k)
                if k_str.startswith(storage.prefix):
                    ctx_candidate = k_str[len(storage.prefix):]
                    if ctx_candidate == "context_vectors" or ctx_candidate.startswith("metadata:"):
                        continue
                    
                    hash_data = storage.client.hgetall(k)
                    for field, val in hash_data.items():
                        field_str = field.decode("utf-8") if isinstance(field, bytes) else str(field)
                        val_str = val.decode("utf-8") if isinstance(val, bytes) else str(val)
                        if ":" in field_str:
                            tool, param_type = field_str.split(":", 1)
                            if ctx_candidate not in beliefs:
                                beliefs[ctx_candidate] = {}
                            if tool not in beliefs[ctx_candidate]:
                                beliefs[ctx_candidate][tool] = {}
                            beliefs[ctx_candidate][tool][param_type] = float(val_str)
        except Exception:
            pass

    return beliefs


def create_mcp_server(
    server_name: str = "BayesBrain",
    db_path: str = "mcp_bandit.db",
    sub_tools: Optional[List[str]] = None,
    tool_executor: Optional[Callable[[str, str], Union[Tuple[str, bool], str]]] = None,
) -> FastMCP:
    """
    Configure and return a FastMCP server wrapping a BayesianToolRouter instance.

    Args:
        server_name: The display name of the FastMCP server.
        db_path: SQLite database path to store tool statistics.
        sub_tools: A list of candidate sub-tools the router can dynamically select.
        tool_executor: A callable taking (selected_tool, task_description) returning
                       either (output, success_bool) or just output (which defaults to success).
    """
    mcp = FastMCP(server_name)
    
    # Use SQLiteStorage for fast, persistent, local-cache statistics
    storage = SQLiteStorage(db_path)
    router = BayesianToolRouter(storage=storage)
    
    available_tools = sub_tools or ["local_pytest", "docker_sandbox", "fallback_api"]

    async def run_tool_logic(tool_name: str, task: str) -> Tuple[str, bool]:
        if tool_executor:
            import inspect
            if inspect.iscoroutinefunction(tool_executor):
                res = await tool_executor(tool_name, task)
            else:
                res = tool_executor(tool_name, task)

            if isinstance(res, tuple):
                return str(res[0]), bool(res[1])
            return str(res), True

        # Default fallback simulator for demonstrations
        if tool_name == "local_pytest":
            # Simulate failure on task requests with styling checks
            success = "style" not in task.lower()
            return f"Pytest execution: {'PASSED' if success else 'FAILED'}", success
        elif tool_name == "docker_sandbox":
            return "Docker sandbox execution completed successfully.", True
        else:
            return "Fallback API request dispatched and processed.", True

    @mcp.tool()
    async def execute_adaptive_action(task_description: str) -> str:
        """
        Dynamically routes task execution to the most reliable sub-tool.

        Args:
            task_description: A description of the code or integration task to execute.
        """
        # Thompson sampling selects the tool
        chosen_tool, trace_id = router.route_with_trace(
            context_text=task_description,
            candidate_tools=available_tools
        )

        try:
            result, success = await run_tool_logic(chosen_tool, task_description)
        except Exception as e:
            result, success = f"Adaptive execution encountered an error: {str(e)}", False

        # Submit execution feedback asynchronously
        router.feedback_by_trace(trace_id=trace_id, success=success)

        return f"Selected Tool: {chosen_tool}\nExecution Output:\n{result}"

    @mcp.tool()
    async def get_tool_beliefs(context: str) -> str:
        """
        Retrieve the current posterior alpha and beta beliefs for all tools under a given context.

        Args:
            context: The context text to look up beliefs for.
        """
        # Resolve the context key (non-mutating lookup first)
        context_key = None
        if router.embedder:
            try:
                vector = router.embedder.embed_query(context)
                context_key = router._context_store.get_nearest_context(
                    query_vector=vector,
                    similarity_threshold=router.similarity_threshold,
                )
            except Exception:
                pass

        if context_key is None:
            context_key = router._hash_context_text(context)

        beliefs = {}
        for tool_name in available_tools:
            alpha, beta = router.storage.get_tool_params(context_key, tool_name)
            if alpha == 1.0 and beta == 1.0 and tool_name in router.priors:
                alpha, beta = router.priors[tool_name]
            beliefs[tool_name] = {"alpha": alpha, "beta": beta}

        return json.dumps(beliefs, indent=2)

    @mcp.tool()
    async def reset_beliefs(context: str, tool: str) -> str:
        """
        Reset the posterior alpha and beta beliefs back to the default prior (1.0, 1.0)
        for a specific tool under a given context.

        Args:
            context: The context text to reset beliefs for.
            tool: The specific tool name to reset beliefs for.
        """
        if tool not in available_tools:
            return f"Error: Tool '{tool}' is not in the list of available tools ({available_tools})."

        # Resolve the context key
        context_key = None
        if router.embedder:
            try:
                vector = router.embedder.embed_query(context)
                context_key = router._context_store.get_nearest_context(
                    query_vector=vector,
                    similarity_threshold=router.similarity_threshold,
                )
            except Exception:
                pass

        if context_key is None:
            context_key = router._hash_context_text(context)

        router.storage.update_tool_params(context_key, tool, 1.0, 1.0)
        return f"Beliefs for tool '{tool}' under context key '{context_key}' have been reset to (1.0, 1.0)."

    @mcp.resource("bayes://metrics")
    async def get_metrics() -> str:
        """
        Expose a JSON/Markdown dashboard of current statistics and beliefs.
        """
        all_beliefs = _get_all_beliefs(router)

        lines = [
            "# Bayes Brain Multi-Armed Bandit Metrics",
            "",
        ]

        if not all_beliefs:
            lines.append("No active beliefs recorded in storage yet.")
        else:
            lines.append(f"**Total Context Clusters:** {len(all_beliefs)}")
            lines.append("")

            # Build full beliefs including defaults/priors for all available tools
            full_beliefs = {}
            for ctx_key, tools_beliefs in all_beliefs.items():
                full_beliefs[ctx_key] = {}
                for t_name in available_tools:
                    params = tools_beliefs.get(t_name, {"alpha": 1.0, "beta": 1.0})
                    if params["alpha"] == 1.0 and params["beta"] == 1.0 and t_name in router.priors:
                        params = {"alpha": router.priors[t_name][0], "beta": router.priors[t_name][1]}
                    full_beliefs[ctx_key][t_name] = params

            for ctx_key, tools_beliefs in full_beliefs.items():
                lines.append(f"### Context Cluster: `{ctx_key}`")
                lines.append("")
                lines.append("| Tool | Alpha (Successes) | Beta (Failures) | Expected Success Rate |")
                lines.append("| :--- | :---: | :---: | :---: |")

                for t_name, params in tools_beliefs.items():
                    alpha = params.get("alpha", 1.0)
                    beta = params.get("beta", 1.0)
                    total = alpha + beta
                    expected_rate = (alpha / total) * 100 if total > 0 else 50.0
                    lines.append(f"| {t_name} | {alpha:.2f} | {beta:.2f} | {expected_rate:.1f}% |")
                lines.append("")

            lines.append("## Raw JSON Data")
            lines.append("```json")
            lines.append(json.dumps(full_beliefs, indent=2))
            lines.append("```")

        return "\n".join(lines)

    return mcp


if __name__ == "__main__":
    # Fetch DB path or configure defaults from environment
    db_path = os.environ.get("BAYES_DB_PATH", "mcp_bandit.db")
    server = create_mcp_server(db_path=db_path)
    server.run()
