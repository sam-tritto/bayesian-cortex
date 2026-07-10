import json
import os
import sqlite3
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from mcp.server.fastmcp import FastMCP
from scipy.stats import beta as scipy_beta

from bayesian_cortex.router import (
    AsyncBayesianRouter,
    BayesianRouter,
)
from bayesian_cortex.storage import AsyncSQLiteStorage, SQLiteStorage


async def _get_all_beliefs(router: Union[BayesianRouter, AsyncBayesianRouter]) -> dict:
    """
    Retrieve all beliefs from the storage backend of the router.
    """
    from bayesian_cortex.storage import (
        AsyncInMemoryStorage,
        AsyncRedisStorage,
        AsyncSQLiteStorage,
        InMemoryStorage,
        RedisStorage,
    )

    storage = router.storage
    beliefs = {}

    if isinstance(storage, AsyncInMemoryStorage):
        async with storage._lock:
            for (ctx_key, candidate), (alpha, beta) in storage._data.items():
                if ctx_key not in beliefs:
                    beliefs[ctx_key] = {}
                beliefs[ctx_key][candidate] = {"alpha": alpha, "beta": beta}
    elif isinstance(storage, InMemoryStorage):
        with storage._lock:
            for (ctx_key, candidate), (alpha, beta) in storage._data.items():
                if ctx_key not in beliefs:
                    beliefs[ctx_key] = {}
                beliefs[ctx_key][candidate] = {"alpha": alpha, "beta": beta}

    elif isinstance(storage, AsyncSQLiteStorage):
        import aiosqlite

        async with aiosqlite.connect(storage.db_path) as conn:
            try:
                async with conn.execute(
                    "SELECT context_key, candidate_name, alpha, beta FROM candidate_params"
                ) as cursor:
                    async for row in cursor:
                        ctx_key, candidate, alpha, beta = row
                        if ctx_key not in beliefs:
                            beliefs[ctx_key] = {}
                        beliefs[ctx_key][candidate] = {
                            "alpha": float(alpha),
                            "beta": float(beta),
                        }
            except Exception:
                pass
    elif isinstance(storage, SQLiteStorage):
        conn = sqlite3.connect(storage.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT context_key, candidate_name, alpha, beta FROM candidate_params"
            )
            for ctx_key, candidate, alpha, beta in cursor.fetchall():
                if ctx_key not in beliefs:
                    beliefs[ctx_key] = {}
                beliefs[ctx_key][candidate] = {
                    "alpha": float(alpha),
                    "beta": float(beta),
                }
        except Exception:
            pass
        finally:
            conn.close()

    elif isinstance(storage, AsyncRedisStorage):
        try:
            keys = await storage.client.keys(f"{storage.prefix}*")
            for k in keys:
                k_str = k.decode("utf-8") if isinstance(k, bytes) else str(k)
                if k_str.startswith(storage.prefix):
                    ctx_candidate = k_str[len(storage.prefix) :]
                    if ctx_candidate == "context_vectors" or ctx_candidate.startswith(
                        "metadata:"
                    ):
                        continue

                    hash_data = await storage.client.hgetall(k)
                    for field, val in hash_data.items():
                        field_str = (
                            field.decode("utf-8")
                            if isinstance(field, bytes)
                            else str(field)
                        )
                        val_str = (
                            val.decode("utf-8") if isinstance(val, bytes) else str(val)
                        )
                        if ":" in field_str:
                            candidate, param_type = field_str.split(":", 1)
                            if ctx_candidate not in beliefs:
                                beliefs[ctx_candidate] = {}
                            if candidate not in beliefs[ctx_candidate]:
                                beliefs[ctx_candidate][candidate] = {}
                            beliefs[ctx_candidate][candidate][param_type] = float(
                                val_str
                            )
        except Exception:
            pass
    elif isinstance(storage, RedisStorage):
        try:
            keys = storage.client.keys(f"{storage.prefix}*")
            for k in keys:
                k_str = k.decode("utf-8") if isinstance(k, bytes) else str(k)
                if k_str.startswith(storage.prefix):
                    ctx_candidate = k_str[len(storage.prefix) :]
                    if ctx_candidate == "context_vectors" or ctx_candidate.startswith(
                        "metadata:"
                    ):
                        continue

                    hash_data = storage.client.hgetall(k)
                    for field, val in hash_data.items():
                        field_str = (
                            field.decode("utf-8")
                            if isinstance(field, bytes)
                            else str(field)
                        )
                        val_str = (
                            val.decode("utf-8") if isinstance(val, bytes) else str(val)
                        )
                        if ":" in field_str:
                            candidate, param_type = field_str.split(":", 1)
                            if ctx_candidate not in beliefs:
                                beliefs[ctx_candidate] = {}
                            if candidate not in beliefs[ctx_candidate]:
                                beliefs[ctx_candidate][candidate] = {}
                            beliefs[ctx_candidate][candidate][param_type] = float(
                                val_str
                            )
        except Exception:
            pass

    return beliefs


def _get_candidate_color(candidate_name: str, index: int = 0) -> str:
    colors = {
        "local_pytest": "#3b82f6",  # Blue
        "docker_sandbox": "#8b5cf6",  # Purple
        "fallback_api": "#f97316",  # Orange
        "tool1": "#10b981",  # Emerald Green
        "tool2": "#ec4899",  # Pink
    }
    if candidate_name in colors:
        return colors[candidate_name]
    default_colors = ["#10b981", "#ec4899", "#f59e0b", "#06b6d4", "#f43f5e", "#14b8a6"]
    return default_colors[index % len(default_colors)]


def generate_ascii_sparkline(alpha: float, beta: float, width: int = 15) -> str:
    """
    Generate an ASCII-based sparkline representing the shape of the Beta distribution.
    """
    try:
        blocks = [" ", " ", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
        x = np.linspace(0.02, 0.98, width)
        y = scipy_beta.pdf(x, alpha, beta)
        y = np.nan_to_num(y, nan=0.0, posinf=100.0, neginf=0.0)
        max_y = np.max(y)
        if max_y <= 0:
            return " " * width
        sparkline = []
        for val in y:
            idx = int(round((val / max_y) * (len(blocks) - 1)))
            idx = max(0, min(len(blocks) - 1, idx))
            sparkline.append(blocks[idx])
        return "".join(sparkline)
    except Exception:
        return " " * width


def generate_beta_pdf_svg(
    candidates_params: dict, width: int = 600, height: int = 250
) -> str:
    """
    Generate SVG plotting the Beta density curves for candidate candidates.
    """
    try:
        x_vals = np.linspace(0.0, 1.0, 101)
        curves = {}
        max_y = 1.0

        for idx, (candidate_name, params) in enumerate(candidates_params.items()):
            alpha = params.get("alpha", 1.0)
            beta = params.get("beta", 1.0)
            y_vals = scipy_beta.pdf(x_vals, alpha, beta)
            y_vals = np.nan_to_num(y_vals, nan=0.0, posinf=100.0, neginf=0.0)
            curves[candidate_name] = y_vals
            max_y = max(max_y, np.max(y_vals))

        padding_top = 20
        padding_bottom = 40
        padding_left = 50
        padding_right = 160

        chart_width = width - padding_left - padding_right
        chart_height = height - padding_top - padding_bottom

        svg_elements = [
            f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" xmlns="http://www.w3.org/2000/svg" style="background-color: #111827; border-radius: 8px; font-family: ui-sans-serif, system-ui, sans-serif;">',
            f'<rect width="{width}" height="{height}" fill="#111827" rx="8" />',
        ]

        # Horizontal grids
        for i in range(5):
            y_frac = i / 4
            y_pos = padding_top + chart_height * y_frac
            y_val = max_y * (1 - y_frac)
            svg_elements.append(
                f'<line x1="{padding_left}" y1="{y_pos}" x2="{padding_left + chart_width}" y2="{y_pos}" stroke="#374151" stroke-dasharray="4" stroke-width="1" />'
            )
            svg_elements.append(
                f'<text x="{padding_left - 8}" y="{y_pos + 4}" fill="#9ca3af" font-size="10" text-anchor="end">{y_val:.2f}</text>'
            )

        # Vertical grids
        for i in range(5):
            x_frac = i / 4
            x_pos = padding_left + chart_width * x_frac
            svg_elements.append(
                f'<line x1="{x_pos}" y1="{padding_top}" x2="{x_pos}" y2="{padding_top + chart_height}" stroke="#374151" stroke-dasharray="4" stroke-width="1" />'
            )
            svg_elements.append(
                f'<text x="{x_pos}" y="{padding_top + chart_height + 16}" fill="#9ca3af" font-size="10" text-anchor="middle">{x_frac:.2f}</text>'
            )

        svg_elements.append(
            f'<text x="{padding_left + chart_width / 2}" y="{height - 8}" fill="#9ca3af" font-size="11" font-weight="500" text-anchor="middle">Success Probability (x)</text>'
        )

        # Plot curves
        for idx, (candidate_name, y_vals) in enumerate(curves.items()):
            candidate_color = _get_candidate_color(candidate_name, idx)
            points = []
            for x, y in zip(x_vals, y_vals):
                px = padding_left + x * chart_width
                py = padding_top + chart_height - (y / max_y) * chart_height
                points.append(f"{px:.1f},{py:.1f}")

            path_d = "M " + " L ".join(points)
            svg_elements.append(
                f'<path d="{path_d}" fill="none" stroke="{candidate_color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />'
            )

        # Legend
        legend_left = padding_left + chart_width + 16
        for idx, (candidate_name, params) in enumerate(candidates_params.items()):
            candidate_color = _get_candidate_color(candidate_name, idx)
            alpha = params.get("alpha", 1.0)
            beta = params.get("beta", 1.0)
            mean = alpha / (alpha + beta) if (alpha + beta) > 0 else 0.5
            y_pos = padding_top + 16 + idx * 28

            svg_elements.append(
                f'<rect x="{legend_left}" y="{y_pos}" width="12" height="12" rx="3" fill="{candidate_color}" />'
            )
            svg_elements.append(
                f'<text x="{legend_left + 18}" y="{y_pos + 10}" fill="#e5e7eb" font-size="11" font-weight="bold">{candidate_name}</text>'
            )
            svg_elements.append(
                f'<text x="{legend_left + 18}" y="{y_pos + 22}" fill="#9ca3af" font-size="9">Beta({alpha:.1f}, {beta:.1f}) | μ={mean*100:.1f}%</text>'
            )

        svg_elements.append("</svg>")
        return "\n".join(svg_elements)
    except Exception as e:
        return f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" xmlns="http://www.w3.org/2000/svg" style="background-color: #111827;"><text x="{width/2}" y="{height/2}" fill="#ef4444" text-anchor="middle">Error rendering SVG: {str(e)}</text></svg>'


def generate_history_svg(
    logs: list, available_candidates: list, width: int = 600, height: int = 250
) -> str:
    """
    Generate SVG plotting the moving average success rate over time.
    """
    try:
        candidate_rewards = {t: [] for t in available_candidates}
        history_points = []

        for idx, log in enumerate(logs):
            c_name = log["candidate_name"]
            reward = log["reward"]
            if c_name in candidate_rewards:
                if reward is not None:
                    candidate_rewards[c_name].append(reward)
                point_avgs = {}
                for t in available_candidates:
                    rewards = candidate_rewards[t]
                    if len(rewards) > 0:
                        window = rewards[-10:]
                        point_avgs[t] = sum(window) / len(window)
                    else:
                        point_avgs[t] = None
                history_points.append(point_avgs)

        has_points = any(
            any(val is not None for val in pt.values()) for pt in history_points
        )

        svg_elements = [
            f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" xmlns="http://www.w3.org/2000/svg" style="background-color: #111827; border-radius: 8px; font-family: ui-sans-serif, system-ui, sans-serif;">',
            f'<rect width="{width}" height="{height}" fill="#111827" rx="8" />',
        ]

        padding_top = 20
        padding_bottom = 40
        padding_left = 50
        padding_right = 160

        chart_width = width - padding_left - padding_right
        chart_height = height - padding_top - padding_bottom

        if not has_points or len(logs) < 2:
            svg_elements.append(
                f'<text x="{width / 2}" y="{height / 2}" fill="#9ca3af" font-size="12" text-anchor="middle">Insufficient feedback data to plot success rates over time.</text>'
            )
            svg_elements.append("</svg>")
            return "\n".join(svg_elements)

        # Horizontal grids
        for i in range(5):
            y_frac = i / 4
            y_pos = padding_top + chart_height * y_frac
            y_val = 100 * (1 - y_frac)
            svg_elements.append(
                f'<line x1="{padding_left}" y1="{y_pos}" x2="{padding_left + chart_width}" y2="{y_pos}" stroke="#374151" stroke-dasharray="4" stroke-width="1" />'
            )
            svg_elements.append(
                f'<text x="{padding_left - 8}" y="{y_pos + 4}" fill="#9ca3af" font-size="10" text-anchor="end">{y_val:.0f}%</text>'
            )

        # Vertical grids
        num_runs = len(history_points)
        x_steps = min(5, num_runs)
        for i in range(x_steps):
            x_frac = i / (x_steps - 1) if x_steps > 1 else 0.0
            x_pos = padding_left + chart_width * x_frac
            run_num = int(round(x_frac * (num_runs - 1))) + 1
            svg_elements.append(
                f'<line x1="{x_pos}" y1="{padding_top}" x2="{x_pos}" y2="{padding_top + chart_height}" stroke="#374151" stroke-dasharray="4" stroke-width="1" />'
            )
            svg_elements.append(
                f'<text x="{x_pos}" y="{padding_top + chart_height + 16}" fill="#9ca3af" font-size="10" text-anchor="middle">#{run_num}</text>'
            )

        svg_elements.append(
            f'<text x="{padding_left + chart_width / 2}" y="{height - 8}" fill="#9ca3af" font-size="11" font-weight="500" text-anchor="middle">Execution Sequence (Timeline)</text>'
        )

        # Plot lines
        for idx, candidate_name in enumerate(available_candidates):
            candidate_color = _get_candidate_color(candidate_name, idx)
            points = []
            for run_idx, pt in enumerate(history_points):
                val = pt.get(candidate_name)
                if val is not None:
                    x_pos = padding_left + (run_idx / (num_runs - 1)) * chart_width
                    y_pos = padding_top + chart_height - val * chart_height
                    points.append(f"{x_pos:.1f},{y_pos:.1f}")

            if len(points) >= 2:
                path_d = "M " + " L ".join(points)
                svg_elements.append(
                    f'<path d="{path_d}" fill="none" stroke="{candidate_color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />'
                )
                for pt_str in points:
                    cx, cy = pt_str.split(",")
                    svg_elements.append(
                        f'<circle cx="{cx}" cy="{cy}" r="3.5" fill="#ffffff" stroke="{candidate_color}" stroke-width="1.5" />'
                    )

        # Legend
        legend_left = padding_left + chart_width + 16
        for idx, candidate_name in enumerate(available_candidates):
            candidate_color = _get_candidate_color(candidate_name, idx)
            rewards = candidate_rewards[candidate_name]
            total_selections = len(rewards)
            current_avg = (
                sum(rewards[-10:]) / len(rewards[-10:])
                if len(rewards[-10:]) > 0
                else 0.0
            )
            y_pos = padding_top + 16 + idx * 28

            svg_elements.append(
                f'<rect x="{legend_left}" y="{y_pos}" width="12" height="12" rx="3" fill="{candidate_color}" />'
            )
            svg_elements.append(
                f'<text x="{legend_left + 18}" y="{y_pos + 10}" fill="#e5e7eb" font-size="11" font-weight="bold">{candidate_name}</text>'
            )
            svg_elements.append(
                f'<text x="{legend_left + 18}" y="{y_pos + 22}" fill="#9ca3af" font-size="9">MA(10): {current_avg*100:.1f}% | Total: {total_selections}</text>'
            )

        svg_elements.append("</svg>")
        return "\n".join(svg_elements)
    except Exception as e:
        return f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" xmlns="http://www.w3.org/2000/svg" style="background-color: #111827;"><text x="{width/2}" y="{height/2}" fill="#ef4444" text-anchor="middle">Error rendering SVG: {str(e)}</text></svg>'


def create_mcp_server(
    server_name: str = "BayesianCortex",
    db_path: str = "mcp_bandit.db",
    candidates: Optional[List[str]] = None,
    candidate_executor: Optional[
        Callable[[str, str], Union[Tuple[str, bool], str]]
    ] = None,
    priors: Optional[Dict[str, Tuple[float, float]]] = None,
    contextual_priors: Optional[List[Dict[str, Any]]] = None,
    enable_tools: bool = True,
    enable_skills: bool = True,
    enable_rag: bool = False,
    sub_tools: Optional[List[str]] = None,
) -> FastMCP:
    """
    Configure and return a FastMCP server wrapping an AsyncBayesianRouter instance.

    Args:
        server_name: The display name of the FastMCP server.
        db_path: SQLite database path to store candidate/skill statistics.
        candidates: A list of candidate sub-candidates/skills the router can dynamically select.
        candidate_executor: A callable taking (selected_candidate, task_description) returning
                       either (output, success_bool) or just output (which defaults to success).
        priors: Preseeded alpha/beta priors for candidates/skills to mitigate cold start.
        contextual_priors: List of context-specific prior rules matching regex or embedding clusters.
        enable_tools: Toggle registration of core execution tools.
        enable_skills: Toggle registration of administrative/belief management tools.
        enable_rag: Toggle registration of retrieval routing endpoints.
        sub_tools: Alternative list parameter for candidates, mapped for compatibility.
    """
    mcp = FastMCP(server_name)

    # Use AsyncSQLiteStorage for non-blocking database operations
    storage = AsyncSQLiteStorage(db_path)
    router = AsyncBayesianRouter(
        storage=storage,
        priors=priors,
        contextual_priors=contextual_priors,
    )

    available_candidates = (
        candidates or sub_tools or ["local_pytest", "docker_sandbox", "fallback_api"]
    )

    async def run_candidate_logic(candidate_name: str, task: str) -> Tuple[str, bool]:
        if candidate_executor:
            import inspect

            if inspect.iscoroutinefunction(candidate_executor):
                res = await candidate_executor(candidate_name, task)
            else:
                res = candidate_executor(candidate_name, task)

            if isinstance(res, tuple):
                return str(res[0]), bool(res[1])
            return str(res), True

        # Default fallback simulator for demonstrations
        if candidate_name == "local_pytest":
            # Simulate failure on task requests with styling checks
            success = "style" not in task.lower()
            return f"Pytest execution: {'PASSED' if success else 'FAILED'}", success
        elif candidate_name == "docker_sandbox":
            return "Docker sandbox execution completed successfully.", True
        else:
            return "Fallback API request dispatched and processed.", True

    if enable_tools:

        @mcp.tool()
        async def execute_adaptive_action(task_description: str) -> str:
            """
            Dynamically routes task execution to the most reliable sub-candidate/skill candidate.

            Args:
                task_description: A description of the code or integration task to execute.
            """
            # Thompson sampling selects the candidate
            chosen_candidate, trace_id = await router.aroute_with_trace(
                context_text=task_description, candidates=available_candidates
            )

            try:
                result, success = await run_candidate_logic(
                    chosen_candidate, task_description
                )
            except Exception as e:
                result, success = (
                    f"Adaptive execution encountered an error: {str(e)}",
                    False,
                )

            # Submit execution feedback asynchronously
            await router.afeedback_by_trace(trace_id=trace_id, success=success)

            return (
                f"Selected Candidate: {chosen_candidate}\nExecution Output:\n{result}"
            )

    if enable_skills:

        @mcp.tool()
        async def get_candidate_beliefs(context: str) -> str:
            """
            Retrieve the current posterior alpha and beta beliefs for all candidates/skills under a given context.

            Args:
                context: The context text to look up beliefs for.
            """
            # Resolve the context key (non-mutating lookup first)
            context_key = None
            if router.embedder:
                try:
                    if hasattr(router.embedder, "aembed_query"):
                        vector = await router.embedder.aembed_query(context)
                    else:
                        vector = router.embedder.embed_query(context)
                    context_key = await router._context_store.aget_nearest_context(
                        query_vector=vector,
                        similarity_threshold=router.similarity_threshold,
                    )
                except Exception:
                    pass

            if context_key is None:
                context_key = router._hash_context_text(context)

            beliefs = {}
            for candidate_name in available_candidates:
                alpha, beta = await router.storage.get_candidate_params(
                    context_key, candidate_name
                )
                # Use an explicit existence check rather than value equality to detect cold
                # start. After a failure with decay_factor=1.0 the params legitimately
                # remain at the floor (1.0, 1.0), which is indistinguishable from a
                # candidate that has never been observed if we only compare values.
                is_cold_start = not await router.storage.ahas_candidate_params(
                    context_key, candidate_name
                )
                if is_cold_start:
                    if hasattr(router, "get_prior"):
                        import inspect

                        if inspect.iscoroutinefunction(router.get_prior):
                            alpha, beta = await router.get_prior(
                                context, candidate_name
                            )
                        else:
                            alpha, beta = router.get_prior(context, candidate_name)
                    elif candidate_name in router.priors:
                        alpha, beta = router.priors[candidate_name]
                beliefs[candidate_name] = {"alpha": alpha, "beta": beta}

            return json.dumps(beliefs, indent=2)

        @mcp.tool()
        async def reset_candidate_beliefs(context: str, candidate: str) -> str:
            """
            Reset the posterior alpha and beta beliefs back to the default prior (1.0, 1.0)
            for a specific candidate/skill under a given context.

            Args:
                context: The context text to reset beliefs for.
                candidate: The specific candidate/skill name to reset beliefs for.
            """
            if candidate not in available_candidates:
                return f"Error: Candidate '{candidate}' is not in the list of available candidates ({available_candidates})."

            # Resolve the context key
            context_key = None
            if router.embedder:
                try:
                    if hasattr(router.embedder, "aembed_query"):
                        vector = await router.embedder.aembed_query(context)
                    else:
                        vector = router.embedder.embed_query(context)
                    context_key = await router._context_store.aget_nearest_context(
                        query_vector=vector,
                        similarity_threshold=router.similarity_threshold,
                    )
                except Exception:
                    pass

            if context_key is None:
                context_key = router._hash_context_text(context)

            await router.storage.update_candidate_params(
                context_key, candidate, 1.0, 1.0
            )
            return f"Beliefs for candidate '{candidate}' under context key '{context_key}' have been reset to (1.0, 1.0)."

    if enable_rag:

        @mcp.tool()
        async def route_knowledge_base(query: str, vector_indices: List[str]) -> str:
            """
            Selects the highest-yielding RAG index source for a semantic query.

            Args:
                query: The semantic search query.
                vector_indices: List of candidate RAG vector index sources/strategies to route to.
            """
            chosen, trace_id = await router.aroute_with_trace(
                context_text=query, candidates=vector_indices
            )
            return f"Selected RAG Index: {chosen}\nTrace ID: {trace_id}"

    @mcp.resource("cortex://metrics")
    async def get_metrics() -> str:
        """
        Expose a JSON/Markdown dashboard of current statistics and beliefs.
        """
        all_beliefs = await _get_all_beliefs(router)
        logs = await router.storage.get_selection_logs()

        lines = [
            "# Bayes Brain Multi-Armed Bandit Metrics",
            "",
        ]

        # Check if anything is enabled
        if not (enable_tools or enable_skills or enable_rag):
            lines.append(
                "*All metric tracking components (Tools, Skills, RAG) are currently disabled.*"
            )
            return "\n".join(lines)

        # 1. Posterior Belief Distributions
        if enable_skills:
            lines.append("## Posterior Belief Distributions")
            lines.append("")
            if not all_beliefs:
                lines.append("*No active beliefs recorded in storage yet.*")
                lines.append("")
            else:
                lines.append(f"**Total Context Clusters:** {len(all_beliefs)}")
                lines.append("")

                # Build full beliefs including defaults/priors for all available candidates
                full_beliefs = {}
                for ctx_key, tools_beliefs in all_beliefs.items():
                    full_beliefs[ctx_key] = {}
                    for c_name in available_candidates:
                        params = tools_beliefs.get(c_name, {"alpha": 1.0, "beta": 1.0})
                        if (
                            params["alpha"] == 1.0
                            and params["beta"] == 1.0
                            and c_name in router.priors
                        ):
                            params = {
                                "alpha": router.priors[c_name][0],
                                "beta": router.priors[c_name][1],
                            }
                        full_beliefs[ctx_key][c_name] = params

                for ctx_key, tools_beliefs in full_beliefs.items():
                    lines.append(f"### Context Cluster: `{ctx_key}`")
                    lines.append("")
                    lines.append(
                        "| Candidate | Alpha (Successes) | Beta (Failures) | Expected Success Rate | Belief Sparkline (0 to 1) |"
                    )
                    lines.append("| :--- | :---: | :---: | :---: | :---: |")

                    for c_name, params in tools_beliefs.items():
                        alpha = params.get("alpha", 1.0)
                        beta = params.get("beta", 1.0)
                        total = alpha + beta
                        expected_rate = (alpha / total) * 100 if total > 0 else 50.0
                        sparkline = generate_ascii_sparkline(alpha, beta)
                        lines.append(
                            f"| {c_name} | {alpha:.2f} | {beta:.2f} | {expected_rate:.1f}% | `{sparkline}` |"
                        )
                    lines.append("")

                    # Render SVG Beta density curve
                    lines.append("#### Probability Density Curves")
                    lines.append('<div align="center">')
                    lines.append(generate_beta_pdf_svg(tools_beliefs))
                    lines.append("</div>")
                    lines.append("")

        # 2. Historical performance statistics
        if enable_tools or enable_rag:
            lines.append("## Selection Frequencies & Success Rates")
            lines.append("")
            total_selections = len(logs)
            lines.append(f"**Total Decisions Logged:** {total_selections}")
            lines.append("")

            # Aggregate counts and success rates
            select_counts = dict.fromkeys(available_candidates, 0)
            feedback_counts = dict.fromkeys(available_candidates, 0)
            rewards = {t: [] for t in available_candidates}

            for log in logs:
                c_name = log["candidate_name"]
                reward = log["reward"]
                if c_name in select_counts:
                    select_counts[c_name] += 1
                if reward is not None:
                    if c_name in feedback_counts:
                        feedback_counts[c_name] += 1
                    if c_name in rewards:
                        rewards[c_name].append(reward)

            lines.append(
                "| Candidate | Total Selections | Selection Frequency | Runs with Feedback | Overall Success Rate | Recent Success Rate (MA10) |"
            )
            lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
            for idx, c_name in enumerate(available_candidates):
                sel_cnt = select_counts[c_name]
                sel_freq = (
                    (sel_cnt / total_selections * 100) if total_selections > 0 else 0.0
                )
                fb_cnt = feedback_counts[c_name]
                r_list = rewards[c_name]
                overall_rate = (
                    (sum(r_list) / len(r_list) * 100) if len(r_list) > 0 else 0.0
                )
                ma_list = r_list[-10:]
                recent_rate = (
                    (sum(ma_list) / len(ma_list) * 100) if len(ma_list) > 0 else 0.0
                )

                recent_rate_str = f"{recent_rate:.1f}%" if len(ma_list) > 0 else "N/A"
                overall_rate_str = f"{overall_rate:.1f}%" if len(r_list) > 0 else "N/A"
                lines.append(
                    f"| {c_name} | {sel_cnt} | {sel_freq:.1f}% | {fb_cnt} | {overall_rate_str} | {recent_rate_str} |"
                )
            lines.append("")

            lines.append("### Moving Average Success Rate Over Time")
            lines.append('<div align="center">')
            lines.append(generate_history_svg(logs, available_candidates))
            lines.append("</div>")
            lines.append("")

        # 3. Recent actions log
        if enable_tools or enable_rag:
            lines.append("## Chronological Execution Log (Recent)")
            lines.append("")
            total_selections = len(logs)
            recent_logs = logs[-20:] if total_selections > 20 else logs
            if not recent_logs:
                lines.append("*No routing actions logged yet.*")
                lines.append("")
            else:
                lines.append(
                    "| Trace ID | Timestamp (UTC) | Context Key | Selected Candidate | Feedback Reward / Outcome |"
                )
                lines.append("| :--- | :--- | :--- | :--- | :--- |")
                for log in reversed(recent_logs):
                    tid = log["trace_id"]
                    tid_display = tid[:15] + "..." if len(tid) > 18 else tid
                    ts = log["timestamp"]
                    ts_display = ts.split(".")[0].replace("T", " ") if "T" in ts else ts
                    ctx = log["context_key"]
                    candidate = log["candidate_name"]
                    rew = log["reward"]
                    if rew is None:
                        rew_display = "*Pending feedback*"
                    elif rew == 1.0:
                        rew_display = "**1.0 (Success)**"
                    elif rew == 0.0:
                        rew_display = "0.0 (Failure)"
                    else:
                        rew_display = f"{rew:.2f}"
                    lines.append(
                        f"| `{tid_display}` | {ts_display} | `{ctx}` | `{candidate}` | {rew_display} |"
                    )
                lines.append("")

        if enable_skills and all_beliefs:
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
