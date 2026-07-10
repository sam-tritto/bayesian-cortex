#!/usr/bin/env python3
"""
Interactive demo for BayesianCortex.
Illustrates Thompson Sampling, Context Clustering, and Drift Adaptation in real time.
"""

import os
import random
import sqlite3
import sys
import time
from typing import Dict, List, Sequence

try:
    import numpy as np
    from scipy.stats import beta as scipy_beta
except ImportError:
    print(
        "⚠️  This demo requires 'numpy' and 'scipy' to perform math and render sparklines."
    )
    print("Please install them or run this script using 'uv':")
    print("    uv run python demo.py")
    sys.exit(1)

# Import BayesianCortex components
try:
    from bayesian_cortex import BayesianRouter
    from bayesian_cortex.storage import InMemoryStorage, SQLiteStorage
except ImportError:
    # If run before installing/building, add src to path
    src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
    sys.path.insert(0, src_path)
    from bayesian_cortex import BayesianRouter
    from bayesian_cortex.storage import InMemoryStorage, SQLiteStorage


# Define task templates for simulation and routing
TASK_TEMPLATES = {
    "sql": [
        "Write a SQL query to find all active customers",
        "Select sum(amount) from transactions group by month",
        "Perform a postgres join between users and orders table",
        "Optimize table index on user_id in sqlite database",
        "Query database to check user status",
    ],
    "python": [
        "Write a python script to parse logs recursively",
        "Create an async function using asyncio loops",
        "Implement a merge sort algorithm in python",
        "Write a regex pattern to extract emails from raw string",
        "Define python class wrapper for API responses",
    ],
    "search": [
        "Search the web for Apple's current stock price",
        "Find the latest news on AI routing papers",
        "What is the weather in New York tomorrow?",
        "Lookup the policy guidelines for parental leave",
        "Who won the soccer world cup in 2022?",
    ],
}

CANDIDATES = ["sql_expert", "python_expert", "web_research"]

# Ground-truth success rates of candidates per task category
GROUND_TRUTH = {
    "sql": {"sql_expert": 0.95, "python_expert": 0.30, "web_research": 0.10},
    "python": {"sql_expert": 0.20, "python_expert": 0.90, "web_research": 0.15},
    "search": {"sql_expert": 0.05, "python_expert": 0.25, "web_research": 0.95},
}


class DemoEmbedder:
    """
    A lightweight, deterministic 3D embedder that maps query keywords
    to distinct vector clusters without neural networks or remote APIs.
    """

    def embed_query(self, text: str) -> Sequence[float]:
        text_lower = text.lower()
        v = [0.0, 0.0, 0.0]

        # SQL-related keywords
        if any(
            w in text_lower
            for w in [
                "sql",
                "database",
                "query",
                "select",
                "table",
                "postgres",
                "sqlite",
                "insert",
                "db",
            ]
        ):
            v[0] = 1.0
        # Coding/Python-related keywords
        if any(
            w in text_lower
            for w in [
                "python",
                "code",
                "script",
                "function",
                "asyncio",
                "loop",
                "regex",
                "algorithm",
            ]
        ):
            v[1] = 1.0
        # Web Search/General keywords
        if any(
            w in text_lower
            for w in [
                "search",
                "find",
                "news",
                "weather",
                "latest",
                "stock",
                "google",
                "policy",
                "wiki",
            ]
        ):
            v[2] = 1.0

        # Default if no keywords match
        if sum(v) == 0:
            v = [0.5, 0.5, 0.5]

        # Normalize the vector to unit length
        norm = float(np.linalg.norm(v))
        if norm > 0:
            return [float(x / norm) for x in v]
        return [0.0, 0.0, 0.0]

    def embed_queries(self, texts: List[str]) -> List[Sequence[float]]:
        return [self.embed_query(t) for t in texts]


# Keep an in-memory mapping of context cluster keys to human-readable names for display purposes
context_labels: Dict[str, str] = {}


def register_context_label(router: BayesianRouter, prompt: str):
    """Resolve and label the context cluster for the current prompt."""
    try:
        ctx_key = router._resolve_context_key(prompt)
    except Exception:
        ctx_key = router._hash_context_text(prompt)

    if ctx_key not in context_labels:
        text_lower = prompt.lower()
        if any(
            w in text_lower
            for w in [
                "sql",
                "database",
                "query",
                "select",
                "table",
                "postgres",
                "sqlite",
                "insert",
                "db",
            ]
        ):
            context_labels[ctx_key] = "SQL Database Tasks"
        elif any(
            w in text_lower
            for w in [
                "python",
                "code",
                "script",
                "function",
                "asyncio",
                "loop",
                "regex",
                "algorithm",
            ]
        ):
            context_labels[ctx_key] = "Python Programming Tasks"
        elif any(
            w in text_lower
            for w in [
                "search",
                "find",
                "news",
                "weather",
                "latest",
                "stock",
                "google",
                "policy",
                "wiki",
            ]
        ):
            context_labels[ctx_key] = "Web Search & FAQ Retrieval"
        else:
            context_labels[ctx_key] = f"Generic / Custom Task: '{prompt[:20]}...'"
    return ctx_key


def generate_ascii_sparkline(alpha: float, beta: float, width: int = 15) -> str:
    """Generate an ASCII-based sparkline representing the shape of the Beta distribution."""
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


def retrieve_all_beliefs(
    router: BayesianRouter,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Fetch all alpha/beta beliefs from the router's storage backend."""
    storage = router.storage
    beliefs = {}

    if isinstance(storage, InMemoryStorage):
        with storage._lock:
            for (ctx_key, candidate), (alpha, beta) in storage._data.items():
                if ctx_key not in beliefs:
                    beliefs[ctx_key] = {}
                beliefs[ctx_key][candidate] = {"alpha": alpha, "beta": beta}
    elif isinstance(storage, SQLiteStorage):
        try:
            conn = sqlite3.connect(storage.db_path)
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
            conn.close()
        except Exception:
            pass

    return beliefs


def print_beliefs_dashboard(router: BayesianRouter):
    """Renders the current posterior parameter beliefs inside a terminal-friendly dashboard."""
    beliefs = retrieve_all_beliefs(router)

    print("\n" + "=" * 90)
    print("                      📊 BAYESIAN CORTEX BELIEFS DASHBOARD")
    print("=" * 90)

    if not beliefs:
        print(
            "   No active beliefs recorded in storage yet. Try routing some tasks first!"
        )
        print("=" * 90 + "\n")
        return

    for ctx_key, candidates_params in beliefs.items():
        label = context_labels.get(ctx_key, f"Cluster key: {ctx_key}")
        print(f"\n📂 Context Cluster: \033[1;36m{label}\033[0m (`{ctx_key}`)")
        print("-" * 90)
        print(
            f"{'Candidate / Tool':<20} | {'Alpha (Succ)':<12} | {'Beta (Fail)':<11} | {'Expected Rate':<13} | {'Posterior Sparkline (0 to 1)':<30}"
        )
        print("-" * 90)
        for candidate_name in CANDIDATES:
            params = candidates_params.get(candidate_name, {"alpha": 1.0, "beta": 1.0})
            alpha = params["alpha"]
            beta = params["beta"]
            total = alpha + beta
            expected_rate = (alpha / total) * 100 if total > 0 else 50.0
            sparkline = generate_ascii_sparkline(alpha, beta)
            print(
                f"{candidate_name:<20} | {alpha:<12.2f} | {beta:<11.2f} | {expected_rate:>11.1f}% | `\033[32m{sparkline}\033[0m`"
            )
    print("=" * 90 + "\n")


def clear_console():
    os.system("cls" if os.name == "nt" else "clear")


def run_interactive_loop(router: BayesianRouter):
    """Option 1: Interactive manual routing loop."""
    while True:
        clear_console()
        print("======================================================================")
        print("               🛠️  MANUAL INTERACTIVE ROUTING LOOP")
        print("======================================================================")
        print(
            "Enter prompts and guide the multi-armed bandit manually by marking decisions"
        )
        print("as a success or failure.")
        print("-" * 70)
        print("Select a prompt type:")
        print("1) SQL:    'postgres join between users and orders table'")
        print("2) Python: 'Write an asyncio network socket script'")
        print("3) Search: 'Google today's stock price of Google'")
        print("4) Custom: Type your own task context")
        print("5) Go back to main menu")

        choice = input("\nSelect option (1-5): ").strip()
        if choice == "5":
            break

        prompt = ""
        if choice == "1":
            prompt = "postgres join between users and orders table"
        elif choice == "2":
            prompt = "Write an asyncio network socket script"
        elif choice == "3":
            prompt = "Google today's stock price of Google"
        elif choice == "4":
            prompt = input("\nEnter your custom task: ").strip()
            if not prompt:
                input("\nPrompt cannot be empty. Press Enter to retry...")
                continue
        else:
            input("\nInvalid choice. Press Enter to retry...")
            continue

        register_context_label(router, prompt)

        print(f"\nRouting query: '\033[1;33m{prompt}\033[0m'")
        print("Resolving Thompson Sampling selection...")

        # Route
        chosen_tool = router.route(context_text=prompt, candidates=CANDIDATES)
        print(f"\n🎯 BayesianRouter selected candidate: \033[1;32m{chosen_tool}\033[0m")

        # Request feedback from user
        feedback_choice = (
            input("\nWas this selection a SUCCESS? (y/n/skip): ").strip().lower()
        )
        if feedback_choice in ["y", "yes"]:
            router.feedback(
                context_text=prompt, candidate_name=chosen_tool, success=True
            )
            print("\033[32m✔ Logged SUCCESS feedback. Alpha incremented.\033[0m")
        elif feedback_choice in ["n", "no"]:
            router.feedback(
                context_text=prompt, candidate_name=chosen_tool, success=False
            )
            print("\033[31m✘ Logged FAILURE feedback. Beta incremented.\033[0m")
        else:
            print("Feedback skipped.")

        print_beliefs_dashboard(router)
        input("Press Enter to continue...")


def run_automated_simulation(router: BayesianRouter, rounds: int = 100):
    """Option 2: Simulated automated convergence demo."""
    clear_console()
    print("======================================================================")
    print("               ⚡ AUTOMATED CONVERGENCE SIMULATION")
    print("======================================================================")
    print(f"We will simulate {rounds} continuous rounds of random user prompts.")
    print("The simulator will automatically determine success based on ground-truth")
    print("probabilities (e.g. sql_expert has 95% success rate for SQL queries).")
    print("\nStarting simulation in 2 seconds...")
    time.sleep(2)

    counts = dict.fromkeys(CANDIDATES, 0)
    successes = dict.fromkeys(CANDIDATES, 0)

    for i in range(1, rounds + 1):
        category = random.choice(["sql", "python", "search"])
        prompt = random.choice(TASK_TEMPLATES[category])
        register_context_label(router, prompt)

        # Route
        chosen_tool = router.route(context_text=prompt, candidates=CANDIDATES)
        counts[chosen_tool] += 1

        # Determine simulated outcome
        success_prob = GROUND_TRUTH[category][chosen_tool]
        success = random.random() <= success_prob

        if success:
            successes[chosen_tool] += 1

        router.feedback(
            context_text=prompt, candidate_name=chosen_tool, success=success
        )

        # Print update every step (or delay slightly to make it visual)
        result_str = "\033[32mSUCCESS\033[0m" if success else "\033[31mFAILURE\033[0m"
        print(
            f"Round {i:03d}/{rounds:03d} | Category: {category:<6} | Routed to: {chosen_tool:<15} | Result: {result_str}"
        )
        time.sleep(0.03)

    print("\nSimulation complete!")
    print("\nAggregated selection statistics:")
    print("-" * 50)
    for tool in CANDIDATES:
        cnt = counts[tool]
        succ = successes[tool]
        rate = (succ / cnt * 100) if cnt > 0 else 0.0
        print(
            f" - {tool:<15}: Selected {cnt:02d} times | Actual Success: {succ:02d}/{cnt:02d} ({rate:.1f}%)"
        )
    print("-" * 50)

    print_beliefs_dashboard(router)
    input("Press Enter to continue...")


def run_drift_simulation():
    """Option 3: Non-stationary drift simulation demonstrating decay factor."""
    clear_console()
    print("======================================================================")
    print("               🌊 NON-STATIONARY PERFORMANCE DRIFT DEMO")
    print("======================================================================")
    print("This mode simulates what happens in real life when a previously reliable")
    print("microservice degrades, and shows how BayesianCortex adapts.")
    print("\nScenario Setup:")
    print(" - Tasks: Only SQL Database tasks (e.g. 'postgres index optimizations')")
    print(" - Candidate options: ['sql_expert', 'python_expert']")
    print(
        " - Rounds 1 to 25:   sql_expert has a 95% success rate; python_expert is 50%."
    )
    print(
        " - Rounds 26 to 50:  sql_expert suffers database downtime, success drops to 10%!"
    )
    print(
        "\nWe initialize a router with decay_factor = 0.90 to allow rapid discount of"
    )
    print("outdated history.")
    print("\nStarting simulation in 3 seconds...")
    time.sleep(3)

    # Initialize a clean transient router for drift demo with decay
    drift_storage = InMemoryStorage()
    drift_router = BayesianRouter(
        storage=drift_storage, embedder=DemoEmbedder(), decay_factor=0.90
    )

    prompt = "postgres index optimizations"
    ctx_key = register_context_label(drift_router, prompt)

    # Pre-populate label for drift
    context_labels[ctx_key] = "SQL Database Tasks (Drifting Environment)"

    candidates = ["sql_expert", "python_expert"]

    for i in range(1, 51):
        # Route
        chosen_tool = drift_router.route(context_text=prompt, candidates=candidates)

        # Ground-truth success rates based on phase
        if i <= 25:
            success_prob = 0.95 if chosen_tool == "sql_expert" else 0.50
            phase_str = "Stationary Phase"
        else:
            success_prob = 0.10 if chosen_tool == "sql_expert" else 0.50
            phase_str = "\033[1;31mDrift Phase (SQL DOWN)\033[0m"

        success = random.random() <= success_prob
        drift_router.feedback(
            context_text=prompt, candidate_name=chosen_tool, success=success
        )

        params = retrieve_all_beliefs(drift_router).get(ctx_key, {})
        sql_p = params.get("sql_expert", {"alpha": 1.0, "beta": 1.0})
        py_p = params.get("python_expert", {"alpha": 1.0, "beta": 1.0})

        result_str = "\033[32mSUCCESS\033[0m" if success else "\033[31mFAILURE\033[0m"
        print(
            f"Round {i:02d}/50 | {phase_str:<32} | Routed: {chosen_tool:<15} | Outcome: {result_str} | sql: (α={sql_p['alpha']:.1f}, β={sql_p['beta']:.1f}) | python: (α={py_p['alpha']:.1f}, β={py_p['beta']:.1f})"
        )
        time.sleep(0.1)

    print("\nDrift simulation complete!")
    print("\nDid you notice?")
    print(" - In Rounds 1-25, the router converged on sql_expert.")
    print(
        " - When sql_expert started failing (Rounds 26+), the old success parameter (Alpha)"
    )
    print(
        "   was decayed exponentially by 0.90 every round, making its parameter values drop."
    )
    print(
        " - Within a few rounds, the router successfully shifted traffic to python_expert!"
    )

    print_beliefs_dashboard(drift_router)
    input("Press Enter to continue...")


def main():
    clear_console()
    print("======================================================================")
    print("     🧠 BayesianCortex: Contextual Multi-Armed Bandit Demo 🧠")
    print("======================================================================")
    print("Thompson Sampling resolves candidate selection dynamically under semantic")
    print("context clusters using exact conjugate updates.")
    print("-" * 70)
    print("Select a storage backend:")
    print("1) In-Memory storage (starts clean every execution)")
    print("2) Local SQLite Database (persists beliefs to 'demo_bayes_cache.db')")

    storage_choice = input("\nSelect option (1-2): ").strip()
    if storage_choice == "2":
        storage_backend = "sqlite"
        storage_path = "demo_bayes_cache.db"
        print("\nUsing SQLite storage (demo_bayes_cache.db)")
    else:
        storage_backend = "memory"
        storage_path = None
        print("\nUsing In-Memory storage (transient)")

    # Initialize the main router
    router = BayesianRouter(
        storage_backend=storage_backend,
        storage_path=storage_path,
        embedder=DemoEmbedder(),
        decay_factor=0.90,
    )

    # Pre-populate any existing contexts in the label map if loaded from SQLite
    beliefs = retrieve_all_beliefs(router)
    for key in beliefs.keys():
        if key not in context_labels:
            context_labels[key] = f"Restored Context Cluster `{key[:10]}`"

    time.sleep(1)

    while True:
        clear_console()
        print("======================================================================")
        print("              🧠 BAYESIAN CORTEX MAIN DEMO MENU")
        print("======================================================================")
        print(f"Backend: {storage_backend.upper()} | Active Clusters: {len(beliefs)}")
        print("-" * 70)
        print(
            "1) 🛠️  Manual Interactive Loop (enter prompts & evaluate success manually)"
        )
        print(
            "2) ⚡ Automated Convergence Simulation (runs 100 rounds with preset rates)"
        )
        print(
            "3) 🌊 Non-Stationary Drift Simulation (shows how decay adapts when tools degrade)"
        )
        print("4) 📊 Print Current Beliefs and Status")
        print("5) ❌ Reset all beliefs for this session")
        print("6) 🚪 Exit")

        choice = input("\nSelect option (1-6): ").strip()
        if choice == "1":
            run_interactive_loop(router)
        elif choice == "2":
            run_automated_simulation(router)
        elif choice == "3":
            run_drift_simulation()
        elif choice == "4":
            print_beliefs_dashboard(router)
            input("Press Enter to continue...")
        elif choice == "5":
            if storage_backend == "sqlite":
                # Clear the db file
                if os.path.exists("demo_bayes_cache.db"):
                    try:
                        os.remove("demo_bayes_cache.db")
                        if os.path.exists("demo_bayes_cache.db-shm"):
                            os.remove("demo_bayes_cache.db-shm")
                        if os.path.exists("demo_bayes_cache.db-wal"):
                            os.remove("demo_bayes_cache.db-wal")
                    except Exception:
                        pass
            # Re-initialize
            router = BayesianRouter(
                storage_backend=storage_backend,
                storage_path=storage_path,
                embedder=DemoEmbedder(),
                decay_factor=0.90,
            )
            context_labels.clear()
            input("\nBeliefs reset. Press Enter to continue...")
        elif choice == "6":
            print("\nThanks for exploring BayesianCortex!")
            break
        else:
            input("\nInvalid choice. Press Enter to retry...")


if __name__ == "__main__":
    main()
