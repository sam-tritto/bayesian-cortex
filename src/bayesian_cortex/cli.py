#!/usr/bin/env python3
"""
Command-Line Interface (CLI) for BayesianCortex.
Allows routing decisions, feedback logging, belief visualization, and starting the FastMCP server.
"""

import argparse
import json
import os
import sys
import sqlite3
from typing import List, Optional

from bayesian_cortex.router import BayesianRouter
from bayesian_cortex.storage import SQLiteStorage
from bayesian_cortex.mcp_server import create_mcp_server, generate_ascii_sparkline


def get_sqlite_candidates(db_path: str) -> List[str]:
    """Retrieve distinct candidate names from the SQLite database if it exists."""
    candidates = []
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT candidate_name FROM candidate_params")
            candidates = [row[0] for row in cursor.fetchall()]
            conn.close()
        except Exception:
            pass
    return candidates


def get_all_beliefs_sync(db_path: str) -> dict:
    """Retrieve all context keys and candidate beliefs from SQLite storage."""
    beliefs = {}
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
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


def init_router(args) -> BayesianRouter:
    """Initialize a BayesianRouter instance based on CLI arguments."""
    db_path = args.db_path or os.environ.get("BAYES_DB_PATH", "mcp_bandit.db")
    storage = SQLiteStorage(db_path)

    # Initialize secret key: check CLI args, then env, fall back to a consistent CLI secret
    secret_key = args.secret_key or os.environ.get("BAYESIAN_CORTEX_SECRET_KEY")
    if not secret_key:
        secret_key = "cli_default_secret_key_non_production"

    # Setup embedder if specified
    embedder = None
    if getattr(args, "embedder_type", None):
        embedder_type = args.embedder_type.lower()
        model_name = getattr(args, "model_name", None)

        if embedder_type == "openai":
            from bayesian_cortex.embeddings import OpenAIEmbedder
            embedder = OpenAIEmbedder(model_name=model_name or "text-embedding-3-small")
        elif embedder_type == "gemini":
            from bayesian_cortex.embeddings import GeminiEmbedder
            embedder = GeminiEmbedder(model_name=model_name or "models/text-embedding-004")
        elif embedder_type == "anthropic":
            from bayesian_cortex.embeddings import AnthropicEmbedder
            embedder = AnthropicEmbedder(model_name=model_name or "voyage-large-2-instruct")
        elif embedder_type == "cohere":
            from bayesian_cortex.embeddings import CohereEmbedder
            embedder = CohereEmbedder(model_name=model_name or "embed-english-v3.0")
        elif embedder_type == "local":
            from bayesian_cortex.embeddings import LocalSentenceTransformerEmbedder
            embedder = LocalSentenceTransformerEmbedder(model_name=model_name or "all-MiniLM-L6-v2")
        elif embedder_type == "llamacpp":
            from bayesian_cortex.embeddings import LlamaCppEmbedder
            embedder = LlamaCppEmbedder(endpoint_url=model_name or "http://localhost:8080/embedding")
        else:
            print(f"Error: Unknown embedder type '{embedder_type}'", file=sys.stderr)
            sys.exit(1)

    # Resolve candidates
    candidates = None
    if getattr(args, "candidates", None):
        candidates = [c.strip() for c in args.candidates.split(",") if c.strip()]
    else:
        candidates = get_sqlite_candidates(db_path)
        if not candidates:
            # Fallback default candidates to allow easy sandbox testing
            candidates = ["local_pytest", "docker_sandbox", "fallback_api"]

    router = BayesianRouter(
        storage=storage,
        embedder=embedder,
        candidates=candidates,
        secret_key=secret_key,
    )
    return router


def handle_route(args):
    """Handle the 'route' CLI subcommand."""
    router = init_router(args)
    context = args.context

    # Route decision
    candidate, trace_id = router.route_with_trace(context_text=context)

    if args.json:
        print(json.dumps({"candidate": candidate, "trace_id": trace_id}, indent=2))
    else:
        print(f"Selected Candidate: {candidate}")
        print(f"Trace ID: {trace_id}")


def handle_feedback(args):
    """Handle the 'feedback' CLI subcommand."""
    router = init_router(args)
    trace_id = args.trace_id

    # Determine reward value
    if args.reward is not None:
        reward = args.reward
    elif args.success:
        reward = 1.0
    elif args.failure:
        reward = 0.0
    else:
        print("Error: Must specify either --success, --failure, or --reward <val>", file=sys.stderr)
        sys.exit(1)

    try:
        # Submit feedback
        alpha, beta = router.feedback_by_trace(trace_id=trace_id, reward=reward, strict=True)
        # Decode context/candidate to display update info
        ctx_key, candidate = router._decode_trace_id(trace_id)

        if args.json:
            print(json.dumps({
                "status": "success",
                "trace_id": trace_id,
                "context_key": ctx_key,
                "candidate": candidate,
                "alpha": alpha,
                "beta": beta
            }, indent=2))
        else:
            print("Feedback submitted successfully!")
            print(f"Context Key: {ctx_key}")
            print(f"Candidate:   {candidate}")
            print(f"Updated posterior beliefs: Alpha={alpha:.1f}, Beta={beta:.1f}")
    except Exception as e:
        if args.json:
            print(json.dumps({"status": "error", "message": str(e)}, indent=2))
        else:
            print(f"Error submitting feedback: {e}", file=sys.stderr)
        sys.exit(1)


def handle_beliefs(args):
    """Handle the 'beliefs' CLI subcommand."""
    router = init_router(args)
    db_path = args.db_path or os.environ.get("BAYES_DB_PATH", "mcp_bandit.db")

    if args.context:
        # Show beliefs for specific context
        context = args.context
        try:
            context_key = router._resolve_context_key(context)
        except Exception:
            context_key = router._hash_context_text(context)

        candidates = router.candidates or ["local_pytest", "docker_sandbox", "fallback_api"]
        beliefs_data = {}
        for c in candidates:
            alpha, beta = router.storage.get_candidate_params(context_key, c)
            mean = alpha / (alpha + beta) if (alpha + beta) > 0 else 0.5
            beliefs_data[c] = {"alpha": alpha, "beta": beta, "mean": mean}

        if args.json:
            print(json.dumps({
                "context": context,
                "context_key": context_key,
                "beliefs": beliefs_data
            }, indent=2))
        else:
            print(f"Context: {context}")
            print(f"Context Key: {context_key}\n")
            print(f"{'Candidate':<20} | {'Alpha':<6} | {'Beta':<6} | {'Mean':<6} | Distribution (Beta PDF)")
            print("-" * 75)
            for c in candidates:
                data = beliefs_data[c]
                spark = generate_ascii_sparkline(data["alpha"], data["beta"])
                print(f"{c:<20} | {data['alpha']:<6.1f} | {data['beta']:<6.1f} | {data['mean']*100:<5.1f}% | {spark}")
    else:
        # Show all beliefs in the DB
        all_beliefs = get_all_beliefs_sync(db_path)
        if args.json:
            print(json.dumps(all_beliefs, indent=2))
        else:
            if not all_beliefs:
                print("No beliefs found in the database.")
                return

            for ctx_key, candidates_data in all_beliefs.items():
                print(f"Context Key: {ctx_key}")
                print(f"{'Candidate':<20} | {'Alpha':<6} | {'Beta':<6} | {'Mean':<6} | Distribution (Beta PDF)")
                print("-" * 75)
                for c, data in candidates_data.items():
                    mean = data["alpha"] / (data["alpha"] + data["beta"]) if (data["alpha"] + data["beta"]) > 0 else 0.5
                    spark = generate_ascii_sparkline(data["alpha"], data["beta"])
                    print(f"{c:<20} | {data['alpha']:<6.1f} | {data['beta']:<6.1f} | {mean*100:<5.1f}% | {spark}")
                print()


def handle_reset(args):
    """Handle the 'reset' CLI subcommand."""
    router = init_router(args)
    db_path = args.db_path or os.environ.get("BAYES_DB_PATH", "mcp_bandit.db")

    if not os.path.exists(db_path):
        if args.json:
            print(json.dumps({"status": "success", "message": "Database file does not exist."}, indent=2))
        else:
            print(f"Database file '{db_path}' does not exist. Nothing to reset.")
        return

    context_key = None
    if args.context:
        try:
            context_key = router._resolve_context_key(args.context)
        except Exception:
            context_key = router._hash_context_text(args.context)

    candidate = args.candidate

    try:
        conn = sqlite3.connect(db_path)
        with conn:
            if context_key and candidate:
                conn.execute(
                    "DELETE FROM candidate_params WHERE context_key = ? AND candidate_name = ?",
                    (context_key, candidate)
                )
                msg = f"Reset beliefs for candidate '{candidate}' under context key '{context_key}'."
            elif context_key:
                conn.execute(
                    "DELETE FROM candidate_params WHERE context_key = ?",
                    (context_key,)
                )
                conn.execute(
                    "DELETE FROM context_vectors WHERE context_key = ?",
                    (context_key,)
                )
                msg = f"Reset beliefs for all candidates under context key '{context_key}'."
            elif candidate:
                conn.execute(
                    "DELETE FROM candidate_params WHERE candidate_name = ?",
                    (candidate,)
                )
                msg = f"Reset beliefs for candidate '{candidate}' across all contexts."
            else:
                conn.execute("DELETE FROM candidate_params")
                conn.execute("DELETE FROM context_vectors")
                conn.execute("DELETE FROM linear_bandit_params")
                conn.execute("DELETE FROM selection_log")
                msg = "Completely reset all candidate beliefs and logs from the database."

        conn.close()

        if args.json:
            print(json.dumps({"status": "success", "message": msg}, indent=2))
        else:
            print(msg)
    except Exception as e:
        if args.json:
            print(json.dumps({"status": "error", "message": str(e)}, indent=2))
        else:
            print(f"Error resetting database: {e}", file=sys.stderr)
        sys.exit(1)


def handle_mcp(args):
    """Handle the 'mcp' CLI subcommand to start FastMCP server."""
    db_path = args.db_path or os.environ.get("BAYES_DB_PATH", "mcp_bandit.db")
    candidates = None
    if args.candidates:
        candidates = [c.strip() for c in args.candidates.split(",") if c.strip()]

    print(f"Starting BayesianCortex FastMCP Server using database: {db_path}", file=sys.stderr)
    if candidates:
        print(f"Exposing candidates: {', '.join(candidates)}", file=sys.stderr)

    mcp = create_mcp_server(db_path=db_path, candidates=candidates)
    mcp.run()


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="BayesianCortex: Command-Line Interface (CLI) for Bayesian Bandits & Claude Integration."
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to execute")

    # Global options
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--db-path",
        help="Path to the SQLite database (default: mcp_bandit.db)",
    )
    parent_parser.add_argument(
        "--secret-key",
        help="HMAC secret key for trace verification (optional)",
    )
    parent_parser.add_argument(
        "--candidates",
        help="Comma-separated list of candidate/skill names",
    )
    parent_parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format",
    )

    # Route Subcommand
    route_parser = subparsers.add_parser("route", parents=[parent_parser], help="Route a context text to a candidate")
    route_parser.add_argument(
        "--context",
        required=True,
        help="The context text/prompt to route",
    )
    route_parser.add_argument(
        "--embedder-type",
        choices=["openai", "gemini", "anthropic", "cohere", "local", "llamacpp"],
        help="Embedder provider type to use",
    )
    route_parser.add_argument(
        "--model-name",
        help="Embedder model name or local endpoint URL",
    )

    # Feedback Subcommand
    feedback_parser = subparsers.add_parser("feedback", parents=[parent_parser], help="Submit routing feedback/outcome")
    feedback_parser.add_argument(
        "--trace-id",
        required=True,
        help="The signed trace ID returned from routing",
    )
    feedback_group = feedback_parser.add_mutually_exclusive_group(required=True)
    feedback_group.add_argument(
        "--success",
        action="store_true",
        help="Report routing success (reward = 1.0)",
    )
    feedback_group.add_argument(
        "--failure",
        action="store_true",
        help="Report routing failure (reward = 0.0)",
    )
    feedback_group.add_argument(
        "--reward",
        type=float,
        help="Direct float reward in [0.0, 1.0]",
    )

    # Beliefs Subcommand
    beliefs_parser = subparsers.add_parser("beliefs", parents=[parent_parser], help="Visualize posterior beliefs")
    beliefs_parser.add_argument(
        "--context",
        help="Specific context text to check beliefs for. If omitted, lists all contexts.",
    )
    beliefs_parser.add_argument(
        "--embedder-type",
        choices=["openai", "gemini", "anthropic", "cohere", "local", "llamacpp"],
        help="Embedder provider type to use",
    )
    beliefs_parser.add_argument(
        "--model-name",
        help="Embedder model name or local endpoint URL",
    )

    # Reset Subcommand
    reset_parser = subparsers.add_parser("reset", parents=[parent_parser], help="Reset database parameters back to cold start")
    reset_parser.add_argument(
        "--context",
        help="Reset beliefs under a specific context prompt",
    )
    reset_parser.add_argument(
        "--candidate",
        help="Reset beliefs for a specific candidate name",
    )

    # MCP Subcommand
    mcp_parser = subparsers.add_parser("mcp", help="Start the FastMCP server for Claude/Cursor integration")
    mcp_parser.add_argument(
        "--db-path",
        help="Path to the SQLite database (default: mcp_bandit.db)",
    )
    mcp_parser.add_argument(
        "--candidates",
        help="Comma-separated list of candidate/skill names",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "route":
        handle_route(args)
    elif args.command == "feedback":
        handle_feedback(args)
    elif args.command == "beliefs":
        handle_beliefs(args)
    elif args.command == "reset":
        handle_reset(args)
    elif args.command == "mcp":
        handle_mcp(args)


if __name__ == "__main__":
    main()
