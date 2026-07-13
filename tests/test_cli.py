import json
import os
import sys
import pytest
import sqlite3
from unittest.mock import patch, MagicMock

from bayesian_cortex.cli import main, get_sqlite_candidates, get_all_beliefs_sync


@pytest.fixture
def clean_db(tmp_path):
    db_path = str(tmp_path / "test_cli.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    yield db_path
    if os.path.exists(db_path):
        os.remove(db_path)


def test_cli_help():
    """Test that running cli without args or with --help prints help text."""
    with patch("sys.argv", ["bayesian-cortex"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0

    with patch("sys.argv", ["bayesian-cortex", "--help"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0


def test_cli_route_and_feedback(clean_db):
    """Test routing a context and then submitting feedback."""
    # 1. Route context
    with patch("sys.argv", [
        "bayesian-cortex", "route",
        "--db-path", clean_db,
        "--context", "Write some SQL query",
        "--candidates", "sql_expert,python_expert",
    ]), patch("sys.stdout.write") as mock_stdout:
        main()

        # Capture output printed to stdout
        output = "".join(call.args[0] for call in mock_stdout.call_args_list)
        assert "Selected Candidate:" in output
        assert "Trace ID:" in output

        # Extract trace ID and selected candidate from output
        lines = output.strip().split("\n")
        trace_id = ""
        selected_candidate = ""
        for line in lines:
            if line.startswith("Trace ID:"):
                trace_id = line.split(":", 1)[1].strip()
            elif line.startswith("Selected Candidate:"):
                selected_candidate = line.split(":", 1)[1].strip()
        assert trace_id != ""
        assert selected_candidate != ""

    # 2. Submit feedback
    with patch("sys.argv", [
        "bayesian-cortex", "feedback",
        "--db-path", clean_db,
        "--trace-id", trace_id,
        "--success"
    ]), patch("sys.stdout.write") as mock_stdout:
        main()

        output = "".join(call.args[0] for call in mock_stdout.call_args_list)
        assert "Feedback submitted successfully!" in output
        assert "Updated posterior beliefs:" in output

    # Check candidates list in SQLite now contains the selected candidate
    candidates = get_sqlite_candidates(clean_db)
    assert selected_candidate in candidates


def test_cli_route_json(clean_db):
    """Test routing with JSON output format."""
    with patch("sys.argv", [
        "bayesian-cortex", "route",
        "--db-path", clean_db,
        "--context", "Write some python code",
        "--candidates", "sql_expert,python_expert",
        "--json"
    ]), patch("sys.stdout.write") as mock_stdout:
        main()

        output = "".join(call.args[0] for call in mock_stdout.call_args_list)
        parsed = json.loads(output)
        assert "candidate" in parsed
        assert "trace_id" in parsed
        assert parsed["candidate"] in ["sql_expert", "python_expert"]


def test_cli_beliefs(clean_db):
    """Test viewing candidate beliefs in text and JSON format."""
    # Seed db with some belief params
    conn = sqlite3.connect(clean_db)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS candidate_params (
            context_key TEXT,
            candidate_name TEXT,
            alpha REAL,
            beta REAL,
            PRIMARY KEY (context_key, candidate_name)
        )
    """)
    cursor.execute("INSERT INTO candidate_params VALUES ('key1', 'candidate_A', 10.0, 2.0)")
    cursor.execute("INSERT INTO candidate_params VALUES ('key1', 'candidate_B', 5.0, 5.0)")
    conn.commit()
    conn.close()

    # 1. Print all beliefs text format
    with patch("sys.argv", [
        "bayesian-cortex", "beliefs",
        "--db-path", clean_db
    ]), patch("sys.stdout.write") as mock_stdout:
        main()
        output = "".join(call.args[0] for call in mock_stdout.call_args_list)
        assert "Context Key: key1" in output
        assert "candidate_A" in output
        assert "candidate_B" in output

    # 2. Print all beliefs JSON format
    with patch("sys.argv", [
        "bayesian-cortex", "beliefs",
        "--db-path", clean_db,
        "--json"
    ]), patch("sys.stdout.write") as mock_stdout:
        main()
        output = "".join(call.args[0] for call in mock_stdout.call_args_list)
        parsed = json.loads(output)
        assert "key1" in parsed
        assert "candidate_A" in parsed["key1"]

    # 3. Print beliefs with context text format
    with patch("sys.argv", [
        "bayesian-cortex", "beliefs",
        "--db-path", clean_db,
        "--context", "some text prompt",
        "--candidates", "candidate_A,candidate_B"
    ]), patch("sys.stdout.write") as mock_stdout:
        main()
        output = "".join(call.args[0] for call in mock_stdout.call_args_list)
        assert "Context: some text prompt" in output
        assert "candidate_A" in output
        assert "candidate_B" in output


def test_cli_reset(clean_db):
    """Test resetting candidate beliefs."""
    # Seed db with some belief params
    conn = sqlite3.connect(clean_db)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS candidate_params (
            context_key TEXT,
            candidate_name TEXT,
            alpha REAL,
            beta REAL,
            PRIMARY KEY (context_key, candidate_name)
        )
    """)
    cursor.execute("INSERT INTO candidate_params VALUES ('key1', 'candidate_A', 10.0, 2.0)")
    cursor.execute("INSERT INTO candidate_params VALUES ('key1', 'candidate_B', 5.0, 5.0)")
    conn.commit()
    conn.close()

    # Verify db initially has beliefs
    beliefs = get_all_beliefs_sync(clean_db)
    assert len(beliefs) > 0

    # 1. Reset specific candidate
    with patch("sys.argv", [
        "bayesian-cortex", "reset",
        "--db-path", clean_db,
        "--candidate", "candidate_A"
    ]), patch("sys.stdout.write") as mock_stdout:
        main()
        output = "".join(call.args[0] for call in mock_stdout.call_args_list)
        assert "Reset beliefs for candidate 'candidate_A'" in output

    beliefs = get_all_beliefs_sync(clean_db)
    assert "candidate_A" not in beliefs["key1"]
    assert "candidate_B" in beliefs["key1"]

    # 2. Complete reset
    with patch("sys.argv", [
        "bayesian-cortex", "reset",
        "--db-path", clean_db
    ]), patch("sys.stdout.write") as mock_stdout:
        main()
        output = "".join(call.args[0] for call in mock_stdout.call_args_list)
        assert "Completely reset all candidate beliefs" in output

    beliefs = get_all_beliefs_sync(clean_db)
    assert len(beliefs) == 0


def test_cli_mcp():
    """Test that starting FastMCP server invokes creation and run."""
    with patch("sys.argv", [
        "bayesian-cortex", "mcp",
        "--db-path", "test_mcp.db",
        "--candidates", "c1,c2"
    ]), patch("bayesian_cortex.cli.create_mcp_server") as mock_create_mcp:
        mock_mcp_instance = MagicMock()
        mock_create_mcp.return_value = mock_mcp_instance

        main()

        mock_create_mcp.assert_called_once_with(db_path="test_mcp.db", candidates=["c1", "c2"])
        mock_mcp_instance.run.assert_called_once()
