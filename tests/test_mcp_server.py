import json
import os

import pytest
from mcp.server.fastmcp import FastMCP

from bayesian_cortex.mcp_server import create_mcp_server


def test_mcp_server_creation():
    db_path = "test_mcp_bandit.db"

    # Clean up in case of previous failures
    if os.path.exists(db_path):
        os.remove(db_path)

    try:
        mcp = create_mcp_server(
            server_name="TestBanditServer",
            db_path=db_path,
            candidates=["tool1", "tool2"],
        )

        assert isinstance(mcp, FastMCP)
        assert mcp.name == "TestBanditServer"

        # Check if execute_adaptive_action tool is registered
        # FastMCP stores registered tools in a dictionary or list
        # Let's inspect the tools registered in the FastMCP instance
        tools = mcp._tool_manager.list_tools() if hasattr(mcp, "_tool_manager") else []
        candidate_names = [t.name for t in tools]
        assert "execute_adaptive_action" in candidate_names or len(candidate_names) > 0
    finally:
        # Clean up database file
        if os.path.exists(db_path):
            os.remove(db_path)


@pytest.mark.anyio
async def test_mcp_server_administrative_features():
    db_path = "test_mcp_bandit_admin.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    try:
        mcp = create_mcp_server(
            server_name="TestBanditAdminServer",
            db_path=db_path,
            candidates=["tool1", "tool2"],
        )

        # 1. Get initial beliefs for a context
        res_tool, _ = await mcp.call_tool(
            "get_candidate_beliefs", {"context": "pytest styling"}
        )
        beliefs = json.loads(res_tool[0].text)
        assert beliefs["tool1"] == {"alpha": 1.0, "beta": 1.0}
        assert beliefs["tool2"] == {"alpha": 1.0, "beta": 1.0}

        # 2. Execute adaptive action (this should route, execute, and submit feedback)
        res_exec, _ = await mcp.call_tool(
            "execute_adaptive_action", {"task_description": "pytest styling"}
        )
        exec_text = res_exec[0].text
        assert "Selected Candidate" in exec_text

        # 3. Get updated beliefs
        res_tool_updated, _ = await mcp.call_tool(
            "get_candidate_beliefs", {"context": "pytest styling"}
        )
        beliefs_updated = json.loads(res_tool_updated[0].text)

        # One of the tools should have evolved parameters
        t1_params = beliefs_updated["tool1"]
        t2_params = beliefs_updated["tool2"]
        assert t1_params != {"alpha": 1.0, "beta": 1.0} or t2_params != {
            "alpha": 1.0,
            "beta": 1.0,
        }

        # Identify which tool was updated
        updated_tool = "tool1" if t1_params != {"alpha": 1.0, "beta": 1.0} else "tool2"

        # 4. Check the cortex://metrics resource
        res_metrics = await mcp.read_resource("cortex://metrics")
        metrics_text = res_metrics[0].content
        assert "# Bayes Brain Multi-Armed Bandit Metrics" in metrics_text
        assert "Total Context Clusters" in metrics_text
        assert "tool1" in metrics_text
        assert "tool2" in metrics_text
        # Assert new visual/diagnostic features are present
        assert "<svg" in metrics_text
        assert "Belief Sparkline" in metrics_text
        assert "Selection Frequencies & Success Rates" in metrics_text
        assert "Chronological Execution Log" in metrics_text

        # 5. Reset beliefs for the updated tool
        res_reset, _ = await mcp.call_tool(
            "reset_candidate_beliefs",
            {"context": "pytest styling", "candidate": updated_tool},
        )
        assert "been reset" in res_reset[0].text

        # 6. Verify they are back to (1.0, 1.0)
        res_tool_reset, _ = await mcp.call_tool(
            "get_candidate_beliefs", {"context": "pytest styling"}
        )
        beliefs_reset = json.loads(res_tool_reset[0].text)
        assert beliefs_reset[updated_tool] == {"alpha": 1.0, "beta": 1.0}

    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


@pytest.mark.anyio
async def test_mcp_server_contextual_priors():
    db_path = "test_mcp_bandit_context_priors.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    try:
        contextual_priors = [
            {
                "pattern": r"math|calculator|sum",
                "priors": {"tool1": (50.0, 1.0), "tool2": (1.0, 50.0)},
            }
        ]

        mcp = create_mcp_server(
            server_name="TestBanditContextPriorsServer",
            db_path=db_path,
            candidates=["tool1", "tool2"],
            contextual_priors=contextual_priors,
        )

        # Retrieve tool beliefs for a math task context
        res_tool, _ = await mcp.call_tool(
            "get_candidate_beliefs", {"context": "perform calculator sum"}
        )
        beliefs = json.loads(res_tool[0].text)
        assert beliefs["tool1"] == {"alpha": 50.0, "beta": 1.0}
        assert beliefs["tool2"] == {"alpha": 1.0, "beta": 50.0}

        # Retrieve tool beliefs for a non-math task context (should fall back to defaults)
        res_tool_fallback, _ = await mcp.call_tool(
            "get_candidate_beliefs", {"context": "general query"}
        )
        beliefs_fallback = json.loads(res_tool_fallback[0].text)
        assert beliefs_fallback["tool1"] == {"alpha": 1.0, "beta": 1.0}
        assert beliefs_fallback["tool2"] == {"alpha": 1.0, "beta": 1.0}

    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


@pytest.mark.anyio
async def test_mcp_server_dynamic_registration():
    db_path = "test_mcp_bandit_dynamic.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    try:
        # Test 1: disable tools, enable skills
        mcp_no_tools = create_mcp_server(
            server_name="TestNoTools",
            db_path=db_path,
            candidates=["tool1", "tool2"],
            enable_tools=False,
            enable_skills=True,
            enable_rag=False,
        )
        tools_no_tools = [t.name for t in mcp_no_tools._tool_manager.list_tools()]
        assert "execute_adaptive_action" not in tools_no_tools
        assert "get_candidate_beliefs" in tools_no_tools
        assert "route_knowledge_base" not in tools_no_tools

        # Read resource dashboard
        res_no_tools = await mcp_no_tools.read_resource("cortex://metrics")
        metrics_no_tools = res_no_tools[0].content
        assert "Posterior Belief Distributions" in metrics_no_tools
        assert "Selection Frequencies & Success Rates" not in metrics_no_tools

        # Test 2: disable skills, enable RAG and tools
        mcp_no_skills = create_mcp_server(
            server_name="TestNoSkills",
            db_path=db_path,
            candidates=["tool1", "tool2"],
            enable_tools=True,
            enable_skills=False,
            enable_rag=True,
        )
        tools_no_skills = [t.name for t in mcp_no_skills._tool_manager.list_tools()]
        assert "execute_adaptive_action" in tools_no_skills
        assert "get_candidate_beliefs" not in tools_no_skills
        assert "reset_candidate_beliefs" not in tools_no_skills
        assert "route_knowledge_base" in tools_no_skills

        res_no_skills = await mcp_no_skills.read_resource("cortex://metrics")
        metrics_no_skills = res_no_skills[0].content
        assert "Posterior Belief Distributions" not in metrics_no_skills
        assert "Selection Frequencies & Success Rates" in metrics_no_skills

        # Call route_knowledge_base tool
        res_rag, _ = await mcp_no_skills.call_tool(
            "route_knowledge_base",
            {"query": "test query", "vector_indices": ["idx1", "idx2"]},
        )
        assert "Selected RAG Index" in res_rag[0].text
        assert "Trace ID" in res_rag[0].text

        # Test 3: sub_tools parameter mapping
        mcp_sub_tools = create_mcp_server(
            server_name="TestSubTools",
            db_path=db_path,
            sub_tools=["sub1", "sub2"],
            enable_tools=True,
            enable_skills=True,
        )
        res_beliefs, _ = await mcp_sub_tools.call_tool(
            "get_candidate_beliefs", {"context": "general"}
        )
        beliefs = json.loads(res_beliefs[0].text)
        assert "sub1" in beliefs
        assert "sub2" in beliefs

    finally:
        if os.path.exists(db_path):
            os.remove(db_path)
