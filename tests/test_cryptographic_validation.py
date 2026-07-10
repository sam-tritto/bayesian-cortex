"""
Cryptographic Validation Checks for Bayesian Cortex.

Verifies that signed trace IDs are bulletproof against manipulation and tampering.
Manually modifies a single character in the payload string and asserts that
feedback_by_trace and afeedback_by_trace raise a strict ValueError under strict mode.
"""

import pytest

from bayesian_cortex.router import AsyncBayesianRouter, BayesianRouter


def test_sync_trace_id_manipulation(mem_storage):
    """
    Sync Test: Captures a valid trace_id, modifies a single character in the payload part,
    and asserts that feedback_by_trace raises a strict ValueError.
    """
    router = BayesianRouter(storage=mem_storage, secret_key="secure_hmac_test_key")

    candidates = ["tool_a", "tool_b"]
    chosen, trace_id = router.route_with_trace("some_query", candidates)

    # Verify valid trace ID works
    router.feedback_by_trace(trace_id, success=True, strict=True)

    # Split the payload and signature
    assert "." in trace_id
    payload_part, sig_part = trace_id.split(".", 1)

    # Modify a single character in the base64 payload part
    char_list = list(payload_part)
    assert len(char_list) > 5, "Payload too short to modify safely"

    # Swap index 5 with a different base64 character to guarantee tampering
    original_char = char_list[5]
    new_char = "X" if original_char != "X" else "Y"
    char_list[5] = new_char
    tampered_payload = "".join(char_list)

    tampered_trace_id = f"{tampered_payload}.{sig_part}"

    # Verify that calling feedback_by_trace under strict mode raises a ValueError
    with pytest.raises(ValueError, match="Invalid or corrupted trace ID"):
        router.feedback_by_trace(tampered_trace_id, success=True, strict=True)


@pytest.mark.anyio
async def test_async_trace_id_manipulation(async_mem_storage):
    """
    Async Test: Captures a valid trace_id, modifies a single character in the payload part,
    and asserts that afeedback_by_trace raises a strict ValueError.
    """
    router = AsyncBayesianRouter(
        storage=async_mem_storage, secret_key="secure_hmac_test_key"
    )

    candidates = ["tool_a", "tool_b"]
    chosen, trace_id = await router.aroute_with_trace("some_query", candidates)

    # Verify valid trace ID works
    await router.afeedback_by_trace(trace_id, success=True, strict=True)

    # Split the payload and signature
    assert "." in trace_id
    payload_part, sig_part = trace_id.split(".", 1)

    # Modify a single character in the base64 payload part
    char_list = list(payload_part)
    assert len(char_list) > 5, "Payload too short to modify safely"

    original_char = char_list[5]
    new_char = "X" if original_char != "X" else "Y"
    char_list[5] = new_char
    tampered_payload = "".join(char_list)

    tampered_trace_id = f"{tampered_payload}.{sig_part}"

    # Verify that calling afeedback_by_trace under strict mode raises a ValueError
    with pytest.raises(ValueError, match="Invalid or corrupted trace ID"):
        await router.afeedback_by_trace(tampered_trace_id, success=True, strict=True)
