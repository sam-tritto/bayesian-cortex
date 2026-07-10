from unittest.mock import Mock

import pytest

from bayesian_cortex import (
    AsyncBayesianRouter,
    BayesianRouter,
    aprocess_ui_feedback,
    calculate_faithfulness,
    check_citation,
    evaluate_rag_success,
    process_ui_feedback,
)


def test_check_citation_default_patterns():
    # Citation failure scenarios
    assert check_citation("I'm sorry, I couldn't find anything in the docs.") is False
    assert check_citation("I am sorry, but I could not find that.") is False
    assert check_citation("I do not have access to that information.") is False
    assert (
        check_citation("No information available on parental leave rollover.") is False
    )
    assert check_citation("No sources found for the query.") is False
    assert check_citation("I do not know.") is False
    assert check_citation("I don't know the answer.") is False
    assert check_citation("Unable to answer due to missing sources.") is False
    assert check_citation("Sorry, but I cannot find that.") is False

    # Success scenarios (valid answer containing matching tokens but not in fallback syntax)
    assert (
        check_citation(
            "Our policy on parental leave rollover allows up to 2 weeks to be rolled over."
        )
        is True
    )
    assert check_citation("The user has access to the billing system.") is True


def test_check_citation_custom_patterns():
    custom_patterns = [r"custom fallback phrase", r"access denied"]
    assert (
        check_citation("Sorry, this is a custom fallback phrase.", custom_patterns)
        is False
    )
    assert check_citation("System says access denied.", custom_patterns) is False
    assert (
        check_citation("I couldn't find it", custom_patterns) is True
    )  # Default pattern ignored


def test_calculate_faithfulness():
    # Clean overlap
    source = "The quick brown fox jumps over the lazy dog."
    response = "A brown fox jumps."
    # response tokens (lowercase): {"a", "brown", "fox", "jumps"}
    # stop words: "a" is a stop word. So meaningful response tokens = {"brown", "fox", "jumps"}
    # source tokens: {"quick", "brown", "fox", "jumps", "lazy", "dog"} ("the", "over" etc. filtered)
    # Intersect: {"brown", "fox", "jumps"}. Faithfulness: 3 / 3 = 1.0
    assert calculate_faithfulness(response, source) == 1.0

    # Half overlap
    response_half = "A brown fox jumps high in the sky."
    # response tokens: {"brown", "fox", "jumps", "high", "sky"} (5 tokens, excluding "a", "in", "the")
    # source tokens: {"quick", "brown", "fox", "jumps", "lazy", "dog"}
    # Intersect: {"brown", "fox", "jumps"} (3 tokens)
    # Faithfulness = 3 / 5 = 0.6
    assert calculate_faithfulness(response_half, source) == pytest.approx(0.6)

    # No overlap
    response_none = "Green apples are sweet."
    # response tokens: {"green", "apples", "sweet"} ("are" excluded)
    # source tokens: {"quick", "brown", "fox", "jumps", "lazy", "dog"}
    # Intersect: {}
    # Faithfulness = 0 / 3 = 0.0
    assert calculate_faithfulness(response_none, source) == 0.0

    # List of source chunks
    source_chunks = ["The quick brown fox", "jumps over the lazy dog."]
    assert calculate_faithfulness("A brown fox jumps.", source_chunks) == 1.0

    # Empty response
    assert calculate_faithfulness("", source) == 0.0
    assert (
        calculate_faithfulness("the a", source) == 0.0
    )  # Only stop words → no meaningful content → 0.0


def test_evaluate_rag_success():
    source = "Employees get 4 weeks of paid vacation yearly."

    # Passes both checks
    assert (
        evaluate_rag_success(
            "Employees get 4 weeks of vacation.", source, faithfulness_threshold=0.5
        )
        is True
    )

    # Fails citation check (matches fallback)
    assert (
        evaluate_rag_success("I'm sorry, I couldn't find vacation info.", source)
        is False
    )

    # Fails faithfulness check (below threshold)
    # response tokens: {"unlimited", "sick", "leave"} ("we", "have" excluded)
    # source tokens: {"employees", "get", "4", "weeks", "paid", "vacation", "yearly"}
    # Overlap = 0.0
    assert (
        evaluate_rag_success(
            "We have unlimited sick leave.", source, faithfulness_threshold=0.5
        )
        is False
    )


def test_evaluate_rag_success_stop_word_only_response():
    """Regression: a response made entirely of stop words must NOT score 1.0 faithfulness.

    Strings like "I will not do that" collapse to zero meaningful tokens after
    stop-word filtering. The old code returned 1.0 in that case to avoid
    division-by-zero, which incorrectly caused evaluate_rag_success to return
    True and rewarded the bandit for a useless answer.
    """
    source = "Employees get 4 weeks of paid vacation yearly."
    stop_word_response = "I will not do that"

    assert calculate_faithfulness(stop_word_response, source) == 0.0
    assert (
        evaluate_rag_success(stop_word_response, source, faithfulness_threshold=0.5)
        is False
    )


def test_evaluate_rag_success_custom_parameters():
    # Make sure we can pass custom parameters through evaluate_rag_success
    source = "Alpha Beta Gamma"
    response = "Delta Epsilon"
    assert (
        evaluate_rag_success(
            response=response,
            source_chunks=source,
            faithfulness_threshold=0.1,
            fallback_patterns=[r"not found"],
            stop_words=set(),
        )
        is False
    )  # Delta, Epsilon not in Alpha Beta Gamma -> 0.0 < 0.1


def test_process_ui_feedback_sync():
    # Mock router
    mock_router = Mock(spec=BayesianRouter)

    # Boolean
    process_ui_feedback(mock_router, "trace_1", True)
    mock_router.feedback_by_trace.assert_called_with(trace_id="trace_1", reward=1.0)

    process_ui_feedback(mock_router, "trace_2", False)
    mock_router.feedback_by_trace.assert_called_with(trace_id="trace_2", reward=0.0)

    # String thumbs up
    for thumbs_up in [
        "thumbs_up",
        "thumbs-up",
        "ThumbsUp",
        "like",
        "upvote",
        "success",
        "yes",
        "true",
        "1",
    ]:
        process_ui_feedback(mock_router, "trace_3", thumbs_up)
        mock_router.feedback_by_trace.assert_called_with(trace_id="trace_3", reward=1.0)

    # String thumbs down
    for thumbs_down in [
        "thumbs_down",
        "thumbs-down",
        "ThumbsDown",
        "dislike",
        "downvote",
        "failure",
        "no",
        "false",
        "0",
    ]:
        process_ui_feedback(mock_router, "trace_4", thumbs_down)
        mock_router.feedback_by_trace.assert_called_with(trace_id="trace_4", reward=0.0)

    # Numeric pass-through
    process_ui_feedback(mock_router, "trace_5", 0.7)
    mock_router.feedback_by_trace.assert_called_with(trace_id="trace_5", reward=0.7)

    # Out of bounds numerical feedback
    with pytest.raises(
        ValueError, match="Numerical UI feedback must be between 0.0 and 1.0"
    ):
        process_ui_feedback(mock_router, "trace_6", 1.5)

    # Invalid string
    with pytest.raises(ValueError, match="Unsupported UI feedback value"):
        process_ui_feedback(mock_router, "trace_7", "meh")


@pytest.mark.anyio
async def test_aprocess_ui_feedback_async():
    # Async mock router
    mock_router = Mock(spec=AsyncBayesianRouter)

    async def dummy_feedback_by_trace(trace_id, reward):
        mock_router.called_trace = trace_id
        mock_router.called_reward = reward

    mock_router.afeedback_by_trace = dummy_feedback_by_trace

    # Boolean
    reward = await aprocess_ui_feedback(mock_router, "trace_1", True)
    assert reward == 1.0
    assert mock_router.called_trace == "trace_1"
    assert mock_router.called_reward == 1.0

    # String thumbs down
    reward = await aprocess_ui_feedback(mock_router, "trace_2", "thumbs_down")
    assert reward == 0.0
    assert mock_router.called_trace == "trace_2"
    assert mock_router.called_reward == 0.0
