import re
from typing import Any, List, Optional, Union

# Common English stop words to filter out before calculating faithfulness
DEFAULT_STOP_WORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "this",
    "that",
    "it",
    "they",
    "i",
    "you",
    "he",
    "she",
    "we",
    "us",
    "them",
    "my",
    "your",
    "his",
    "her",
    "our",
    "their",
    "as",
    "by",
    "from",
    "into",
    "through",
    "during",
    "including",
    "until",
    "against",
    "among",
    "throughout",
    "despite",
    "towards",
    "upon",
    "concerning",
    "about",
    "above",
    "after",
    "before",
    "behind",
    "below",
    "between",
    "under",
    "within",
    "without",
    "will",
    "would",
    "shall",
    "should",
    "can",
    "could",
    "may",
    "might",
    "must",
    "up",
    "down",
    "out",
    "off",
    "over",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "s",
    "t",
    "just",
    "don",
    "now",
}

# Regex patterns indicating a citation failure or retrieval failure
DEFAULT_FALLBACK_PATTERNS = [
    r"sorry.*(could not|couldn't|cannot) find",
    r"could not find.*(documentation|info|sources|information)",
    r"couldn't find.*(documentation|info|sources|information)",
    r"no information available",
    r"not found in the (documentation|sources|docs)",
    r"no sources found",
    r"i (do not|don't) know",
    r"unable to answer",
    r"cannot find",
    r"could not find",
    r"i (do not|don't) (have|possess) access",
    r"information is not present in",
]


def check_citation(
    response: str, fallback_patterns: Optional[List[str]] = None
) -> bool:
    """
    Verify that the LLM did not return a standard fallback indicating information was missing.

    Args:
        response: The generated text response from the LLM.
        fallback_patterns: Custom list of regex pattern strings to check. If None,
                           uses DEFAULT_FALLBACK_PATTERNS.

    Returns:
        bool: True if no fallback/failure patterns are found (successful retrieval citation),
              False if a fallback pattern is matched.
    """
    if not response:
        return False

    patterns = (
        fallback_patterns
        if fallback_patterns is not None
        else DEFAULT_FALLBACK_PATTERNS
    )

    for pattern in patterns:
        if re.search(pattern, response, re.IGNORECASE):
            return False

    return True


def calculate_faithfulness(
    response: str,
    source_chunks: Union[str, List[str]],
    stop_words: Optional[set] = None,
) -> float:
    """
    Calculate a lightweight token containment/overlap metric indicating how much of
    the generated answer is supported by the retrieved source chunks.

    Args:
        response: The generated text response from the LLM.
        source_chunks: A single string or list of retrieved source text chunks.
        stop_words: Set of words to exclude from overlap checks. Defaults to a standard set.

    Returns:
        float: A score between 0.0 and 1.0 representing faithfulness (percentage of unique
               non-stopword tokens in the response that exist in the sources).
    """
    if not response:
        return 0.0

    if isinstance(source_chunks, str):
        source_chunks = [source_chunks]

    # Clean and tokenize helper
    def tokenize(text: str) -> set:
        clean_text = text.lower()
        # Find all alphanumeric words
        tokens = re.findall(r"\b\w+\b", clean_text)
        return set(tokens)

    resp_tokens = tokenize(response)

    # Filter out stop words
    exclude = stop_words if stop_words is not None else DEFAULT_STOP_WORDS
    resp_tokens = {t for t in resp_tokens if t not in exclude}

    if not resp_tokens:
        # Response contains no meaningful tokens (all stop words or empty).
        # Return 0.0 — a content-free response has no faithfulness to the source.
        return 0.0

    # Tokenize source chunks
    source_text = " ".join(source_chunks)
    source_tokens = tokenize(source_text)
    source_tokens = {t for t in source_tokens if t not in exclude}

    # Intersect response tokens and source tokens
    overlap = resp_tokens.intersection(source_tokens)

    return len(overlap) / len(resp_tokens)


def evaluate_rag_success(
    response: str,
    source_chunks: Union[str, List[str]],
    faithfulness_threshold: float = 0.5,
    fallback_patterns: Optional[List[str]] = None,
    stop_words: Optional[set] = None,
) -> bool:
    """
    Automated check combining citation patterns and token overlap to assess RAG query success.

    Args:
        response: The generated text response from the LLM.
        source_chunks: A single string or list of retrieved source text chunks.
        faithfulness_threshold: Minimum overlap score required to count as True (0.0 to 1.0).
        fallback_patterns: Custom fallback regex patterns.
        stop_words: Custom set of stop words to exclude from overlap checks.

    Returns:
        bool: True if citation check passes AND faithfulness is >= threshold, else False.
    """
    if not check_citation(response, fallback_patterns):
        return False

    score = calculate_faithfulness(response, source_chunks, stop_words)
    return score >= faithfulness_threshold


def _map_ui_feedback_to_reward(value: Any) -> float:
    """
    Map common UI feedback indicators (thumbs up/down, True/False, numerical) to a 1.0 or 0.0 reward.
    """
    if isinstance(value, bool):
        return 1.0 if value else 0.0

    if isinstance(value, (int, float)):
        # Ensure we return a valid clamped reward
        val_float = float(value)
        if 0.0 <= val_float <= 1.0:
            return val_float
        raise ValueError(
            f"Numerical UI feedback must be between 0.0 and 1.0, got: {value}"
        )

    if isinstance(value, str):
        val_lower = value.strip().lower()
        if val_lower in (
            "thumbs_up",
            "thumbs-up",
            "thumbsup",
            "like",
            "upvote",
            "success",
            "yes",
            "true",
            "1",
        ):
            return 1.0
        if val_lower in (
            "thumbs_down",
            "thumbs-down",
            "thumbsdown",
            "dislike",
            "downvote",
            "failure",
            "no",
            "false",
            "0",
        ):
            return 0.0

    raise ValueError(f"Unsupported UI feedback value: {value}")


def process_ui_feedback(router: Any, trace_id: str, feedback_value: Any) -> float:
    """
    Process client-side UI feedback (like thumbs up / down) and update the synchronous router.

    Args:
        router: An instance of BayesianRouter.
        trace_id: Cryptographically signed or normal trace ID string.
        feedback_value: UI feedback value ("thumbs_up", "thumbs_down", True, False, etc.)

    Returns:
        float: The mapped reward value (1.0 or 0.0) applied to the router.
    """
    reward = _map_ui_feedback_to_reward(feedback_value)
    router.feedback_by_trace(trace_id=trace_id, reward=reward)
    return reward


async def aprocess_ui_feedback(
    router: Any, trace_id: str, feedback_value: Any
) -> float:
    """
    Process client-side UI feedback (like thumbs up / down) and update the asynchronous router.

    Args:
        router: An instance of AsyncBayesianRouter.
        trace_id: Cryptographically signed or normal trace ID string.
        feedback_value: UI feedback value ("thumbs_up", "thumbs_down", True, False, etc.)

    Returns:
        float: The mapped reward value (1.0 or 0.0) applied to the router.
    """
    reward = _map_ui_feedback_to_reward(feedback_value)
    await router.afeedback_by_trace(trace_id=trace_id, reward=reward)
    return reward
