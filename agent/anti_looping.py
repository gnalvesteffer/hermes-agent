"""Anti-looping mechanism for detecting repetitive thinking patterns.

When a model gets stuck in a loop -- repeatedly producing the same or very
similar reasoning/thinking content without taking action or making progress --
this module detects it and returns a nudge message to break the cycle.

Detection strategy:
  - Intra-message: split a single response's reasoning into paragraph blocks,
    detect repeated/near-duplicate blocks using trigram overlap.
  - Inter-message: walk backwards through recent assistant messages, compare
    token overlap between consecutive reasoning blocks.
  - If either check triggers, return the nudge message.

The conversation loop in ``agent/conversation_loop.py`` calls
``check_for_thinking_loop()`` after an assistant message with no tool calls
but with thinking content is detected (no visible text after think blocks).
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Configuration ────────────────────────────────────────────────────────────

# Number of consecutive turns to look back for inter-message repetition.
_INTER_WINDOW_SIZE = 3

# Token overlap ratio threshold (0.0-1.0) for inter-message comparison.
_INTER_OVERLAP_THRESHOLD = 0.65

# Minimum consecutive matches to declare an inter-message loop.
_INTER_CONSECUTIVE_MATCHES_REQUIRED = 2

# Number of paragraph blocks to look back within a single message.
_INTRA_WINDOW_SIZE = 4

# Trigram overlap threshold for intra-message detection.
# Trigrams catch near-duplicates much better than single-token Jaccard,
# because "screen X coordinate" vs "screen Y coordinate" shares most
# trigrams even though they differ by one word.
_INTRA_OVERLAP_THRESHOLD = 0.5

# Minimum consecutive intra-message matches to declare a loop.
_INTRA_CONSECUTIVE_MATCHES_REQUIRED = 2

# The nudge message injected when a loop is detected.
_NUDGE_MESSAGE = (
    "You've been repeating the same thoughts without making progress. "
    "Try a different approach or take action instead of continuing to think."
)


def _tokenize(text: str) -> List[str]:
    """Simple tokeniser: lowercase, split on whitespace/punctuation."""
    if not text:
        return []
    # Normalise whitespace and split on word boundaries.
    return re.findall(r"[a-zA-Z0-9_]+|[^\w\s]", (text or "").lower())


def _trigrams(text: str) -> List[str]:
    """Generate character trigrams from text for near-duplicate detection."""
    if not text:
        return []
    t = text.lower().strip()
    return [t[i:i+3] for i in range(len(t) - 2)]


def _word_ngrams(text: str, n: int = 4) -> List[str]:
    """Generate word n-grams from text.

    Uses lowercase words to catch repeated phrases like
    'thinking about this more carefully' regardless of surrounding context.
    """
    if not text:
        return []
    words = re.findall(r"[a-zA-Z0-9_]+", (text or "").lower())
    if len(words) < n:
        return []
    return [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]


def _token_overlap(tokens_a: List[str], tokens_b: List[str]) -> float:
    """Jaccard-like overlap between two token lists."""
    if not tokens_a or not tokens_b:
        return 0.0
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _trigram_overlap(tris_a: List[str], tris_b: List[str]) -> float:
    """Jaccard overlap between two trigram sets."""
    if not tris_a or not tris_b:
        return 0.0
    set_a = set(tris_a)
    set_b = set(tris_b)
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _word_ngram_overlap(ng_a: List[str], ng_b: List[str]) -> float:
    """Jaccard overlap between two word-n-gram sets."""
    if not ng_a or not ng_b:
        return 0.0
    set_a = set(ng_a)
    set_b = set(ng_b)
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


# Inline think-tag patterns used by various models/providers.
# Mirrors the extraction logic in agent.agent_runtime_helpers.extract_reasoning()
# so that _extract_reasoning_from_message finds reasoning regardless of format.
_INLINE_THINK_PATTERNS = (
    r"```(.*?)```",
    r"</?think>",
    r"<thinking>(.*?)</thinking>",
    r"<thought>(.*?)</thought>",
    r"<reasoning>(.*?)</reasoning>",
    r"<REASONING_SCRATCHPAD>(.*?)</REASONING_SCRATCHPAD>",
)


def _split_paragraphs(text: str) -> List[str]:
    """Split text into paragraph blocks (separated by blank lines).

    Handles both single-newline and double-newline separators to cover
    different model output formats.
    """
    if not text:
        return []
    # Split on one or more blank lines (one or more consecutive newlines),
    # filter empty blocks. This covers both \n\n and \n separators.
    blocks = re.split(r"\n\s*\n", text)
    return [b.strip() for b in blocks if b.strip()]


def _extract_reasoning_from_message(msg: dict) -> Optional[str]:
    """Extract reasoning text from an assistant message dict.

    Checks the structured ``reasoning`` field first, then falls back to
    inline think blocks in ``content`` (both markdown code fences and
    XML-style tags).  Mirrors the extraction logic in
    ``agent.agent_runtime_helpers.extract_reasoning`` so that reasoning
    is found regardless of output format.

    Fallback: if no structured reasoning field, no reasoning_content, and
    no inline think blocks are present but the message has non-empty content
    (and no tool calls), treat the entire content as potential reasoning.
    This catches models like Qwen3.x via llama.cpp that emit reasoning as
    plain text without any wrapping tags.
    """
    # Structured reasoning field (from API).
    reasoning = msg.get("reasoning") or ""
    if isinstance(reasoning, str) and reasoning.strip():
        logger.debug(
            "anti_loop: extracted %d chars from 'reasoning' field",
            len(reasoning),
        )
        return reasoning.strip()

    # Inline think blocks in content — try all known tag formats.
    content = msg.get("content") or ""
    if isinstance(content, str):
        for pattern in _INLINE_THINK_PATTERNS:
            blocks = re.findall(pattern, content, flags=re.DOTALL)
            combined = "\n\n".join(b.strip() for b in blocks if b.strip())
            if combined:
                logger.debug(
                    "anti_loop: extracted %d chars from inline pattern '%s'",
                    len(combined),
                    pattern[:40],
                )
                return combined

    # Structured reasoning_content field.
    rc = msg.get("reasoning_content") or ""
    if isinstance(rc, str) and rc.strip():
        logger.debug(
            "anti_loop: extracted %d chars from 'reasoning_content' field",
            len(rc),
        )
        return rc.strip()

    # Fallback for models that emit reasoning as plain text without any
    # wrapping tags (e.g. Qwen3.x via llama.cpp).  If there is no structured
    # reasoning at all and the content looks like it could be reasoning
    # rather than actionable output, treat it as such.  We only do this
    # when the message has no tool calls — a model that actually produced
    # useful output would typically include tool usage or a clear directive.
    if isinstance(content, str) and content.strip():
        # Only use untagged content as reasoning when there are no tool calls;
        # otherwise we'd falsely treat normal assistant responses as reasoning.
        has_tool_calls = msg.get("tool_calls")
        if not has_tool_calls:
            logger.debug(
                "anti_loop: treating %d chars of untagged content as reasoning",
                len(content),
            )
            return content.strip()

    logger.debug("anti_loop: no reasoning found in message keys=%s", list(msg.keys()))
    return None


def _check_intra_message_loop(reasoning_text: str) -> bool:
    """Check for repetition within a single reasoning block.

    Uses two complementary checks with word 4-grams (catches repeated phrases):
      1. Consecutive overlap — catches near-duplicate adjacent paragraphs.
      2. Average pairwise overlap — catches when many paragraphs share common
         phrases even if no two consecutive pairs exceed the threshold.
      3. Recurring sentence starters — detects models that keep restarting
         with the same phrasing patterns (e.g., "Actually, thinking about",
         "But wait", "Let me reconsider").

    Returns True if a loop is detected within this message's reasoning.
    """
    paragraphs = _split_paragraphs(reasoning_text)
    if len(paragraphs) < _INTRA_CONSECUTIVE_MATCHES_REQUIRED:
        return False

    # Build word 4-grams for each paragraph
    paragraph_ngs = []
    for p in paragraphs:
        ng = set(_word_ngrams(p, n=4))
        if ng:
            paragraph_ngs.append(ng)

    if not paragraph_ngs:
        return False

    # ── Check 1: consecutive word-ngram overlap ────────────────────────
    consecutive_matches = 0
    for i in range(1, len(paragraph_ngs)):
        overlap = _word_ngram_overlap(
            list(paragraph_ngs[i]), list(paragraph_ngs[i - 1])
        )
        if overlap >= _INTRA_OVERLAP_THRESHOLD:
            consecutive_matches += 1
            if consecutive_matches >= _INTRA_CONSECUTIVE_MATCHES_REQUIRED:
                return True
        else:
            consecutive_matches = 0

    # ── Check 2: average pairwise word-ngram overlap ───────────────────
    if len(paragraph_ngs) >= _INTRA_WINDOW_SIZE:
        total_overlap = 0.0
        pair_count = 0
        for i in range(len(paragraph_ngs)):
            for j in range(i + 1, len(paragraph_ngs)):
                overlap = _word_ngram_overlap(
                    list(paragraph_ngs[i]), list(paragraph_ngs[j])
                )
                total_overlap += overlap
                pair_count += 1
        if pair_count > 0:
            avg_overlap = total_overlap / pair_count
            if avg_overlap >= _INTRA_OVERLAP_THRESHOLD * 0.7:
                return True

    # ── Check 3: recurring sentence starters ───────────────────────────
    # Models in a loop often restart with the same phrasing patterns.
    # Detect repeated sentence-starting phrases across paragraphs.
    _RECURRING_STARTERS = [
        "actually, thinking about",
        "but wait",
        "let me reconsider",
        "let me try",
        "actually, i'm overcomplicating",
        "actually, let me step back",
        "i'm realizing this approach",
    ]

    starter_counts = {s: 0 for s in _RECURRING_STARTERS}
    for p in paragraphs:
        lower_p = p.lower()
        for starter in _RECURRING_STARTERS:
            if lower_p.startswith(starter) or lower_p.strip().startswith(starter):
                starter_counts[starter] += 1

    # If any recurring starter appears >= 2 times, it's a loop signal.
    max_count = max(starter_counts.values()) if starter_counts else 0
    if max_count >= 2:
        return True

    return False


def check_for_thinking_loop(
    messages: List[dict],
    *,
    window_size: int = _INTER_WINDOW_SIZE,
    overlap_threshold: float = _INTER_OVERLAP_THRESHOLD,
    consecutive_required: int = _INTER_CONSECUTIVE_MATCHES_REQUIRED,
) -> Optional[str]:
    """Check if the recent conversation history shows a thinking loop.

    Performs two checks:
      1. Intra-message: checks each assistant message's reasoning for
         repeated paragraph blocks (near-duplicate detection via trigrams).
      2. Inter-message: scans the last ``window_size`` assistant messages
         for repetitive reasoning patterns using token overlap.

    Returns the nudge message string if a loop is detected, or
    ``None`` otherwise.

    Algorithm (inter-message):
      1. Walk backwards through ``messages``, collecting consecutive assistant
         messages that have non-empty reasoning/thinking content but no tool
         calls and no visible text after think blocks.
      2. For each new reasoning block, compare token overlap against all
         previous blocks in the window.
      3. Track how many consecutive matches occur. If ``consecutive_required``
         consecutive messages all overlap with their predecessor above the
         threshold, a loop is declared.

    Args:
        messages: The full conversation message list (may contain assistant,
            user, tool, system messages).
        window_size: How many recent assistant messages to examine.
        overlap_threshold: Token overlap ratio required to count as a match.
        consecutive_required: Minimum consecutive matches to declare a loop.

    Returns:
        The nudge message string if a thinking loop is detected, ``None``
        otherwise.
    """
    # ── Intra-message check ────────────────────────────────────────────
    # Check the most recent assistant message for repeated paragraphs
    # within its own reasoning content.
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "assistant":
            continue
        reasoning = _extract_reasoning_from_message(msg)
        if reasoning and _check_intra_message_loop(reasoning):
            return _NUDGE_MESSAGE
        break  # Only check the most recent assistant message

    # ── Inter-message check ────────────────────────────────────────────
    # Collect recent assistant messages with reasoning but no tool calls.
    candidates: List[dict] = []
    for msg in reversed(messages):
        if len(candidates) >= window_size:
            break
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "assistant":
            continue
        # Skip messages that have tool calls -- those are productive turns.
        if msg.get("tool_calls"):
            continue
        # Skip messages with visible content after think blocks -- the model
        # actually produced something useful.  However, if there is no
        # structured ``reasoning`` field at all, treat untagged content as
        # potential reasoning (catches models like Qwen3.x via llama.cpp
        # that emit reasoning as plain text without wrapping tags).
        content = msg.get("content") or ""
        has_structured_reasoning = bool(msg.get("reasoning"))
        if isinstance(content, str) and content.strip() and has_structured_reasoning:
            # Quick check: does it have actual text outside think tags?
            stripped = re.sub(
                r"</?(?:REASONING_SCRATCHPAD|think|reasoning)>",
                "",
                content,
                flags=re.IGNORECASE,
            ).strip()
            if stripped:
                continue
        # Skip messages that are synthetic scaffolding (empty recovery, etc.).
        if msg.get("_empty_recovery_synthetic") or msg.get(
            "_empty_terminal_sentinel"
        ):
            continue
        candidates.append(msg)

    if len(candidates) < consecutive_required:
        return None

    # Extract reasoning from each candidate.
    reasonings: List[Optional[str]] = []
    for c in candidates:
        r = _extract_reasoning_from_message(c)
        reasonings.append(r)

    # Check for consecutive matches.
    consecutive_matches = 0
    for i in range(1, len(reasonings)):
        r_curr = reasonings[i]
        r_prev = reasonings[i - 1]
        if r_curr is None or r_prev is None:
            continue
        tokens_new = _tokenize(r_curr)
        tokens_prev = _tokenize(r_prev)
        overlap = _token_overlap(tokens_new, tokens_prev)
        if overlap >= overlap_threshold:
            consecutive_matches += 1
            if consecutive_matches >= consecutive_required:
                return _NUDGE_MESSAGE
        else:
            consecutive_matches = 0

    return None
