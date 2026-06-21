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

import re
from typing import Dict, List, Optional


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


def _split_paragraphs(text: str) -> List[str]:
    """Split text into paragraph blocks (separated by blank lines)."""
    if not text:
        return []
    # Split on double-newline or more, filter empty blocks.
    blocks = re.split(r"\n{2,}", text)
    return [b.strip() for b in blocks if b.strip()]


def _extract_reasoning_from_message(msg: dict) -> Optional[str]:
    """Extract reasoning text from an assistant message dict.

    Checks the structured ``reasoning`` field first, then falls back to
    inline think blocks in ``content``.
    """
    # Structured reasoning field (from API).
    reasoning = msg.get("reasoning") or ""
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()

    # Inline think blocks in content.
    content = msg.get("content") or ""
    if isinstance(content, str):
        think_blocks = re.findall(
            r"```(.*?)```", content, flags=re.DOTALL
        )
        combined = "\n\n".join(b.strip() for b in think_blocks if b.strip())
        if combined:
            return combined

    # Structured reasoning_content field.
    rc = msg.get("reasoning_content") or ""
    if isinstance(rc, str) and rc.strip():
        return rc.strip()

    return None


def _check_intra_message_loop(reasoning_text: str) -> bool:
    """Check for repetition within a single reasoning block.

    Splits the reasoning into paragraph blocks and checks if consecutive
    blocks have high trigram overlap (catches near-duplicates that differ
    by only a few words).

    Returns True if a loop is detected within this message's reasoning.
    """
    paragraphs = _split_paragraphs(reasoning_text)
    if len(paragraphs) < _INTRA_CONSECUTIVE_MATCHES_REQUIRED:
        return False

    consecutive_matches = 0
    for i in range(1, len(paragraphs)):
        tris_curr = set(_trigrams(paragraphs[i]))
        tris_prev = set(_trigrams(paragraphs[i - 1]))
        if not tris_curr or not tris_prev:
            continue
        overlap = _trigram_overlap(list(tris_curr), list(tris_prev))
        if overlap >= _INTRA_OVERLAP_THRESHOLD:
            consecutive_matches += 1
            if consecutive_matches >= _INTRA_CONSECUTIVE_MATCHES_REQUIRED:
                return True
        else:
            consecutive_matches = 0

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
        # actually produced something useful.
        content = msg.get("content") or ""
        if isinstance(content, str) and content.strip():
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
