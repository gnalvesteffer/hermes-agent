"""Anti-looping mechanism for detecting repetitive thinking patterns.

When a model gets stuck in a loop -- repeatedly producing the same or very
similar reasoning/thinking content without taking action or making progress --
this module detects it and returns a nudge message to break the cycle.

Detection strategy:
  - Walk backwards through recent assistant messages.
  - Collect consecutive turns that have reasoning/thinking but no tool calls
    and no visible text after think blocks.
  - Compare token overlap between consecutive reasoning blocks.
  - If overlap exceeds a threshold for N consecutive turns, declare a loop.

The conversation loop in ``agent/conversation_loop.py`` calls
``check_for_thinking_loop()`` after an assistant message with no tool calls
but with thinking content is detected (no visible text after think blocks).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional


# ── Configuration ────────────────────────────────────────────────────────────

# Number of consecutive turns to look back for repetition.
_WINDOW_SIZE = 3

# Token overlap ratio threshold (0.0-1.0). If the new reasoning shares this
# fraction or more of tokens with any message in the window, it's a candidate.
_OVERLAP_THRESHOLD = 0.65

# Minimum consecutive matches to declare a loop.
_CONSECUTIVE_MATCHES_REQUIRED = 2

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


def _token_overlap(tokens_a: List[str], tokens_b: List[str]) -> float:
    """Jaccard-like overlap between two token lists."""
    if not tokens_a or not tokens_b:
        return 0.0
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


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


def check_for_thinking_loop(
    messages: List[dict],
    *,
    window_size: int = _WINDOW_SIZE,
    overlap_threshold: float = _OVERLAP_THRESHOLD,
    consecutive_required: int = _CONSECUTIVE_MATCHES_REQUIRED,
) -> Optional[str]:
    """Check if the recent conversation history shows a thinking loop.

    Scans the last ``window_size`` assistant messages for repetitive reasoning
    patterns.  Returns the nudge message string if a loop is detected, or
    ``None`` otherwise.

    Algorithm:
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
