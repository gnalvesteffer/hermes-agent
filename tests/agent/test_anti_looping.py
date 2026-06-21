"""Tests for agent/anti_looping.py — thinking-loop detection."""

from __future__ import annotations

import pytest

from agent.anti_looping import check_for_thinking_loop


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _assistant(
    reasoning: str = "", content: str = "", tool_calls=None, **kwargs
) -> dict:
    """Helper to build an assistant message dict."""
    msg: dict = {
        "role": "assistant",
        "content": content,
        "reasoning": reasoning,
    }
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    msg.update(kwargs)
    return msg


# ── Intra-message tests (trigram-based near-duplicate detection) ─────────────


class TestIntraMessageLoop:
    """Cases where a loop exists within a SINGLE message's reasoning."""

    def test_exact_repeat_paragraphs(self):
        """Identical paragraphs repeated should trigger."""
        paragraph = "Actually, thinking about this more carefully: when looking straight ahead horizontally, all trajectory points would naturally project to nearly the same screen X coordinate since they are aligned along your line of sight"
        msgs = [_assistant(reasoning="\n\n".join([paragraph] * 4))]
        assert check_for_thinking_loop(msgs) is not None

    def test_near_duplicate_paragraphs(self):
        """Paragraphs differing by one word should trigger (trigram overlap ~0.97)."""
        msgs = [_assistant(
            reasoning=(
                "Actually, thinking about this more carefully: when looking "
                "straight ahead horizontally, all trajectory points would "
                "naturally project to nearly the same screen X coordinate"
                "\n\n"
                "Actually, thinking about this more carefully: when looking "
                "straight ahead horizontally, all trajectory points would "
                "naturally project to nearly the same screen Y coordinate"
                "\n\n"
                "Actually, thinking about this more carefuly: when looking "
                "straight ahead horizontally, all trajectory points would "
                "naturally project to nearly the same screen Z coordinate"
            )
        )]
        assert check_for_thinking_loop(msgs) is not None

    def test_no_intra_repeat(self):
        """Distinct paragraphs should NOT trigger."""
        msgs = [_assistant(
            reasoning=(
                "I think about trajectory and how it affects the game physics"
                "\n\n"
                "Looking at the database schema, I need to join three tables"
                "\n\n"
                "The API endpoint returns a JSON object with nested arrays"
            )
        )]
        assert check_for_thinking_loop(msgs) is None

    def test_single_paragraph_no_loop(self):
        """A single paragraph can't loop."""
        msgs = [_assistant(reasoning="just one paragraph")]
        assert check_for_thinking_loop(msgs) is None

    def test_recurring_sentence_starters(self):
        """Models in a loop restart with the same phrasing patterns."""
        msgs = [_assistant(
            reasoning=(
                "But wait - I'm realizing this approach might not be what they want.\n\n"
                "Actually, thinking about this more carefully: when looking straight ahead horizontally,\n"
                "all trajectory points would naturally project to nearly the same screen X coordinate.\n\n"
                "Let me try a different approach: instead of using camera forward as the primary direction.\n\n"
                "But wait, that would mislead players about where their shots actually land."
            )
        )]
        assert check_for_thinking_loop(msgs) is not None

    def test_no_recurring_starters(self):
        """Different sentence starters should NOT trigger."""
        msgs = [_assistant(
            reasoning=(
                "I think about trajectory and how it affects the game physics.\n\n"
                "Looking at the database schema, I need to join three tables.\n\n"
                "The API endpoint returns a JSON object with nested arrays."
            )
        )]
        assert check_for_thinking_loop(msgs) is None


# ── Inter-message tests (token overlap between consecutive messages) ─────────


class TestInterMessageLoop:
    """Cases where a loop spans multiple assistant messages."""

    def test_exact_repeat_across_messages(self):
        msgs = [
            _assistant(
                "Actually, thinking about this more carefully: when looking "
                "straight ahead horizontally, all trajectory points would "
                "naturally project to nearly the same screen X coordinate"
            ),
            _assistant(
                "Actually, thinking about this more carefully: when looking "
                "straight ahead horizontally, all trajectory points would "
                "naturally project to nearly the same screen Y coordinate"
            ),
            _assistant(
                "Actually, thinking about this more carefully: when looking "
                "straight ahead horizontally, all trajectory points would "
                "naturally project to nearly the same screen Z coordinate"
            ),
        ]
        result = check_for_thinking_loop(msgs)
        assert result is not None

    def test_high_overlap(self):
        """Similar but not identical reasoning should still trigger."""
        msgs = [
            _assistant(
                "Actually, thinking about this more carefully: when looking "
                "straight ahead horizontally, all trajectory points would "
                "naturally project to nearly the same screen X coordinate"
            ),
            _assistant(
                "Actually, thinking about this more carefully: when looking "
                "straight ahead horizontally, all trajectory points would "
                "naturally project to nearly the same screen Y coordinate"
            ),
            _assistant(
                "Actually, thinking about this more carefully: when looking "
                "straight ahead horizontally, all trajectory points would "
                "naturally project to nearly the same screen Z coordinate"
            ),
        ]
        result = check_for_thinking_loop(msgs)
        assert result is not None

    def test_consecutive_required_threshold(self):
        """With consecutive_required=3, 2 repeats should NOT trigger."""
        msgs = [
            _assistant("I think about trajectory and physics"),
            _assistant("I think about bullet drop and gravity"),
        ]
        result = check_for_thinking_loop(msgs, consecutive_required=3)
        assert result is None

    def test_nudge_message_content(self):
        """The returned nudge should be the expected message."""
        from agent import anti_looping
        msgs = [
            _assistant(
                "Actually, thinking about this more carefully: when looking "
                "straight ahead horizontally, all trajectory points would "
                "naturally project to nearly the same screen X coordinate"
            ),
            _assistant(
                "Actually, thinking about this more carefully: when looking "
                "straight ahead horizontally, all trajectory points would "
                "naturally project to nearly the same screen Y coordinate"
            ),
            _assistant(
                "Actually, thinking about this more carefully: when looking "
                "straight ahead horizontally, all trajectory points would "
                "naturally project to nearly the same screen Z coordinate"
            ),
        ]
        result = check_for_thinking_loop(msgs)
        assert result == anti_looping._NUDGE_MESSAGE

    def test_realistic_loop_pattern(self):
        """Simulate the user's example of a model stuck in repetitive thinking."""
        msgs = [
            _assistant(
                "Actually, thinking about this more carefully: when looking "
                "straight ahead horizontally, all trajectory points would "
                "naturally project to nearly the same screen X coordinate "
                "since they are aligned along your line of sight"
            ),
            _assistant(
                "Actually, thinking about this more carefully: when looking "
                "straight ahead horizontally, all trajectory points would "
                "naturally project to nearly the same screen X coordinate "
                "since they are aligned along your line of sight"
            ),
            _assistant(
                "Actually, thinking about this more carefully: when looking "
                "straight ahead horizontally, all trajectory points would "
                "naturally project to nearly the same screen X coordinate "
                "since they are aligned along your line of sight"
            ),
        ]
        result = check_for_thinking_loop(msgs)
        assert result is not None


class TestNoLoopDetected:
    """Cases where a loop should NOT be detected."""

    def test_empty_messages(self):
        assert check_for_thinking_loop([]) is None

    def test_single_assistant_message(self):
        msgs = [_assistant("thinking about this", "answer")]
        assert check_for_thinking_loop(msgs) is None

    def test_different_reasoning(self):
        """Completely different reasoning should not trigger."""
        msgs = [
            _assistant(
                "I think the best approach is to use a binary search algorithm"
            ),
            _assistant(
                "Looking at the database schema, I need to join three tables"
            ),
            _assistant(
                "The API endpoint returns a JSON object with nested arrays"
            ),
        ]
        assert check_for_thinking_loop(msgs) is None

    def test_has_tool_calls_breaks_pattern(self):
        """Tool-call turns are productive and should not count."""
        msgs = [
            _assistant(
                "repeating reasoning about trajectory",
                tool_calls=[{"id": "1"}],
            ),
            _assistant("repeating reasoning about physics"),
            _assistant("repeating reasoning about camera"),
        ]
        assert check_for_thinking_loop(msgs) is None

    def test_has_visible_content_breaks_pattern(self):
        """Messages with actual content after think blocks are productive."""
        msgs = [
            _assistant(
                "thinking about this",
                content="Actually, I'll do X.",
            ),
            _assistant("thinking again", content="Now doing Y."),
            _assistant("more thinking", content="Final answer: Z."),
        ]
        assert check_for_thinking_loop(msgs) is None

    def test_synthetic_scaffolding_skipped(self):
        """Empty recovery scaffolding should be skipped."""
        msgs = [
            _assistant("repeating reasoning 1"),
            {
                "role": "assistant",
                "content": "(empty)",
                "reasoning": "repeating reasoning 2",
                "_empty_recovery_synthetic": True,
            },
            _assistant("repeating reasoning 3"),
        ]
        # Only 1 real candidate (the third one), so no loop.
        assert check_for_thinking_loop(msgs) is None

    def test_high_threshold_prevents_false_positive(self):
        """High overlap threshold should prevent detection of similar but different."""
        msgs = [
            _assistant(
                "I think about trajectory and how it affects the game physics"
            ),
            _assistant(
                "I think about bullet drop and gravity calculations for accuracy"
            ),
            _assistant(
                "I think about camera forward direction and rendering pipeline"
            ),
        ]
        result = check_for_thinking_loop(msgs, overlap_threshold=0.95)
        assert result is None

    def test_non_dict_messages_ignored(self):
        """Non-dict entries in the message list should be skipped."""
        msgs = [
            "not a dict",
            123,
            _assistant("completely different thought one"),
            _assistant("completely different thought two"),
            _assistant("completely different thought three"),
        ]
        assert check_for_thinking_loop(msgs) is None

    def test_custom_window_size_limits_lookback(self):
        """Custom window size should limit lookback."""
        msgs = [
            _assistant(
                "I think the first approach involves binary search algorithms"
            ),
            _assistant(
                "The second approach requires database schema modifications"
            ),
            _assistant("repeating reasoning about trajectory and physics"),
            _assistant("repeating reasoning about camera direction"),
        ]
        # With window_size=2, only last 2 are checked — they share few tokens.
        result = check_for_thinking_loop(msgs, window_size=2)
        assert result is None

    def test_structured_reasoning_content_field(self):
        """Should also detect loops in reasoning_content field."""
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": (
                    "Actually, thinking about this more carefully: when looking "
                    "straight ahead horizontally, all trajectory points would "
                    "naturally project to nearly the same screen X coordinate"
                ),
            },
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": (
                    "Actually, thinking about this more carefully: when looking "
                    "straight ahead horizontally, all trajectory points would "
                    "naturally project to nearly the same screen Y coordinate"
                ),
            },
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": (
                    "Actually, thinking about this more carefully: when looking "
                    "straight ahead horizontally, all trajectory points would "
                    "naturally project to nearly the same screen Z coordinate"
                ),
            },
        ]
        result = check_for_thinking_loop(msgs)
        assert result is not None


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_none_reasoning_skipped(self):
        """Messages with None reasoning are skipped, breaking the chain."""
        msgs = [
            _assistant("real reasoning text here"),
            {"role": "assistant", "content": "", "reasoning": None},
            _assistant("more reasoning text here"),
        ]
        # The None reasoning message is skipped. With only 2 candidates
        # and the comparison chain broken by None, no loop is detected.
        assert check_for_thinking_loop(msgs) is None

    def test_none_reasoning_with_enough_candidates(self):
        """With enough non-None candidates, a loop can still be detected."""
        msgs = [
            _assistant("real reasoning text here"),
            {"role": "assistant", "content": "", "reasoning": None},
            _assistant("more reasoning text here"),
            _assistant("even more reasoning text here"),
        ]
        # 3 candidates, but the None breaks the comparison chain.
        # Only one valid comparison (index 2 vs index 3), which is < 2 consecutive.
        assert check_for_thinking_loop(msgs) is None

    def test_mixed_content_and_reasoning(self):
        """Messages with both content and reasoning are productive."""
        msgs = [
            _assistant("thinking A", content="answer A"),
            _assistant("thinking B", content="answer B"),
            _assistant("thinking C", content="answer C"),
        ]
        assert check_for_thinking_loop(msgs) is None

    def test_custom_window_size(self):
        """Custom window size should limit lookback."""
        msgs = [
            _assistant(
                "I think the first approach involves binary search algorithms"
            ),
            _assistant(
                "The second approach requires database schema modifications"
            ),
            _assistant("repeating reasoning about trajectory and physics"),
            _assistant("repeating reasoning about camera direction"),
        ]
        # With window_size=2, only last 2 are checked — they share few tokens.
        result = check_for_thinking_loop(msgs, window_size=2)
        assert result is None

    def test_custom_overlap_threshold(self):
        """Custom threshold should affect detection."""
        msgs = [
            _assistant(
                "Actually, thinking about this more carefully: when looking "
                "straight ahead horizontally, all trajectory points would "
                "naturally project to nearly the same screen X coordinate"
            ),
            _assistant(
                "Actually, thinking about this more carefully: when looking "
                "straight ahead horizontally, all trajectory points would "
                "naturally project to nearly the same screen Y coordinate"
            ),
            _assistant(
                "Actually, thinking about this more carefully: when looking "
                "straight ahead horizontally, all trajectory points would "
                "naturally project to nearly the same screen Z coordinate"
            ),
        ]
        # Low threshold — should detect.
        result_low = check_for_thinking_loop(msgs, overlap_threshold=0.3)
        assert result_low is not None
        # High threshold — should NOT detect.
        result_high = check_for_thinking_loop(msgs, overlap_threshold=0.95)
        assert result_high is None
