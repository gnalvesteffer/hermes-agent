"""Tests for per-turn rolling summary (ContextCompressor._apply_rolling_summary)."""

import pytest
from contextlib import contextmanager
from unittest.mock import patch, MagicMock

from agent.context_compressor import ContextCompressor


@pytest.fixture()
def compressor():
    """Create a ContextCompressor with mocked dependencies."""
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        c = ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=2,
            protect_last_n=2,
            quiet_mode=True,
        )
        return c


@pytest.fixture()
def enabled_compressor():
    """Create a ContextCompressor with rolling summary enabled."""
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        c = ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=2,
            protect_last_n=2,
            quiet_mode=True,
            rolling_summary_enabled=True,
            rolling_summary_recent_n=3,
        )
        return c


def _make_messages(n_turns):
    """Create a message list with n user/assistant pairs plus system prompt."""
    msgs = [{"role": "system", "content": "You are helpful."}]
    for i in range(n_turns):
        msgs.append({"role": "user", "content": f"User turn {i}"})
        msgs.append({"role": "assistant", "content": f"Assistant turn {i}"})
    return msgs


@contextmanager
def _patch_tokens_above_threshold(compressor, tokens=70000):
    """Patch estimate_messages_tokens_rough to return a value above the threshold.

    The default intra_turn_context_threshold is 60% of context_length (60k for
    a 100k model). Tests need this patched so rolling summary actually fires.
    """
    with patch("agent.context_compressor.estimate_messages_tokens_rough", return_value=tokens):
        yield


class TestRollingSummaryDisabled:
    """When rolling_summary_enabled is False, _apply_rolling_summary is a no-op."""

    def test_returns_messages_unchanged_when_disabled(self, compressor):
        messages = _make_messages(5)
        result = compressor._apply_rolling_summary(messages)
        assert result == messages

    def test_returns_messages_unchanged_when_too_few(self, enabled_compressor):
        """With fewer than 3 messages, returns unchanged even when enabled."""
        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        result = enabled_compressor._apply_rolling_summary(msgs)
        assert result == msgs


class TestRollingSummarySplitting:
    """Verify correct splitting of old vs recent messages."""

    def test_recent_n_defaults_to_5_pairs(self, compressor):
        compressor.rolling_summary_enabled = True
        # 12 turns = 24 messages + system = 25 total; recent_n defaults to 5 in fixture
        msgs = _make_messages(12)
        with _patch_tokens_above_threshold(compressor):
            with patch.object(compressor, "_serialize_for_summary", return_value="old"):
                with patch("agent.context_compressor.call_llm") as mock_call:
                    mock_resp = MagicMock()
                    mock_resp.choices = [MagicMock()]
                    mock_resp.choices[0].message.content = "summary text"
                    mock_call.return_value = mock_resp

                    result = compressor._apply_rolling_summary(msgs)

                    # recent_n=5 → protect last 5*2=10 messages
                    assert len(result) == 1 + 1 + 10  # system + summary + recent
                    # Recent messages should be the last 10 (turns 7-11)
                    assert result[-1]["content"] == "Assistant turn 11"

    def test_recent_n_custom(self):
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(
                model="test/model", rolling_summary_enabled=True,
                rolling_summary_recent_n=2, quiet_mode=True,
            )
        msgs = _make_messages(6)  # 13 messages total
        with _patch_tokens_above_threshold(c):
            with patch.object(c, "_serialize_for_summary", return_value="old"):
                with patch("agent.context_compressor.call_llm") as mock_call:
                    mock_resp = MagicMock()
                    mock_resp.choices = [MagicMock()]
                    mock_resp.choices[0].message.content = "summary"
                    mock_call.return_value = mock_resp

                    result = c._apply_rolling_summary(msgs)

                    # Should protect last 2*2=4 messages
                    assert len(result) == 1 + 1 + 4


class TestRollingSummaryLLMCall:
    """Verify the LLM call is made correctly."""

    def test_calls_llm_with_correct_params(self, enabled_compressor):
        msgs = _make_messages(6)
        with _patch_tokens_above_threshold(enabled_compressor):
            with patch.object(enabled_compressor, "_serialize_for_summary", return_value="old turns"):
                with patch("agent.context_compressor.call_llm") as mock_call:
                    mock_resp = MagicMock()
                    mock_resp.choices = [MagicMock()]
                    mock_resp.choices[0].message.content = "concise summary"
                    mock_call.return_value = mock_resp

                    enabled_compressor._apply_rolling_summary(msgs)

                    assert mock_call.called
                    kwargs = mock_call.call_args[1]
                    assert kwargs["task"] == "compression"
                    # max_tokens is now derived from rolling_summary_max_tokens (default 250)
                    assert kwargs["max_tokens"] == max(150, int(250 * 1.3))
                    # Check prompt contains old text
                    prompt = kwargs["messages"][0]["content"]
                    assert "old turns" in prompt

    def test_uses_custom_model_when_set(self):
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(
                model="big/model", rolling_summary_enabled=True,
                rolling_summary_model_override="small/fast-model", quiet_mode=True,
            )
        # Need enough turns so old_messages isn't empty after filtering system + recent
        msgs = _make_messages(6)  # 13 messages; with recent_n=5 → split_idx=3 → 2 old turns
        with _patch_tokens_above_threshold(c):
            with patch.object(c, "_serialize_for_summary", return_value="old"):
                with patch("agent.context_compressor.call_llm") as mock_call:
                    mock_resp = MagicMock()
                    mock_resp.choices = [MagicMock()]
                    mock_resp.choices[0].message.content = "summary"
                    mock_call.return_value = mock_resp

                    c._apply_rolling_summary(msgs)

                    assert mock_call.called
                    kwargs = mock_call.call_args[1]
                    assert kwargs["model"] == "small/fast-model"


class TestRollingSummaryFallback:
    """Verify graceful fallback when LLM call fails."""

    def test_returns_original_on_llm_failure(self, enabled_compressor):
        msgs = _make_messages(5)
        with _patch_tokens_above_threshold(enabled_compressor):
            with patch.object(enabled_compressor, "_serialize_for_summary", return_value="old"):
                with patch("agent.context_compressor.call_llm") as mock_call:
                    mock_call.side_effect = Exception("LLM unavailable")

                    result = enabled_compressor._apply_rolling_summary(msgs)
                    assert result == msgs  # unchanged

    def test_returns_original_on_non_string_response(self, enabled_compressor):
        msgs = _make_messages(5)
        with _patch_tokens_above_threshold(enabled_compressor):
            with patch.object(enabled_compressor, "_serialize_for_summary", return_value="old"):
                with patch("agent.context_compressor.call_llm") as mock_call:
                    mock_resp = MagicMock()
                    mock_resp.choices = [MagicMock()]
                    mock_resp.choices[0].message.content = None  # non-string
                    mock_call.return_value = mock_resp

                    result = enabled_compressor._apply_rolling_summary(msgs)
                    assert len(result) == 1 + 1 + (len(msgs) - 2)  # system + summary + recent


class TestRollingSummaryIncremental:
    """Verify the rolling summary is updated incrementally across turns."""

    def test_second_turn_merges_into_existing_summary(self, enabled_compressor):
        msgs = _make_messages(5)
        with _patch_tokens_above_threshold(enabled_compressor):
            with patch.object(enabled_compressor, "_serialize_for_summary", return_value="old turns"):
                with patch("agent.context_compressor.call_llm") as mock_call:
                    # First call — no existing summary
                    mock_resp1 = MagicMock()
                    mock_resp1.choices = [MagicMock()]
                    mock_resp1.choices[0].message.content = "first summary"
                    mock_call.return_value = mock_resp1

                    enabled_compressor._apply_rolling_summary(msgs)
                    assert enabled_compressor._rolling_summary == "first summary"

                    # Second call — existing summary present
                    mock_resp2 = MagicMock()
                    mock_resp2.choices = [MagicMock()]
                    mock_resp2.choices[0].message.content = "updated summary with first + second"
                    mock_call.return_value = mock_resp2

                    enabled_compressor._apply_rolling_summary(msgs)

                    # Verify the prompt included the existing summary
                    call_args = mock_call.call_args_list[-1]
                    prompt = call_args[1]["messages"][0]["content"]
                    assert "EXISTING ROLLING SUMMARY" in prompt
                    assert "first summary" in prompt


class TestRollingSummaryOutput:
    """Verify the output message list structure."""

    def test_output_has_system_plus_summary_plus_recent(self, enabled_compressor):
        msgs = _make_messages(5)  # system + 10 turns = 11 messages
        with _patch_tokens_above_threshold(enabled_compressor):
            with patch.object(enabled_compressor, "_serialize_for_summary", return_value="old"):
                with patch("agent.context_compressor.call_llm") as mock_call:
                    mock_resp = MagicMock()
                    mock_resp.choices = [MagicMock()]
                    mock_resp.choices[0].message.content = "checkpoint summary"
                    mock_call.return_value = mock_resp

                    result = enabled_compressor._apply_rolling_summary(msgs)

                    # First message is system prompt (preserved from old messages)
                    assert result[0]["role"] == "system"
                    # Second message is the rolling summary user message
                    assert result[1]["role"] == "user"
                    assert "[CONTEXT CHECKPOINT" in result[1]["content"]
                    assert "checkpoint summary" in result[1]["content"]
                    # Remaining are recent messages (unchanged)
                    for msg in result[2:]:
                        assert msg["role"] in ("user", "assistant")


class TestRollingSummaryReset:
    """Verify _rolling_summary is cleared on session reset."""

    def test_reset_clears_rolling_summary(self, enabled_compressor):
        enabled_compressor._rolling_summary = "some summary"
        enabled_compressor.on_session_reset()
        assert enabled_compressor._rolling_summary is None
