"""Tests for progressive summarization in ContextCompressor.

Progressive summarization enables context compression on auxiliary models with
small context windows (8K–32K) by splitting large serialized conversations into
chunks, summarizing each independently, and merging partial summaries iteratively.
"""

import pytest
from unittest.mock import MagicMock, patch


class TestEstimateSerializedTokens:
    """Test _estimate_serialized_tokens token estimation."""

    def test_empty_content(self):
        from agent.context_compressor import ContextCompressor
        cc = ContextCompressor.__new__(ContextCompressor)  # bypass __init__
        result = cc._estimate_serialized_tokens("")
        assert result == 0

    def test_none_content(self):
        from agent.context_compressor import ContextCompressor
        cc = ContextCompressor.__new__(ContextCompressor)
        result = cc._estimate_serialized_tokens(None)
        assert result == 0

    def test_short_text(self):
        """Token estimate should scale linearly with content length."""
        from agent.context_compressor import ContextCompressor
        cc = ContextCompressor.__new__(ContextCompressor)
        short = "a" * 100
        long_content = "a" * 400
        est_short = cc._estimate_serialized_tokens(short)
        est_long = cc._estimate_serialized_tokens(long_content)
        assert est_long == est_short * 4

    def test_returns_at_least_one(self):
        from agent.context_compressor import ContextCompressor
        cc = ContextCompressor.__new__(ContextCompressor)
        result = cc._estimate_serialized_tokens("x")
        assert result >= 1


class TestChunkTurnsForSummary:
    """Test _chunk_turns_for_summary turn splitting."""

    def test_single_turn_returns_as_is(self):
        from agent.context_compressor import ContextCompressor
        cc = ContextCompressor.__new__(ContextCompressor)
        turns = [{"role": "user", "content": "hello"}]
        chunks = cc._chunk_turns_for_summary(turns, max_chunk_tokens=1000)
        assert len(chunks) == 1
        assert chunks[0] == turns

    def test_empty_turns(self):
        from agent.context_compressor import ContextCompressor
        cc = ContextCompressor.__new__(ContextCompressor)
        chunks = cc._chunk_turns_for_summary([], max_chunk_tokens=1000)
        # Empty input → empty result or single empty chunk
        assert len(chunks) == 0 or (len(chunks) == 1 and chunks[0] == [])

    def test_fits_in_single_chunk(self):
        """Small conversation should not be split."""
        from agent.context_compressor import ContextCompressor
        cc = ContextCompressor.__new__(ContextCompressor)
        turns = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there!"},
            {"role": "user", "content": "how are you?"},
        ]
        chunks = cc._chunk_turns_for_summary(turns, max_chunk_tokens=10000)
        assert len(chunks) == 1

    def test_splits_into_multiple_chunks(self):
        """Large conversation should be split into multiple chunks."""
        from agent.context_compressor import ContextCompressor
        cc = ContextCompressor.__new__(ContextCompressor)
        # Create turns with substantial content to exceed chunk limit
        long_text = "x" * 5000
        turns = [
            {"role": "user", "content": f"message {i}: {long_text}"}
            for i in range(20)
        ]
        chunks = cc._chunk_turns_for_summary(turns, max_chunk_tokens=4096)
        assert len(chunks) > 1
        # Each chunk should be a list of turns
        for chunk in chunks:
            assert isinstance(chunk, list)

    def test_preserves_turn_order(self):
        """Chunk boundaries should preserve original turn order."""
        from agent.context_compressor import ContextCompressor
        cc = ContextCompressor.__new__(ContextCompressor)
        turns = [{"role": "user", "content": f"msg {i}"} for i in range(10)]
        chunks = cc._chunk_turns_for_summary(turns, max_chunk_tokens=50)

        # Reconstruct original order from chunks
        reconstructed = []
        for chunk in chunks:
            reconstructed.extend(chunk)
        assert reconstructed == turns


class TestProgressiveSummarizationIntegration:
    """Integration tests for _progressive_generate_summary."""

    def test_delegates_to_direct_when_content_fits(self):
        """When serialized content fits within aux window, use direct summarization."""
        from agent.context_compressor import ContextCompressor
        cc = ContextCompressor.__new__(ContextCompressor)
        cc.aux_context_length = 64000

        with patch.object(cc, '_serialize_for_summary', return_value="short content"):
            with patch.object(cc, '_generate_summary', return_value="summary body") as mock_gen:
                result = cc._progressive_generate_summary(
                    [{"role": "user", "content": "hello"}],
                    focus_topic=None,
                )

                # Should have called _generate_summary (direct path)
                mock_gen.assert_called_once()
                assert result is not None

    def test_uses_progressive_when_content_exceeds(self):
        """When content exceeds aux window, use progressive chunking."""
        from agent.context_compressor import ContextCompressor
        cc = ContextCompressor.__new__(ContextCompressor)
        # Minimal attributes needed by _summarize_chunk and _progressive_generate_summary
        cc.model = "test-model"
        cc.provider = ""
        cc.base_url = ""
        cc.api_key = ""
        cc.api_mode = ""
        cc.max_summary_tokens = 1024
        # Very small aux window so even modest content triggers progressive mode.
        # effective_limit = max(1024, int(32768*0.85)) - 2048 = 6963-2048=4915 tokens
        # "x"*5000 ≈ 1250 tokens → fits. So we need smaller:
        # effective_limit with aux=8192: max(1024, 6963)-2048 = 4915 → still fits.
        # With aux=4096: max(1024, 3481)-2048 = 1433 → "x"*5000 (1250) barely fits.
        # Use a tiny window to guarantee progressive path.
        cc.aux_context_length = 2048

        call_count = [0]
        def mock_serialize(turns):
            call_count[0] += 1
            return "x" * 5000  # ~1250 tokens, exceeds effective_limit of -308

        with patch.object(cc, '_serialize_for_summary', side_effect=mock_serialize):
            with patch.object(cc, '_chunk_turns_for_summary', return_value=[
                [{"role": "user", "content": "chunk1"}],
                [{"role": "assistant", "content": "chunk2"}],
            ]) as mock_chunk:
                def mock_summarize(content, previous_summary=None):
                    if not previous_summary:
                        return "summary1"
                    return "merged"

                with patch.object(cc, '_summarize_chunk', side_effect=mock_summarize):
                    result = cc._progressive_generate_summary(
                        [
                            {"role": "user", "content": "msg1"},
                            {"role": "assistant", "content": "msg2"},
                        ],
                        focus_topic=None,
                    )

                    # Should have chunked and summarized progressively
                    mock_chunk.assert_called_once()
                    assert result is not None


class TestChunkSummarization:
    """Test _summarize_chunk individual chunk summarization."""

    def _make_cc(self):
        from agent.context_compressor import ContextCompressor
        cc = ContextCompressor.__new__(ContextCompressor)
        # Minimal attributes needed by _summarize_chunk
        cc.model = "test-model"
        cc.provider = ""
        cc.base_url = ""
        cc.api_key = ""
        cc.api_mode = ""
        cc.max_summary_tokens = 1024
        return cc

    def test_summarizes_without_previous_summary(self):
        from agent.context_compressor import ContextCompressor
        cc = self._make_cc()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="chunk summary"))]

        with patch(
            "agent.context_compressor.call_llm",
            return_value=mock_response,
        ):
            result = cc._summarize_chunk("serialized turn content")
            assert result == "chunk summary"

    def test_summarizes_with_previous_summary(self):
        from agent.context_compressor import ContextCompressor
        cc = self._make_cc()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="updated"))]

        with patch(
            "agent.context_compressor.call_llm",
            return_value=mock_response,
        ):
            result = cc._summarize_chunk("new content", previous_summary="old summary")
            assert result == "updated"

    def test_returns_none_on_failure(self):
        from agent.context_compressor import ContextCompressor
        cc = self._make_cc()

        with patch(
            "agent.context_compressor.call_llm",
            side_effect=Exception("API error"),
        ):
            result = cc._summarize_chunk("content")
            assert result is None


class TestConstants:
    """Test progressive summarization constants."""

    def test_chunk_headroom_is_reasonable(self):
        from agent.context_compressor import _PROGRESSIVE_CHUNK_HEADROOM_TOKENS
        # 2K tokens headroom for prompt + metadata per chunk
        assert 1024 <= _PROGRESSIVE_CHUNK_HEADROOM_TOKENS <= 4096

    def test_merge_headroom_is_reasonable(self):
        from agent.context_compressor import _PROGRESSIVE_MERGE_HEADROOM_TOKENS
        # Merge headroom should be less than chunk headroom (merges are simpler)
        assert _PROGRESSIVE_MERGE_HEADROOM_TOKENS < 4096
