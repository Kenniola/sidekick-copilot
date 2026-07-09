"""Regression tests for the Whisper-only speech recogniser (v0.3.0).

These tests do NOT exercise the real Whisper model — they stub it out so
the suite runs in milliseconds without downloading model weights.

Covered:
  * ``_format_ts`` — formatting + negative-value clamp.
  * ``create_recogniser`` — Whisper-only factory, warning on legacy ``azure``
    backend, config plumbing (``model`` / ``compute_type``).
  * ``transcribe_chunk`` — ``chunk_start_offset`` is added to every segment
    timestamp (the bug fixed in v0.3.0).
  * Hallucination guards — high ``no_speech_prob`` segments are dropped;
    repeated identical text is dropped after the third occurrence.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import pytest

from sidekick.transcript import speech_recogniser as sr


# ---------------------------------------------------------------------------
# _format_ts
# ---------------------------------------------------------------------------


class TestFormatTs:
    def test_zero(self):
        assert sr._format_ts(0.0) == "0:00:00.000"

    def test_sub_second_precision(self):
        assert sr._format_ts(0.123) == "0:00:00.123"

    def test_minutes(self):
        assert sr._format_ts(90.5) == "0:01:30.500"

    def test_hours(self):
        assert sr._format_ts(3661.25) == "1:01:01.250"

    def test_negative_clamped_to_zero(self):
        assert sr._format_ts(-1.5) == "0:00:00.000"

    def test_large_offset(self):
        # 72-min meeting style offset
        assert sr._format_ts(4320.0).startswith("1:12:00")


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


@dataclass
class _Seg:
    start: float
    end: float
    text: str
    no_speech_prob: float = 0.0


class _FakeWhisperModel:
    """Stand-in for faster_whisper.WhisperModel."""

    def __init__(self, segments: Iterable[_Seg]):
        self._segments = list(segments)
        self.calls: list[dict] = []

    def transcribe(self, audio, **kwargs):
        self.calls.append({"audio_len": len(audio), **kwargs})
        return iter(self._segments), object()


def _install_fake_whisper(monkeypatch, segments):
    """Patch the WhisperRecogniser constructor to use the fake model."""
    fake = _FakeWhisperModel(segments)

    def _fake_init(self, model_size=None, compute_type=None, device=None, **kwargs):
        self.model = fake
        self.model_size = model_size or "small.en"
        self.compute_type = compute_type or "int8"
        self.device = device or "cpu"
        self._last_text = {}
        self._repeat_count = {}
        self._prev_tail = {}
        self._vad_min_silence_ms = kwargs.get("vad_min_silence_ms", 500)
        self._no_speech_threshold = kwargs.get("no_speech_threshold", 0.6)
        self._log_prob_threshold = kwargs.get("log_prob_threshold", -1.0)
        self._compression_ratio_threshold = kwargs.get(
            "compression_ratio_threshold", 2.4
        )
        self._echo_suppression = kwargs.get("echo_suppression", True)
        self._recent_lines = deque(maxlen=24)

    monkeypatch.setattr(sr.WhisperRecogniser, "__init__", _fake_init)
    return fake


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestCreateRecogniser:
    def test_returns_whisper_for_none_config(self, monkeypatch):
        _install_fake_whisper(monkeypatch, [])
        rec = sr.create_recogniser(None)
        assert isinstance(rec, sr.WhisperRecogniser)
        assert rec.model_size == "small.en"
        assert rec.compute_type == "int8"

    def test_returns_whisper_for_whisper_backend(self, monkeypatch):
        _install_fake_whisper(monkeypatch, [])

        @dataclass
        class Cfg:
            backend: str = "whisper"
            model: str = "medium.en"
            compute_type: str = "float16"

        rec = sr.create_recogniser(Cfg())
        assert rec.model_size == "medium.en"
        assert rec.compute_type == "float16"

    def test_legacy_azure_backend_logs_warning_and_falls_back(
        self, monkeypatch, caplog
    ):
        _install_fake_whisper(monkeypatch, [])

        @dataclass
        class Cfg:
            backend: str = "azure"
            model: str = "small.en"
            compute_type: str = "int8"

        with caplog.at_level(logging.WARNING, logger=sr.logger.name):
            rec = sr.create_recogniser(Cfg())

        assert isinstance(rec, sr.WhisperRecogniser)
        assert any(
            "no longer supported" in r.message and "azure" in r.message
            for r in caplog.records
        )


# ---------------------------------------------------------------------------
# Device / compute resolution
# ---------------------------------------------------------------------------


class TestResolveDeviceAndCompute:
    def test_auto_uses_cuda_when_available(self, monkeypatch):
        monkeypatch.delenv("SIDEKICK_WHISPER_DEVICE", raising=False)
        monkeypatch.delenv("SIDEKICK_WHISPER_COMPUTE", raising=False)
        monkeypatch.setattr(sr, "_cuda_available", lambda: True)
        device, compute = sr._resolve_device_and_compute(None, None)
        assert device == "cuda"
        assert compute == "float16"

    def test_auto_falls_back_to_cpu_without_gpu(self, monkeypatch):
        monkeypatch.delenv("SIDEKICK_WHISPER_DEVICE", raising=False)
        monkeypatch.delenv("SIDEKICK_WHISPER_COMPUTE", raising=False)
        monkeypatch.setattr(sr, "_cuda_available", lambda: False)
        device, compute = sr._resolve_device_and_compute(None, None)
        assert device == "cpu"
        assert compute == "int8"

    def test_cuda_request_without_gpu_warns_and_uses_cpu(self, monkeypatch, caplog):
        monkeypatch.setattr(sr, "_cuda_available", lambda: False)
        with caplog.at_level(logging.WARNING, logger=sr.logger.name):
            device, compute = sr._resolve_device_and_compute("cuda", None)
        assert device == "cpu"
        assert compute == "int8"
        assert any("no CUDA GPU" in r.message for r in caplog.records)

    def test_explicit_compute_is_honoured(self, monkeypatch):
        monkeypatch.setattr(sr, "_cuda_available", lambda: True)
        device, compute = sr._resolve_device_and_compute("auto", "int8_float16")
        assert device == "cuda"
        assert compute == "int8_float16"

    def test_unknown_device_falls_back_to_cpu(self, monkeypatch):
        monkeypatch.delenv("SIDEKICK_WHISPER_COMPUTE", raising=False)
        device, compute = sr._resolve_device_and_compute("npu", None)
        assert device == "cpu"
        assert compute == "int8"

    def test_env_device_override(self, monkeypatch):
        monkeypatch.setenv("SIDEKICK_WHISPER_DEVICE", "cpu")
        monkeypatch.delenv("SIDEKICK_WHISPER_COMPUTE", raising=False)
        monkeypatch.setattr(sr, "_cuda_available", lambda: True)
        device, compute = sr._resolve_device_and_compute(None, None)
        assert device == "cpu"
        assert compute == "int8"


# ---------------------------------------------------------------------------
# transcribe_chunk
# ---------------------------------------------------------------------------


class TestTranscribeChunk:
    def test_chunk_offset_zero_session_relative_matches_chunk_relative(
        self, monkeypatch
    ):
        _install_fake_whisper(
            monkeypatch,
            [_Seg(0.0, 2.5, "Hello world."), _Seg(2.5, 4.8, "How are you?")],
        )
        rec = sr.WhisperRecogniser()
        audio = np.zeros(16_000, dtype=np.float32)

        lines = asyncio.run(rec.transcribe_chunk(audio, chunk_start_offset=0.0))

        assert [(ln.start, ln.end, ln.text) for ln in lines] == [
            ("0:00:00.000", "0:00:02.500", "Hello world."),
            ("0:00:02.500", "0:00:04.800", "How are you?"),
        ]
        assert all(ln.speaker == "(audio)" for ln in lines)

    def test_speaker_tag_applied_to_all_lines(self, monkeypatch):
        """5d: the speaker arg overrides the default "(audio)" tag."""
        _install_fake_whisper(
            monkeypatch,
            [_Seg(0.0, 1.0, "Over here."), _Seg(1.0, 2.0, "And here.")],
        )
        rec = sr.WhisperRecogniser()
        audio = np.zeros(16_000, dtype=np.float32)

        lines = asyncio.run(rec.transcribe_chunk(audio, speaker="(me)"))

        assert lines  # sanity
        assert all(ln.speaker == "(me)" for ln in lines)
        """The v0.3.0 fix: chunk_start_offset must be added to every timestamp."""
        _install_fake_whisper(
            monkeypatch,
            [_Seg(0.0, 1.0, "First"), _Seg(2.0, 4.5, "Second")],
        )
        rec = sr.WhisperRecogniser()
        audio = np.zeros(16_000, dtype=np.float32)

        # 72-minute meeting — chunk starting at 1h 12m
        offset = 4320.0
        lines = asyncio.run(
            rec.transcribe_chunk(audio, chunk_start_offset=offset)
        )

        assert lines[0].start == "1:12:00.000"
        assert lines[0].end == "1:12:01.000"
        assert lines[1].start == "1:12:02.000"
        assert lines[1].end == "1:12:04.500"

    def test_high_no_speech_prob_dropped(self, monkeypatch):
        _install_fake_whisper(
            monkeypatch,
            [
                _Seg(0.0, 1.0, "Real speech."),
                _Seg(1.0, 2.0, "Hallucinated.", no_speech_prob=0.95),
            ],
        )
        rec = sr.WhisperRecogniser()
        lines = asyncio.run(
            rec.transcribe_chunk(np.zeros(16_000, dtype=np.float32))
        )
        assert [ln.text for ln in lines] == ["Real speech."]

    def test_repeated_text_dropped_after_threshold(self, monkeypatch):
        _install_fake_whisper(
            monkeypatch,
            [_Seg(i, i + 1, "Thank you.") for i in range(5)],
        )
        rec = sr.WhisperRecogniser()
        lines = asyncio.run(
            rec.transcribe_chunk(np.zeros(16_000, dtype=np.float32))
        )
        # First three identical segments pass through; the 4th and 5th are
        # dropped (repeat_count reaches the >=3 threshold on the 4th).
        assert len(lines) == 3
        assert all(ln.text == "Thank you." for ln in lines)

    def test_repetition_guard_is_per_speaker(self, monkeypatch):
        """C2.3: one speaker's repeats must not suppress another speaker's
        identical short utterance.

        Regression for the shared ``_last_text``/``_repeat_count`` state: with a
        single shared counter, three "Yes." from ``(me)`` would push the count
        to the drop threshold, so the *remote* speaker's first "Yes." was
        silently dropped. Keyed by speaker, each side is tracked independently.
        """
        fake = _install_fake_whisper(monkeypatch, [])
        rec = sr.WhisperRecogniser()

        # (me) says "Yes." enough times to trip its own guard.
        fake._segments = [_Seg(float(i), i + 1.0, "Yes.") for i in range(4)]
        me_lines = asyncio.run(
            rec.transcribe_chunk(
                np.zeros(16_000, dtype=np.float32), speaker="(me)"
            )
        )
        assert len(me_lines) == 3  # 4th dropped for (me)

        # (remote) says "Yes." once — must NOT be suppressed by (me)'s state.
        fake._segments = [_Seg(0.0, 1.0, "Yes.")]
        remote_lines = asyncio.run(
            rec.transcribe_chunk(
                np.zeros(16_000, dtype=np.float32), speaker="(remote)"
            )
        )
        assert [ln.text for ln in remote_lines] == ["Yes."]
        assert all(ln.speaker == "(remote)" for ln in remote_lines)

    def test_empty_text_segments_skipped(self, monkeypatch):
        _install_fake_whisper(
            monkeypatch,
            [_Seg(0.0, 1.0, "   "), _Seg(1.0, 2.0, "Real.")],
        )
        rec = sr.WhisperRecogniser()
        lines = asyncio.run(
            rec.transcribe_chunk(np.zeros(16_000, dtype=np.float32))
        )
        assert [ln.text for ln in lines] == ["Real."]

    def test_does_not_block_event_loop(self, monkeypatch):
        """Regression: CPU-bound Whisper inference must run off the event loop.

        Running ``model.transcribe`` synchronously on the loop starved
        concurrent research tasks (their wall-clock ``wait_for`` timeouts kept
        ticking and expired) and made the ``status`` tool unresponsive. The
        inference now runs via ``asyncio.to_thread``; this test asserts a
        concurrent coroutine keeps making progress while a (blocking) transcribe
        is in flight.
        """
        import time

        fake = _install_fake_whisper(monkeypatch, [_Seg(0.0, 1.0, "Hello.")])

        # Make the fake model's transcribe genuinely block its thread.
        def _blocking_transcribe(audio, **kwargs):
            time.sleep(0.3)
            return iter([_Seg(0.0, 1.0, "Hello.")]), object()

        fake.transcribe = _blocking_transcribe
        rec = sr.WhisperRecogniser()
        audio = np.zeros(16_000, dtype=np.float32)

        async def _run() -> int:
            ticks = 0

            async def _ticker() -> None:
                nonlocal ticks
                while True:
                    await asyncio.sleep(0.01)
                    ticks += 1

            ticker = asyncio.create_task(_ticker())
            await rec.transcribe_chunk(audio)
            ticker.cancel()
            return ticks

        # If transcription blocked the loop, the ticker could not advance and
        # ``ticks`` would be ~0. Off-loaded, it advances many times during 0.3s.
        ticks = asyncio.run(_run())
        assert ticks >= 5


# ---------------------------------------------------------------------------
# Phase 5e — cross-chunk coherence (previous-text conditioning)
# ---------------------------------------------------------------------------


class TestCoherencePromptHelpers:
    def test_combine_prompt_joins_both(self):
        assert sr._combine_prompt("vocab", "tail") == "vocab tail"

    def test_combine_prompt_skips_empty(self):
        assert sr._combine_prompt(None, "tail") == "tail"
        assert sr._combine_prompt("vocab", "") == "vocab"

    def test_combine_prompt_none_when_both_empty(self):
        assert sr._combine_prompt(None, "") is None

    def test_tail_text_keeps_last_words(self):
        text = " ".join(str(i) for i in range(50))
        tail = sr._tail_text(text, max_words=3)
        assert tail == "47 48 49"


class TestCrossChunkCoherence:
    def test_prev_tail_feeds_next_chunk_prompt(self, monkeypatch):
        fake = _install_fake_whisper(
            monkeypatch,
            [_Seg(0.0, 1.0, "alpha"), _Seg(1.0, 2.0, "omega")],
        )
        rec = sr.WhisperRecogniser()
        audio = np.zeros(16_000, dtype=np.float32)

        asyncio.run(rec.transcribe_chunk(audio))
        asyncio.run(rec.transcribe_chunk(audio))

        # First chunk had no prior context; the second is conditioned on the
        # first chunk's trailing text.
        assert fake.calls[0]["initial_prompt"] is None
        assert "omega" in fake.calls[1]["initial_prompt"]
        assert "alpha" in fake.calls[1]["initial_prompt"]

    def test_tail_is_per_speaker(self, monkeypatch):
        fake = _install_fake_whisper(
            monkeypatch,
            [_Seg(0.0, 1.0, "alpha"), _Seg(1.0, 2.0, "omega")],
        )
        rec = sr.WhisperRecogniser()
        audio = np.zeros(16_000, dtype=np.float32)

        asyncio.run(rec.transcribe_chunk(audio, speaker="(me)"))
        asyncio.run(rec.transcribe_chunk(audio, speaker="(remote)"))

        # The remote speaker's first chunk must not inherit the local speaker's
        # tail — its context is independent.
        assert fake.calls[1]["initial_prompt"] is None

    def test_vocab_and_tail_combined_on_second_chunk(self, monkeypatch):
        fake = _install_fake_whisper(
            monkeypatch,
            [_Seg(0.0, 1.0, "alpha"), _Seg(1.0, 2.0, "omega")],
        )
        rec = sr.WhisperRecogniser()
        audio = np.zeros(16_000, dtype=np.float32)

        asyncio.run(
            rec.transcribe_chunk(audio, initial_prompt="Glossary: Northwind.")
        )
        asyncio.run(
            rec.transcribe_chunk(audio, initial_prompt="Glossary: Northwind.")
        )

        prompt = fake.calls[1]["initial_prompt"]
        assert "Northwind" in prompt  # vocabulary prior (5b)
        assert "omega" in prompt   # previous-chunk tail (5e)

    def test_silent_chunk_does_not_erase_tail(self, monkeypatch):
        # A chunk that yields no lines must not wipe the running context.
        fake = _install_fake_whisper(
            monkeypatch, [_Seg(0.0, 1.0, "alpha omega")]
        )
        rec = sr.WhisperRecogniser()
        audio = np.zeros(16_000, dtype=np.float32)

        asyncio.run(rec.transcribe_chunk(audio))  # sets tail
        fake._segments = []  # next chunk transcribes to nothing
        asyncio.run(rec.transcribe_chunk(audio))
        fake._segments = [_Seg(0.0, 1.0, "next")]
        asyncio.run(rec.transcribe_chunk(audio))

        # The third chunk's prompt still carries the first chunk's tail.
        assert "omega" in fake.calls[2]["initial_prompt"]


# ---------------------------------------------------------------------------
# Protocol contract
# ---------------------------------------------------------------------------


class TestSpeechRecogniserProtocol:
    def test_whisper_recogniser_satisfies_protocol(self, monkeypatch):
        _install_fake_whisper(monkeypatch, [])
        rec = sr.WhisperRecogniser()
        # Duck-typed protocol — runtime check via attribute presence.
        assert hasattr(rec, "transcribe_chunk")
        assert hasattr(rec, "close")
        rec.close()
        assert rec.model is None

class TestDecodeThresholds:
    """Phase 2 / C2: tuned VAD + decode thresholds reach faster-whisper."""

    def test_default_thresholds_passed_to_transcribe(self, monkeypatch):
        fake = _install_fake_whisper(monkeypatch, [_Seg(0.0, 1.0, "hello")])
        rec = sr.WhisperRecogniser()
        asyncio.run(rec.transcribe_chunk(np.zeros(16_000, dtype=np.float32)))
        call = fake.calls[0]
        assert call["vad_filter"] is True
        assert call["vad_parameters"] == {"min_silence_duration_ms": 500}
        assert call["no_speech_threshold"] == 0.6
        assert call["log_prob_threshold"] == -1.0
        assert call["compression_ratio_threshold"] == 2.4

    def test_config_values_flow_through_factory(self, monkeypatch):
        _install_fake_whisper(monkeypatch, [])
        cfg = SimpleNamespace(
            backend="whisper",
            model="small.en",
            compute_type="int8",
            device="cpu",
            vad_min_silence_ms=300,
            no_speech_threshold=0.5,
            log_prob_threshold=-0.8,
            compression_ratio_threshold=2.0,
            echo_suppression=False,
        )
        rec = sr.create_recogniser(cfg)
        assert rec._vad_min_silence_ms == 300
        assert rec._no_speech_threshold == 0.5
        assert rec._log_prob_threshold == -0.8
        assert rec._compression_ratio_threshold == 2.0
        assert rec._echo_suppression is False


class TestEchoSuppression:
    """Phase 2 / C3 Tier 1: cross-speaker speaker-bleed is de-duplicated."""

    def _audio(self):
        return np.zeros(16_000, dtype=np.float32)

    def test_cross_speaker_duplicate_dropped(self, monkeypatch):
        fake = _install_fake_whisper(monkeypatch, [])
        rec = sr.WhisperRecogniser()
        # (remote) speaks at offset 10.0
        fake._segments = [_Seg(0.0, 1.0, "the egress cost is high")]
        asyncio.run(
            rec.transcribe_chunk(self._audio(), chunk_start_offset=10.0, speaker="(remote)")
        )
        # (me) mic picks up the same speaker output ~0.5s later → echo, dropped
        fake._segments = [_Seg(0.5, 1.5, "the egress cost is high")]
        me_lines = asyncio.run(
            rec.transcribe_chunk(self._audio(), chunk_start_offset=10.0, speaker="(me)")
        )
        assert me_lines == []

    def test_outside_window_kept(self, monkeypatch):
        fake = _install_fake_whisper(monkeypatch, [])
        rec = sr.WhisperRecogniser()
        fake._segments = [_Seg(0.0, 1.0, "same words here")]
        asyncio.run(
            rec.transcribe_chunk(self._audio(), chunk_start_offset=10.0, speaker="(remote)")
        )
        # 3s later — beyond the 2s echo window → not an echo
        fake._segments = [_Seg(0.0, 1.0, "same words here")]
        me_lines = asyncio.run(
            rec.transcribe_chunk(self._audio(), chunk_start_offset=13.0, speaker="(me)")
        )
        assert [ln.text for ln in me_lines] == ["same words here"]

    def test_low_similarity_kept(self, monkeypatch):
        fake = _install_fake_whisper(monkeypatch, [])
        rec = sr.WhisperRecogniser()
        fake._segments = [_Seg(0.0, 1.0, "the egress cost is high")]
        asyncio.run(
            rec.transcribe_chunk(self._audio(), chunk_start_offset=10.0, speaker="(remote)")
        )
        fake._segments = [_Seg(0.2, 1.2, "what about capacity sizing")]
        me_lines = asyncio.run(
            rec.transcribe_chunk(self._audio(), chunk_start_offset=10.0, speaker="(me)")
        )
        assert [ln.text for ln in me_lines] == ["what about capacity sizing"]

    def test_same_speaker_not_treated_as_echo(self, monkeypatch):
        fake = _install_fake_whisper(monkeypatch, [])
        rec = sr.WhisperRecogniser()
        fake._segments = [_Seg(0.0, 1.0, "repeat me")]
        asyncio.run(
            rec.transcribe_chunk(self._audio(), chunk_start_offset=10.0, speaker="(remote)")
        )
        # Same speaker repeats within the window — echo guard must NOT drop it
        # (the repetition guard allows up to 3 before dropping).
        fake._segments = [_Seg(0.3, 1.3, "repeat me")]
        again = asyncio.run(
            rec.transcribe_chunk(self._audio(), chunk_start_offset=10.0, speaker="(remote)")
        )
        assert [ln.text for ln in again] == ["repeat me"]

    def test_disabled_keeps_cross_speaker_duplicate(self, monkeypatch):
        fake = _install_fake_whisper(monkeypatch, [])
        rec = sr.WhisperRecogniser()
        rec._echo_suppression = False
        fake._segments = [_Seg(0.0, 1.0, "the egress cost is high")]
        asyncio.run(
            rec.transcribe_chunk(self._audio(), chunk_start_offset=10.0, speaker="(remote)")
        )
        fake._segments = [_Seg(0.5, 1.5, "the egress cost is high")]
        me_lines = asyncio.run(
            rec.transcribe_chunk(self._audio(), chunk_start_offset=10.0, speaker="(me)")
        )
        assert [ln.text for ln in me_lines] == ["the egress cost is high"]

    def test_short_utterance_not_echo_suppressed(self, monkeypatch):
        # Both sides genuinely saying "Yes." must survive — short, common words
        # are below the echo min-length and are never treated as echoes.
        fake = _install_fake_whisper(monkeypatch, [])
        rec = sr.WhisperRecogniser()
        fake._segments = [_Seg(0.0, 1.0, "Yes.")]
        asyncio.run(
            rec.transcribe_chunk(self._audio(), chunk_start_offset=10.0, speaker="(remote)")
        )
        fake._segments = [_Seg(0.3, 1.3, "Yes.")]
        me_lines = asyncio.run(
            rec.transcribe_chunk(self._audio(), chunk_start_offset=10.0, speaker="(me)")
        )
        assert [ln.text for ln in me_lines] == ["Yes."]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
