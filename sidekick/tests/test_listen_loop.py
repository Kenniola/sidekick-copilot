"""Tests for the live audio loop extracted into ``sidekick.engine`` (Phase 3).

The loop drives real audio hardware and Whisper in production, so these tests
substitute fake capture/recogniser components and monkeypatch the loop tuning
constants to exercise every branch deterministically — no hardware, no clock
sleeps. ``classify_and_dispatch`` is stubbed so the tests focus purely on the
loop's batching, silence-timeout, error-budget, and cleanup behaviour.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from sidekick import engine
from sidekick.session_state import SessionState


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeRecogniser:
    """Stand-in for the Whisper recogniser."""

    def __init__(self, lines_per_chunk=None, raise_exc=None):
        self._lines = list(lines_per_chunk) if lines_per_chunk is not None else []
        self.raise_exc = raise_exc
        self.closed = False
        self.calls = 0
        self.offsets: list[float] = []
        self.prompts: list[str | None] = []
        self.speakers: list[str] = []

    async def transcribe_chunk(
        self, audio, chunk_start_offset=0.0, initial_prompt=None, speaker="(audio)"
    ):
        self.calls += 1
        self.offsets.append(chunk_start_offset)
        self.prompts.append(initial_prompt)
        self.speakers.append(speaker)
        if self.raise_exc is not None:
            raise self.raise_exc
        return list(self._lines)

    def close(self):
        self.closed = True


class FakeAudioCapture:
    """Stand-in for WASAPI loopback capture.

    ``start()`` returns ``self``; ``self`` is its own async iterator. Yields the
    pre-loaded ``chunks`` then raises ``StopAsyncIteration``, unless
    ``anext_exc`` is set (raised on every ``__anext__``) or ``infinite`` is set
    (re-yields the last chunk forever).
    """

    chunk_duration = 5.0

    def __init__(self, chunks=None, anext_exc=None, infinite=False, **kwargs):
        self.chunks = list(chunks) if chunks is not None else []
        self.anext_exc = anext_exc
        self.infinite = infinite
        self.stopped = False
        self.closed = False
        self.began = False
        self._idx = 0
        # Mirror the real capture's 5d attributes so the engine can read them.
        self.speaker_label = kwargs.get("speaker_label", "(audio)")
        self.capture_mode = kwargs.get("capture_mode", "loopback")

    def begin(self):
        self.began = True

    def start(self):
        return self

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.anext_exc is not None:
            raise self.anext_exc
        if self._idx >= len(self.chunks):
            if self.infinite and self.chunks:
                return self.chunks[-1]
            raise StopAsyncIteration
        chunk = self.chunks[self._idx]
        self._idx += 1
        return chunk

    async def aclose(self):
        self.closed = True

    def stop(self):
        self.stopped = True

    def list_devices(self):
        return []


def _make_state(recogniser=None, capture=None) -> SessionState:
    s = SessionState()
    s.config = SimpleNamespace(
        sensitivity=SimpleNamespace(analyst_interval_seconds=10),
        speech=SimpleNamespace(),
    )
    s.recogniser = recogniser
    s.audio_capture = capture
    return s


# ---------------------------------------------------------------------------
# _consume_audio — normal dispatch
# ---------------------------------------------------------------------------


class TestConsumeAudioDispatch:
    @pytest.mark.asyncio
    async def test_dispatches_each_chunk_at_interval(self, monkeypatch):
        # Interval 0 → classify on every chunk; silence never fires.
        monkeypatch.setenv("SIDEKICK_CLASSIFY_INTERVAL", "0")
        monkeypatch.setattr(engine, "SILENCE_TIMEOUT_SECS", 10_000)
        dispatch = AsyncMock()
        monkeypatch.setattr(engine, "classify_and_dispatch", dispatch)

        rec = FakeRecogniser(lines_per_chunk=["L"])
        cap = FakeAudioCapture(chunks=["chunk1", "chunk2"])
        state = _make_state(rec, cap)

        await engine._consume_audio(state, lambda r: None)

        assert dispatch.await_count == 2
        # Each batch is flushed (pending cleared) — first positional arg is state.
        first_call = dispatch.await_args_list[0]
        assert first_call.args[0] is state

    @pytest.mark.asyncio
    async def test_no_dispatch_before_interval_elapses(self, monkeypatch):
        # Large interval → chunk accumulates but is not classified; loop ends
        # on StopAsyncIteration without an interval-triggered dispatch.
        monkeypatch.setenv("SIDEKICK_CLASSIFY_INTERVAL", "10000")
        monkeypatch.setattr(engine, "SILENCE_TIMEOUT_SECS", 10_000)
        dispatch = AsyncMock()
        monkeypatch.setattr(engine, "classify_and_dispatch", dispatch)

        rec = FakeRecogniser(lines_per_chunk=["L"])
        cap = FakeAudioCapture(chunks=["chunk1"])
        state = _make_state(rec, cap)

        await engine._consume_audio(state, lambda r: None)

        assert dispatch.await_count == 0

    @pytest.mark.asyncio
    async def test_cleans_up_on_normal_exit(self, monkeypatch):
        monkeypatch.setenv("SIDEKICK_CLASSIFY_INTERVAL", "0")
        monkeypatch.setattr(engine, "SILENCE_TIMEOUT_SECS", 10_000)
        monkeypatch.setattr(engine, "classify_and_dispatch", AsyncMock())

        rec = FakeRecogniser(lines_per_chunk=["L"])
        cap = FakeAudioCapture(chunks=["chunk1"])
        state = _make_state(rec, cap)

        await engine._consume_audio(state, lambda r: None)

        assert cap.stopped is True
        assert rec.closed is True


# ---------------------------------------------------------------------------
# _consume_audio — silence auto-stop
# ---------------------------------------------------------------------------


class TestConsumeAudioSilence:
    @pytest.mark.asyncio
    async def test_auto_stop_when_no_audio_arrives(self, monkeypatch):
        # __anext__ times out and the speech timer has elapsed → auto-stop.
        monkeypatch.setattr(engine, "SILENCE_TIMEOUT_SECS", 0)
        monkeypatch.setattr(engine, "classify_and_dispatch", AsyncMock())

        cap = FakeAudioCapture(anext_exc=asyncio.TimeoutError())
        state = _make_state(FakeRecogniser(), cap)

        await engine._consume_audio(state, lambda r: None)

        assert "no speech detected" in state.last_error
        assert cap.stopped

    @pytest.mark.asyncio
    async def test_auto_stop_when_audio_flows_without_speech(self, monkeypatch):
        # Chunks arrive but recogniser returns no words → speech timer elapses.
        monkeypatch.setenv("SIDEKICK_CLASSIFY_INTERVAL", "0")
        monkeypatch.setattr(engine, "SILENCE_TIMEOUT_SECS", 0)
        dispatch = AsyncMock()
        monkeypatch.setattr(engine, "classify_and_dispatch", dispatch)

        rec = FakeRecogniser(lines_per_chunk=[])  # no recognised speech
        cap = FakeAudioCapture(chunks=["c1", "c2"])
        state = _make_state(rec, cap)

        await engine._consume_audio(state, lambda r: None)

        assert "no recognised speech" in state.last_error
        # No words → nothing pending → no dispatch on the silence path.
        assert dispatch.await_count == 0

    @pytest.mark.asyncio
    async def test_flushes_pending_lines_before_silence_stop(self, monkeypatch):
        # A chunk WITH words leaves pending lines; the immediate silence check
        # (timeout 0) flushes them through classify_and_dispatch before stopping.
        monkeypatch.setenv("SIDEKICK_CLASSIFY_INTERVAL", "10000")
        monkeypatch.setattr(engine, "SILENCE_TIMEOUT_SECS", 0)
        dispatch = AsyncMock()
        monkeypatch.setattr(engine, "classify_and_dispatch", dispatch)

        rec = FakeRecogniser(lines_per_chunk=["L"])
        cap = FakeAudioCapture(chunks=["c1"])
        state = _make_state(rec, cap)

        await engine._consume_audio(state, lambda r: None)

        assert dispatch.await_count == 1
        assert "no recognised speech" in state.last_error


# ---------------------------------------------------------------------------
# _consume_audio — error budget & cancellation
# ---------------------------------------------------------------------------


class TestConsumeAudioErrors:
    @pytest.mark.asyncio
    async def test_aborts_after_max_consecutive_errors(self, monkeypatch):
        monkeypatch.setattr(engine, "SILENCE_TIMEOUT_SECS", 10_000)
        monkeypatch.setattr(engine, "MAX_CONSECUTIVE_ERRORS", 2)
        monkeypatch.setattr(engine, "classify_and_dispatch", AsyncMock())

        rec = FakeRecogniser(raise_exc=ValueError("boom"))
        cap = FakeAudioCapture(chunks=["c"], infinite=True)
        state = _make_state(rec, cap)

        await engine._consume_audio(state, lambda r: None)

        assert "Too many consecutive errors" in state.last_error
        assert cap.stopped and rec.closed

    @pytest.mark.asyncio
    async def test_cancellation_still_cleans_up(self, monkeypatch):
        monkeypatch.setattr(engine, "SILENCE_TIMEOUT_SECS", 10_000)
        monkeypatch.setattr(engine, "classify_and_dispatch", AsyncMock())

        cap = FakeAudioCapture(anext_exc=asyncio.CancelledError())
        rec = FakeRecogniser()
        state = _make_state(rec, cap)

        await engine._consume_audio(state, lambda r: None)

        # CancelledError is caught by the loop's own handler; cleanup still runs.
        assert cap.stopped and rec.closed


# ---------------------------------------------------------------------------
# run_listen_loop wiring & _initialise_capture
# ---------------------------------------------------------------------------


class TestRunListenLoop:
    @pytest.mark.asyncio
    async def test_skips_consume_when_init_fails(self, monkeypatch):
        monkeypatch.setattr(engine, "_initialise_capture", AsyncMock(return_value=False))
        consume = AsyncMock()
        monkeypatch.setattr(engine, "_consume_audio", consume)

        await engine.run_listen_loop(_make_state(), lambda r: None)

        consume.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_runs_consume_when_init_succeeds(self, monkeypatch):
        monkeypatch.setattr(engine, "_initialise_capture", AsyncMock(return_value=True))
        consume = AsyncMock()
        monkeypatch.setattr(engine, "_consume_audio", consume)
        state = _make_state()
        notify = lambda r: None  # noqa: E731

        await engine.run_listen_loop(state, notify)

        consume.assert_awaited_once_with(state, notify)


class TestInitialiseCapture:
    @pytest.mark.asyncio
    async def test_returns_false_when_speech_model_fails(self, monkeypatch):
        # create_recogniser raises → init reports the failure and returns False.
        from sidekick.transcript import speech_recogniser

        def boom(_speech_cfg):
            raise RuntimeError("no model")

        monkeypatch.setattr(speech_recogniser, "create_recogniser", boom)
        state = _make_state()

        ok = await engine._initialise_capture(state)

        assert ok is False
        assert "Failed to load speech model" in state.last_error

    @pytest.mark.asyncio
    async def test_returns_true_on_success(self, monkeypatch):
        from sidekick.transcript import audio_capture, speech_recogniser

        rec = FakeRecogniser()
        monkeypatch.setattr(speech_recogniser, "create_recogniser", lambda _cfg: rec)
        monkeypatch.setattr(audio_capture, "AudioCapture", FakeAudioCapture)
        state = _make_state()

        ok = await engine._initialise_capture(state)

        assert ok is True
        assert state.recogniser is rec
        assert isinstance(state.audio_capture, FakeAudioCapture)


# ---------------------------------------------------------------------------
# Phase 5a — sample-based audio offset propagation
# ---------------------------------------------------------------------------


class _OffsetAudioCapture(FakeAudioCapture):
    """Capture that exposes a per-chunk ``last_chunk_offset`` like the real one."""

    def __init__(self, offsets, **kw):
        super().__init__(chunks=[f"c{i}" for i in range(len(offsets))], **kw)
        self._offsets = list(offsets)
        self.last_chunk_offset = 0.0

    async def __anext__(self):
        chunk = await super().__anext__()
        self.last_chunk_offset = self._offsets[self._idx - 1]
        return chunk


class TestSampleBasedOffset:
    @pytest.mark.asyncio
    async def test_uses_capture_offset_when_available(self, monkeypatch):
        # Capture-provided offsets (e.g. 15s gaps over silence) must reach the
        # recogniser verbatim — NOT the index*chunk_duration fallback (0,5,10).
        monkeypatch.setenv("SIDEKICK_CLASSIFY_INTERVAL", "0")
        monkeypatch.setattr(engine, "SILENCE_TIMEOUT_SECS", 10_000)
        monkeypatch.setattr(engine, "classify_and_dispatch", AsyncMock())

        rec = FakeRecogniser(lines_per_chunk=["L"])
        cap = _OffsetAudioCapture(offsets=[0.0, 15.0, 30.0])
        state = _make_state(rec, cap)

        await engine._consume_audio(state, lambda r: None)

        assert rec.offsets == [0.0, 15.0, 30.0]

    @pytest.mark.asyncio
    async def test_falls_back_to_index_offset_without_capture_offset(self, monkeypatch):
        # FakeAudioCapture exposes no last_chunk_offset → index*chunk_duration.
        monkeypatch.setenv("SIDEKICK_CLASSIFY_INTERVAL", "0")
        monkeypatch.setattr(engine, "SILENCE_TIMEOUT_SECS", 10_000)
        monkeypatch.setattr(engine, "classify_and_dispatch", AsyncMock())

        rec = FakeRecogniser(lines_per_chunk=["L"])
        cap = FakeAudioCapture(chunks=["c1", "c2"])  # chunk_duration = 5.0
        state = _make_state(rec, cap)

        await engine._consume_audio(state, lambda r: None)

        assert rec.offsets == [0.0, 5.0]


# ---------------------------------------------------------------------------
# Phase 5b — derived vocabulary prior propagation
# ---------------------------------------------------------------------------


class _FakeVocab:
    def __init__(self, prompt):
        self._prompt = prompt
        self.updated: list = []

    def initial_prompt(self):
        return self._prompt

    def update(self, texts):
        self.updated.append(texts)


class TestVocabularyPrior:
    @pytest.mark.asyncio
    async def test_passes_initial_prompt_from_vocabulary(self, monkeypatch):
        monkeypatch.setenv("SIDEKICK_CLASSIFY_INTERVAL", "0")
        monkeypatch.setattr(engine, "SILENCE_TIMEOUT_SECS", 10_000)
        monkeypatch.setattr(engine, "classify_and_dispatch", AsyncMock())

        rec = FakeRecogniser(lines_per_chunk=["L"])
        cap = FakeAudioCapture(chunks=["c1"])
        state = _make_state(rec, cap)
        state.vocabulary = _FakeVocab("Glossary: Northwind, Microsoft Fabric.")

        await engine._consume_audio(state, lambda r: None)

        assert rec.prompts == ["Glossary: Northwind, Microsoft Fabric."]

    @pytest.mark.asyncio
    async def test_no_vocabulary_passes_none_prompt(self, monkeypatch):
        monkeypatch.setenv("SIDEKICK_CLASSIFY_INTERVAL", "0")
        monkeypatch.setattr(engine, "SILENCE_TIMEOUT_SECS", 10_000)
        monkeypatch.setattr(engine, "classify_and_dispatch", AsyncMock())

        rec = FakeRecogniser(lines_per_chunk=["L"])
        cap = FakeAudioCapture(chunks=["c1"])
        state = _make_state(rec, cap)
        state.vocabulary = None

        await engine._consume_audio(state, lambda r: None)

        assert rec.prompts == [None]


# ---------------------------------------------------------------------------
# Phase 5d — speaker attribution (loopback "(remote)" + microphone "(me)")
# ---------------------------------------------------------------------------


class TestSpeakerAttribution:
    @pytest.mark.asyncio
    async def test_single_capture_label_reaches_recogniser(self, monkeypatch):
        # Default loopback-only path tags every chunk with the capture's label.
        monkeypatch.setenv("SIDEKICK_CLASSIFY_INTERVAL", "0")
        monkeypatch.setattr(engine, "SILENCE_TIMEOUT_SECS", 10_000)
        monkeypatch.setattr(engine, "classify_and_dispatch", AsyncMock())

        rec = FakeRecogniser(lines_per_chunk=["L"])
        cap = FakeAudioCapture(chunks=["c1", "c2"], speaker_label="(audio)")
        state = _make_state(rec, cap)

        await engine._consume_audio(state, lambda r: None)

        assert rec.speakers == ["(audio)", "(audio)"]

    @pytest.mark.asyncio
    async def test_merge_captures_interleaves_both_labels(self):
        # Fan-in yields (label, offset, chunk) from every capture.
        remote = FakeAudioCapture(chunks=["r1"], speaker_label="(remote)")
        mine = FakeAudioCapture(chunks=["m1"], speaker_label="(me)")

        seen = []
        async for label, _offset, chunk in engine._merge_captures([remote, mine]):
            seen.append((label, chunk))

        assert ("(remote)", "r1") in seen
        assert ("(me)", "m1") in seen
        assert len(seen) == 2

    @pytest.mark.asyncio
    async def test_dual_capture_dispatches_tagged_lines(self, monkeypatch):
        # With two captures registered, both speakers reach the recogniser.
        monkeypatch.setenv("SIDEKICK_CLASSIFY_INTERVAL", "0")
        monkeypatch.setattr(engine, "SILENCE_TIMEOUT_SECS", 10_000)
        monkeypatch.setattr(engine, "classify_and_dispatch", AsyncMock())

        rec = FakeRecogniser(lines_per_chunk=["L"])
        remote = FakeAudioCapture(chunks=["r1"], speaker_label="(remote)")
        mine = FakeAudioCapture(chunks=["m1"], speaker_label="(me)")
        state = _make_state(rec, remote)
        state.audio_captures = [remote, mine]

        await engine._consume_audio(state, lambda r: None)

        assert set(rec.speakers) == {"(remote)", "(me)"}
        # Both captures are stopped on cleanup.
        assert remote.stopped and mine.stopped


class TestInitialiseCaptureMicrophone:
    @pytest.mark.asyncio
    async def test_microphone_enabled_creates_remote_and_me_captures(
        self, monkeypatch
    ):
        from sidekick.transcript import audio_capture, speech_recogniser

        created: list[FakeAudioCapture] = []

        def _factory(*a, **k):
            cap = FakeAudioCapture(**k)
            created.append(cap)
            return cap

        rec = FakeRecogniser()
        monkeypatch.setattr(speech_recogniser, "create_recogniser", lambda _c: rec)
        monkeypatch.setattr(audio_capture, "AudioCapture", _factory)

        state = _make_state()
        state.config.speech.capture_microphone = True

        ok = await engine._initialise_capture(state)

        assert ok is True
        labels = {c.speaker_label for c in created}
        assert labels == {"(remote)", "(me)"}
        assert len(state.audio_captures) == 2
        assert all(c.began for c in created)

    @pytest.mark.asyncio
    async def test_microphone_disabled_creates_single_audio_capture(
        self, monkeypatch
    ):
        from sidekick.transcript import audio_capture, speech_recogniser

        created: list[FakeAudioCapture] = []

        def _factory(*a, **k):
            cap = FakeAudioCapture(**k)
            created.append(cap)
            return cap

        rec = FakeRecogniser()
        monkeypatch.setattr(speech_recogniser, "create_recogniser", lambda _c: rec)
        monkeypatch.setattr(audio_capture, "AudioCapture", _factory)

        state = _make_state()
        state.config.speech.capture_microphone = False

        ok = await engine._initialise_capture(state)

        assert ok is True
        assert len(created) == 1
        assert created[0].speaker_label == "(audio)"
        assert len(state.audio_captures) == 1
