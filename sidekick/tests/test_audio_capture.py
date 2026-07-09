"""Tests for the audio-capture queue + offset behaviour (Phase 5a).

The real ``AudioCapture.start`` opens WASAPI loopback hardware via
``pyaudiowpatch`` and runs a background reader thread, so these tests exercise
the two pieces that carry the 5a fix without touching hardware:

  * ``_enqueue_chunk`` / ``stop`` — the bounded **drop-to-latest** queue policy
    that keeps the live transcript current under sustained overload.
  * ``_drain_queue`` — yields offset-tagged chunks and records each chunk's
    ``last_chunk_offset`` (the sample-based audio position that replaces the
    old wall-clock timestamp, fixing the "56 min for a 32 min meeting" drift).
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from sidekick.transcript.audio_capture import AudioCapture


def _chunk(n: int = 8) -> np.ndarray:
    return np.zeros(n, dtype=np.float32)


# ---------------------------------------------------------------------------
# Bounded drop-to-latest queue
# ---------------------------------------------------------------------------


class TestDropToLatestQueue:
    @pytest.mark.asyncio
    async def test_enqueue_within_capacity_keeps_all(self):
        cap = AudioCapture(max_queue_chunks=3)
        for i in range(3):
            cap._enqueue_chunk((float(i), _chunk()))
        assert cap._queue.qsize() == 3
        assert cap.dropped_chunks == 0

    @pytest.mark.asyncio
    async def test_overflow_drops_oldest_and_keeps_newest(self):
        cap = AudioCapture(max_queue_chunks=2)
        cap._enqueue_chunk((0.0, _chunk()))
        cap._enqueue_chunk((5.0, _chunk()))
        cap._enqueue_chunk((10.0, _chunk()))  # full → drop oldest (0.0)

        assert cap._queue.qsize() == 2
        assert cap.dropped_chunks == 1
        first = cap._queue.get_nowait()
        second = cap._queue.get_nowait()
        assert first[0] == 5.0
        assert second[0] == 10.0

    @pytest.mark.asyncio
    async def test_maxsize_floor_is_one(self):
        # max_queue_chunks <= 0 must not create an unbounded/invalid queue.
        cap = AudioCapture(max_queue_chunks=0)
        assert cap._queue.maxsize == 1

    @pytest.mark.asyncio
    async def test_stop_sentinel_enqueued_even_when_full(self):
        cap = AudioCapture(max_queue_chunks=1)
        cap._enqueue_chunk((0.0, _chunk()))  # queue now full
        cap.stop()  # must make room for the None sentinel

        # The sentinel is somewhere in the queue; draining reaches it.
        cap.is_capturing = True  # let _drain_queue run one pass
        items = []

        async def drain():
            async for c in cap._drain_queue():
                items.append(c)

        await asyncio.wait_for(drain(), timeout=2.0)
        # Sentinel ended the drain; at most the one real chunk was yielded.
        assert len(items) <= 1


# ---------------------------------------------------------------------------
# _drain_queue — offset tracking
# ---------------------------------------------------------------------------


class TestDrainQueueOffsets:
    @pytest.mark.asyncio
    async def test_yields_chunks_and_records_offsets(self):
        cap = AudioCapture(max_queue_chunks=8)
        cap.is_capturing = True
        cap._queue.put_nowait((0.0, _chunk()))
        cap._queue.put_nowait((5.0, _chunk()))
        cap._queue.put_nowait((10.0, _chunk()))
        cap._queue.put_nowait(None)  # stop sentinel

        seen_offsets = []

        async def drain():
            async for _chunk_out in cap._drain_queue():
                seen_offsets.append(cap.last_chunk_offset)

        await asyncio.wait_for(drain(), timeout=2.0)

        assert seen_offsets == [0.0, 5.0, 10.0]
        assert cap.last_chunk_offset == 10.0

    @pytest.mark.asyncio
    async def test_sentinel_stops_iteration(self):
        cap = AudioCapture(max_queue_chunks=8)
        cap.is_capturing = True
        cap._queue.put_nowait((1.0, _chunk()))
        cap._queue.put_nowait(None)
        cap._queue.put_nowait((99.0, _chunk()))  # after sentinel — must be ignored

        count = 0

        async def drain():
            nonlocal count
            async for _c in cap._drain_queue():
                count += 1

        await asyncio.wait_for(drain(), timeout=2.0)
        assert count == 1
        assert cap.last_chunk_offset == 1.0

    @pytest.mark.asyncio
    async def test_offsets_are_continuous_across_dropped_silence(self):
        # Simulate the capture thread's contract: silent chunks are NOT
        # enqueued but still advance the offset, so an emitted chunk after a
        # silent gap carries its true audio position (not a compressed one).
        cap = AudioCapture(max_queue_chunks=8)
        cap.is_capturing = True
        # chunk at 0s emitted, 5s + 10s silent (skipped), 15s emitted.
        cap._queue.put_nowait((0.0, _chunk()))
        cap._queue.put_nowait((15.0, _chunk()))
        cap._queue.put_nowait(None)

        seen = []

        async def drain():
            async for _c in cap._drain_queue():
                seen.append(cap.last_chunk_offset)

        await asyncio.wait_for(drain(), timeout=2.0)
        assert seen == [0.0, 15.0]


# ---------------------------------------------------------------------------
# begin() / start() — pre-roll (5c)
# ---------------------------------------------------------------------------


class TestBeginPreroll:
    @pytest.mark.asyncio
    async def test_begin_is_idempotent_when_already_capturing(self):
        # When capture is already running, begin() must return immediately
        # without touching audio hardware (the guard is before the pyaudio
        # import, so this is safe without a device).
        cap = AudioCapture()
        cap.is_capturing = True
        cap.begin()  # must not raise / must not import pyaudio
        assert cap.is_capturing is True

    @pytest.mark.asyncio
    async def test_start_reuses_already_begun_capture(self, monkeypatch):
        # Simulates 5c pre-roll: begin() was called early, so start() must NOT
        # begin again — it just drains the buffered (pre-roll) + live chunks.
        cap = AudioCapture(max_queue_chunks=8)
        began_again = {"v": False}

        def _fake_begin():
            began_again["v"] = True

        monkeypatch.setattr(cap, "begin", _fake_begin)
        cap.is_capturing = True  # pretend pre-roll already started
        cap._queue.put_nowait((0.0, _chunk()))
        cap._queue.put_nowait(None)

        chunks = []
        async for c in cap.start():
            chunks.append(c)

        assert began_again["v"] is False
        assert len(chunks) == 1

    @pytest.mark.asyncio
    async def test_start_begins_when_not_yet_capturing(self, monkeypatch):
        # No pre-roll: start() must lazily begin capture itself.
        cap = AudioCapture(max_queue_chunks=8)

        def _fake_begin():
            cap.is_capturing = True
            cap._queue.put_nowait((0.0, _chunk()))
            cap._queue.put_nowait(None)

        monkeypatch.setattr(cap, "begin", _fake_begin)

        chunks = []
        async for c in cap.start():
            chunks.append(c)

        assert len(chunks) == 1


# ---------------------------------------------------------------------------
# Speaker attribution / capture mode (5d)
# ---------------------------------------------------------------------------


class TestSpeakerAttribution:
    def test_defaults_to_loopback_audio_tag(self):
        cap = AudioCapture()
        assert cap.capture_mode == "loopback"
        assert cap.speaker_label == "(audio)"

    def test_microphone_mode_and_label_stored(self):
        cap = AudioCapture(capture_mode="input", speaker_label="(me)")
        assert cap.capture_mode == "input"
        assert cap.speaker_label == "(me)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
