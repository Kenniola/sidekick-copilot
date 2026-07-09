"""WASAPI loopback audio capture — records system audio output.

Captures whatever is playing through the default speakers or headset
(i.e., the meeting audio from Teams/Zoom/etc.) and yields PCM chunks
suitable for speech recognition.

Windows-only — requires PyAudioWPatch for WASAPI loopback support.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import AsyncIterator

import numpy as np

logger = logging.getLogger(__name__)

# Target format for speech recognition
TARGET_RATE = 16_000
TARGET_CHANNELS = 1


class AudioCapture:
    """Capture system audio via WASAPI loopback and yield chunked PCM data.

    Usage::

        capture = AudioCapture(chunk_duration=5.0)
        async for chunk in capture.start():
            # chunk is np.ndarray, float32, 16kHz, mono
            lines = await recogniser.transcribe_chunk(chunk)
    """

    def __init__(
        self,
        device_index: int | None = None,
        chunk_duration: float = 5.0,
        silence_threshold: float = 0.002,
        max_queue_chunks: int = 32,
        capture_mode: str = "loopback",
        speaker_label: str = "(audio)",
    ):
        self.device_index = device_index
        self.chunk_duration = chunk_duration
        self.silence_threshold = silence_threshold
        self.is_capturing = False

        # Capture source (5d). ``loopback`` records system audio output (the
        # remote participants via WASAPI loopback); ``input`` records a
        # physical input device (the local microphone). ``speaker_label`` tags
        # every transcript line this capture produces so the analyst can tell
        # who spoke. Loopback-only sessions keep the historical ``(audio)``
        # tag; dual capture uses ``(remote)`` and ``(me)``.
        self.capture_mode = capture_mode
        self.speaker_label = speaker_label

        # Bounded queue with a drop-to-latest policy (5a). Under sustained CPU
        # overload transcription falls behind real time; rather than let the
        # backlog (and memory) grow without limit, ``_enqueue_chunk`` drops the
        # oldest queued chunk so live suggestions stay current. In normal
        # operation the queue sits near-empty and nothing is dropped.
        self._queue: asyncio.Queue[tuple[float, np.ndarray] | None] = asyncio.Queue(
            maxsize=max(1, max_queue_chunks)
        )
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        # Audio-position bookkeeping (5a). ``last_chunk_offset`` is the
        # start offset (seconds from capture start) of the most recently
        # yielded chunk, derived from audio actually captured so it is immune
        # to processing backlog (fixes the wall-clock timestamp drift).
        # ``dropped_chunks`` counts chunks discarded by the drop-to-latest
        # policy, for diagnostics.
        self.last_chunk_offset: float = 0.0
        self.dropped_chunks: int = 0

        # Resolved at start time
        self._source_rate: int = 0
        self._source_channels: int = 0

        # PyAudio handle, owned across begin()/start()/stop() so capture can be
        # started early for pre-roll (5c) and torn down cleanly later.
        self._pa = None

    async def start(self) -> AsyncIterator[np.ndarray]:
        """Begin capturing (if not already) and yield audio chunks.

        If :meth:`begin` was already called (5c pre-roll), this reuses the
        running capture and simply drains the buffered + live chunks; otherwise
        it starts capture now. Each yielded array is approximately
        ``chunk_duration`` seconds long. Silent chunks (RMS below
        ``silence_threshold``) are skipped.
        """
        if not self.is_capturing:
            self.begin()

        try:
            # Drain queued chunks (offset-tagged) and yield the audio.
            async for chunk in self._drain_queue():
                yield chunk
        finally:
            self.is_capturing = False
            # Wait for capture thread to close its stream before
            # terminating PyAudio — avoids C-level crash.
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=3.0)
            if self._pa is not None:
                self._pa.terminate()
                self._pa = None

    def begin(self) -> None:
        """Open the capture device and start the reader thread WITHOUT draining.

        Lets the WASAPI stream buffer audio (pre-roll, 5c) while slower startup
        work — notably loading the Whisper model — runs concurrently, so the
        opening of the meeting is not lost while the model loads. Idempotent: a
        subsequent :meth:`start` reuses an already-begun capture. Must be called
        from the event-loop thread (it captures the running loop for
        thread-safe enqueues from the reader thread).
        """
        if self.is_capturing:
            return

        import pyaudiowpatch as pyaudio

        self._pa = pyaudio.PyAudio()
        device = self._resolve_device(self._pa)
        self._source_rate = int(device["defaultSampleRate"])
        self._source_channels = device["maxInputChannels"]

        logger.info(
            "Audio capture: %s (rate=%d, ch=%d)",
            device["name"],
            self._source_rate,
            self._source_channels,
        )

        self.is_capturing = True
        self._loop = asyncio.get_running_loop()

        # Start capture thread
        self._thread = threading.Thread(
            target=self._capture_thread,
            args=(self._pa, device),
            daemon=True,
        )
        self._thread.start()

    async def _drain_queue(self) -> AsyncIterator[np.ndarray]:
        """Yield audio chunks from the queue, tracking each chunk's offset.

        Queue items are ``(start_offset_seconds, chunk)`` tuples produced by the
        capture thread, or ``None`` (the stop sentinel). The start offset is
        recorded on ``last_chunk_offset`` immediately before the chunk is
        yielded so the consumer can read the audio position of the chunk it
        just received. Separated from :meth:`start` (which owns the hardware)
        so the draining/offset logic is testable without audio hardware.
        """
        while self.is_capturing:
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=2.0)
            except asyncio.TimeoutError:
                continue
            if item is None:
                break
            offset, chunk = item
            self.last_chunk_offset = offset
            yield chunk

    def _enqueue_chunk(self, item: tuple[float, np.ndarray]) -> None:
        """Enqueue a captured chunk, dropping the oldest if the queue is full.

        Runs on the event loop thread (scheduled via ``call_soon_threadsafe``).
        Drop-to-latest keeps the live transcript current under sustained
        overload instead of letting latency and memory grow unbounded.
        """
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            try:
                self._queue.get_nowait()  # discard the oldest queued chunk
                self.dropped_chunks += 1
            except asyncio.QueueEmpty:
                pass
            try:
                self._queue.put_nowait(item)
            except asyncio.QueueFull:
                pass

    def stop(self):
        """Signal the capture to stop."""
        self.is_capturing = False
        # Push sentinel to unblock the async generator. If the queue is full,
        # make room so the sentinel is never lost.
        try:
            self._queue.put_nowait(None)
        except asyncio.QueueFull:
            try:
                self._queue.get_nowait()
                self.dropped_chunks += 1
            except asyncio.QueueEmpty:
                pass
            try:
                self._queue.put_nowait(None)
            except asyncio.QueueFull:
                pass
        logger.info("Audio capture stopped.")

    def list_devices(self) -> list[dict]:
        """List available WASAPI loopback devices (for diagnostics)."""
        import pyaudiowpatch as pyaudio

        pa = pyaudio.PyAudio()
        devices = []
        try:
            for i in range(pa.get_device_count()):
                dev = pa.get_device_info_by_index(i)
                if dev.get("isLoopbackDevice"):
                    devices.append({
                        "index": i,
                        "name": dev["name"],
                        "channels": dev["maxInputChannels"],
                        "rate": int(dev["defaultSampleRate"]),
                    })
        finally:
            pa.terminate()
        return devices

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve_device(self, pa) -> dict:
        """Find the device to capture from (loopback output or input mic)."""
        import pyaudiowpatch as pyaudio

        if self.device_index is not None:
            dev = pa.get_device_info_by_index(self.device_index)
            if self.capture_mode == "input":
                if dev["maxInputChannels"] < 1 or dev.get("isLoopbackDevice"):
                    raise RuntimeError(
                        f"Device {self.device_index} ({dev['name']}) is not a "
                        "microphone input device."
                    )
                return dev
            if not dev.get("isLoopbackDevice"):
                raise RuntimeError(
                    f"Device {self.device_index} ({dev['name']}) is not a loopback device."
                )
            return dev

        if self.capture_mode == "input":
            return self._resolve_input_device(pa)

        # Auto-detect: find loopback for the default WASAPI output
        wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_out = pa.get_device_info_by_index(wasapi["defaultOutputDevice"])
        default_name_prefix = default_out["name"].split("(")[0].strip()

        for i in range(pa.get_device_count()):
            dev = pa.get_device_info_by_index(i)
            if dev.get("isLoopbackDevice") and dev["name"].startswith(default_name_prefix):
                return dev

        # Fallback: use any loopback device
        for i in range(pa.get_device_count()):
            dev = pa.get_device_info_by_index(i)
            if dev.get("isLoopbackDevice"):
                logger.warning(
                    "Could not match default output; falling back to: %s",
                    dev["name"],
                )
                return dev

        available = self.list_devices()
        raise RuntimeError(
            f"No WASAPI loopback devices found. Available devices: {available}"
        )

    def _resolve_input_device(self, pa) -> dict:
        """Find the default WASAPI microphone input device (5d)."""
        import pyaudiowpatch as pyaudio

        wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_in_index = wasapi.get("defaultInputDevice", -1)
        if default_in_index is not None and default_in_index >= 0:
            dev = pa.get_device_info_by_index(default_in_index)
            if dev["maxInputChannels"] >= 1 and not dev.get("isLoopbackDevice"):
                return dev

        # Fallback: first non-loopback input device.
        for i in range(pa.get_device_count()):
            dev = pa.get_device_info_by_index(i)
            if dev["maxInputChannels"] >= 1 and not dev.get("isLoopbackDevice"):
                logger.warning(
                    "No default WASAPI input; falling back to: %s", dev["name"]
                )
                return dev

        raise RuntimeError("No microphone input device found for (me) capture.")

    def _capture_thread(self, pa, device: dict):
        """Run in a background thread — reads audio and pushes chunks to the async queue."""
        import pyaudiowpatch as pyaudio

        frames_per_chunk = int(self._source_rate * self.chunk_duration)
        buffer_size = 1024
        buffer: list[bytes] = []
        frames_collected = 0
        # Cumulative seconds of audio read from the stream before the current
        # chunk. Because the stream is read in real time this equals the true
        # audio position of each chunk, independent of how far transcription
        # has fallen behind. Silent chunks are counted here (so the timeline
        # stays continuous) even though they are not emitted.
        captured_seconds = 0.0

        stream = None
        try:
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=self._source_channels,
                rate=self._source_rate,
                input=True,
                input_device_index=device["index"],
                frames_per_buffer=buffer_size,
            )

            while self.is_capturing:
                try:
                    data = stream.read(buffer_size, exception_on_overflow=False)
                except OSError:
                    logger.warning("Audio read error, retrying...")
                    continue

                buffer.append(data)
                frames_collected += buffer_size

                if frames_collected >= frames_per_chunk:
                    chunk = self._process_buffer(buffer)
                    chunk_start_offset = captured_seconds
                    captured_seconds += frames_collected / self._source_rate
                    buffer.clear()
                    frames_collected = 0

                    if chunk is not None:
                        # Push to async queue from the thread (drop-to-latest).
                        if self._loop and self._loop.is_running():
                            self._loop.call_soon_threadsafe(
                                self._enqueue_chunk,
                                (chunk_start_offset, chunk),
                            )

        except Exception:
            logger.exception("Audio capture thread error")
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
            # Push sentinel
            if self._loop and self._loop.is_running():
                self._loop.call_soon_threadsafe(
                    self._enqueue_chunk_sentinel
                )

    def _enqueue_chunk_sentinel(self) -> None:
        """Enqueue the stop sentinel from the capture thread (drop-to-latest)."""
        try:
            self._queue.put_nowait(None)
        except asyncio.QueueFull:
            try:
                self._queue.get_nowait()
                self.dropped_chunks += 1
            except asyncio.QueueEmpty:
                pass
            try:
                self._queue.put_nowait(None)
            except asyncio.QueueFull:
                pass

    def _process_buffer(self, raw_frames: list[bytes]) -> np.ndarray | None:
        """Convert raw audio buffer to float32 16kHz mono. Returns None if silent."""
        # Combine raw int16 frames
        raw = b"".join(raw_frames)
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

        # Downmix to mono if stereo
        if self._source_channels > 1:
            audio = audio.reshape(-1, self._source_channels).mean(axis=1)

        # Resample to 16kHz if needed
        if self._source_rate != TARGET_RATE:
            num_samples = int(len(audio) * TARGET_RATE / self._source_rate)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, num_samples),
                np.arange(len(audio)),
                audio,
            ).astype(np.float32)

        # Silence detection
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < self.silence_threshold:
            logger.debug("Skipping silent chunk (RMS=%.4f)", rms)
            return None

        return audio
