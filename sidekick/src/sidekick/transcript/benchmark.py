"""Speech-to-text model benchmark (Phase 0 / C1).

Measures the **real-time factor (RTF)** of candidate Whisper models on *this*
machine so the default model can be chosen from evidence rather than assumed.
RTF = transcription_seconds / audio_seconds; RTF < 1 means faster than
real-time. Sidekick needs headroom for the rest of the live pipeline, so the
default acceptance threshold is 0.7.

Design: the pure decision logic (``real_time_factor``, ``recommend_model``,
``run_benchmark``) is separated from the heavy I/O (model load, audio decode,
loopback recording) via an injected ``transcribe_fn``, so it is unit-testable
offline without downloading weights or touching audio hardware.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger(__name__)

# Candidate models, ordered smallest → most accurate. `distil-large-v3` is the
# recommended accuracy target (≈ large-v3 WER, 6.3× faster, low hallucination).
CANDIDATE_MODELS: list[str] = ["small.en", "medium.en", "distil-large-v3"]

# Acceptance ceiling: transcription must use < 70% of real time to leave
# headroom for classification, research, and the rest of the live loop.
DEFAULT_RTF_THRESHOLD = 0.7

# Relative accuracy ranking (higher = more accurate) used to pick the *best*
# model among those that meet the RTF threshold.
_ACCURACY_RANK: dict[str, int] = {
    "tiny.en": 0, "tiny": 0,
    "base.en": 1, "base": 1,
    "small.en": 2, "small": 2,
    "medium.en": 3, "medium": 3,
    "distil-large-v2": 4, "distil-large-v3": 4,
    "large-v1": 5, "large-v2": 5, "turbo": 5, "large-v3-turbo": 5,
    "large-v3": 6,
}


def accuracy_rank(model: str) -> int:
    """Relative accuracy rank of a model (higher = better). Unknown → -1."""
    return _ACCURACY_RANK.get(model, -1)


# transcribe_fn(model_name, audio) -> (load_seconds, transcribe_seconds, text)
TranscribeFn = Callable[[str, object], "tuple[float, float, str]"]


@dataclass
class BenchmarkResult:
    """Outcome of benchmarking a single model."""

    model: str
    audio_seconds: float
    load_seconds: float | None = None
    transcribe_seconds: float | None = None
    rtf: float | None = None
    sample_text: str = ""
    ok: bool = True
    error: str = ""

    def passes(self, threshold: float = DEFAULT_RTF_THRESHOLD) -> bool:
        return self.ok and self.rtf is not None and self.rtf < threshold


def real_time_factor(process_seconds: float, audio_seconds: float) -> float:
    """RTF = process/audio. Returns +inf for non-positive audio duration."""
    if audio_seconds <= 0:
        return float("inf")
    return process_seconds / audio_seconds


def run_benchmark(
    audio: object,
    audio_seconds: float,
    candidates: list[str],
    transcribe_fn: TranscribeFn,
) -> list[BenchmarkResult]:
    """Benchmark each candidate model on the same audio.

    ``transcribe_fn`` performs the (heavy) model load + transcription and
    returns ``(load_seconds, transcribe_seconds, text)``. A failure for one
    model is captured on its result and does not abort the others.
    """
    results: list[BenchmarkResult] = []
    for model in candidates:
        try:
            load_s, tx_s, text = transcribe_fn(model, audio)
            results.append(
                BenchmarkResult(
                    model=model,
                    audio_seconds=audio_seconds,
                    load_seconds=load_s,
                    transcribe_seconds=tx_s,
                    rtf=real_time_factor(tx_s, audio_seconds),
                    sample_text=(text or "").strip()[:200],
                    ok=True,
                )
            )
        except Exception as e:  # noqa: BLE001 — one model failing must not abort
            logger.debug("Benchmark failed for %s", model, exc_info=True)
            results.append(
                BenchmarkResult(
                    model=model,
                    audio_seconds=audio_seconds,
                    ok=False,
                    error=f"{type(e).__name__}: {e}",
                )
            )
    return results


def recommend_model(
    results: list[BenchmarkResult], threshold: float = DEFAULT_RTF_THRESHOLD
) -> str | None:
    """Return the most accurate model that meets the RTF threshold, else None."""
    passing = [r for r in results if r.passes(threshold)]
    if not passing:
        return None
    return max(passing, key=lambda r: accuracy_rank(r.model)).model


# ---------------------------------------------------------------------------
# Heavy I/O (lazy imports; not exercised by unit tests)
# ---------------------------------------------------------------------------


def load_audio(path: str) -> "tuple[object, float]":
    """Decode an audio file to 16 kHz mono float32 and return (samples, seconds)."""
    from faster_whisper.audio import decode_audio

    samples = decode_audio(path, sampling_rate=16_000)
    return samples, len(samples) / 16_000


def record_loopback(seconds: int) -> "tuple[object, float]":
    """Record ~``seconds`` of system audio (WASAPI loopback) for benchmarking.

    Best-effort: requires the ``live`` extra (pyaudiowpatch). Raises on failure
    so the caller can fall back to ``--audio``.
    """
    import asyncio

    import numpy as np

    from sidekick.transcript.audio_capture import AudioCapture

    async def _rec() -> "np.ndarray":
        cap = AudioCapture(capture_mode="loopback", chunk_duration=1.0)
        chunks: list = []
        collected = 0.0
        try:
            async for chunk in cap.start():
                chunks.append(chunk)
                collected += len(chunk) / 16_000
                if collected >= seconds:
                    break
        finally:
            cap.stop()
        return (
            np.concatenate(chunks)
            if chunks
            else np.zeros(0, dtype=np.float32)
        )

    # Wall-clock guard so a silent device cannot hang the benchmark forever.
    async def _guarded():
        return await asyncio.wait_for(_rec(), timeout=max(10, seconds * 3))

    samples = asyncio.run(_guarded())
    return samples, len(samples) / 16_000


def default_transcribe_fn(model_name: str, audio: object) -> "tuple[float, float, str]":
    """Load ``model_name`` and transcribe ``audio``, timing load and decode.

    Uses the same device/compute resolution as the live recogniser so the RTF
    reflects the real runtime configuration.
    """
    from faster_whisper import WhisperModel

    from sidekick.transcript.speech_recogniser import _resolve_device_and_compute

    device, compute = _resolve_device_and_compute(None, None)

    t0 = time.perf_counter()
    model = WhisperModel(model_name, device=device, compute_type=compute)
    load_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    segments, _info = model.transcribe(
        audio,
        beam_size=5,
        language="en",
        vad_filter=True,
        condition_on_previous_text=False,
    )
    # ``segments`` is a generator — consuming it here is what actually runs
    # (and therefore times) the transcription.
    text = " ".join(seg.text for seg in segments)
    tx_s = time.perf_counter() - t1
    return load_s, tx_s, text
