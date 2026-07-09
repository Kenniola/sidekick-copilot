"""Local speech recognition via faster-whisper (CTranslate2).

Sidekick uses local Whisper exclusively for transcription. Audio is captured
from WASAPI loopback, processed in-memory on CPU, and **never leaves the
device** — giving a clean privacy posture for customer meetings.

Configuration (``speech`` section in the customer YAML config)::

    speech:
      backend: whisper        # only supported value
      language: en-GB         # informational; Whisper uses language="en"
      model: small.en         # base.en | small.en | medium.en | large-v3
      compute_type: int8      # int8 | int8_float16 | float16 | float32

Environment overrides::

    SIDEKICK_WHISPER_MODEL=small.en
    SIDEKICK_WHISPER_COMPUTE=int8
    SIDEKICK_WHISPER_DEVICE=auto    # auto | cpu | cuda

Device selection: the faster-whisper (CTranslate2) backend supports **CUDA GPU
and CPU only** — there is no NPU/DirectML path. With ``device: auto`` (the
default) Sidekick uses a CUDA GPU when one is available (compute ``float16``)
and otherwise falls back to CPU (compute ``int8``). VAD gating is always on via
``vad_filter=True`` in :meth:`WhisperRecogniser.transcribe_chunk`.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections import deque
from typing import Protocol

import numpy as np

from sidekick.analyst.context import TranscriptLine

logger = logging.getLogger(__name__)


def _cuda_available() -> bool:
    """Return True if CTranslate2 reports at least one usable CUDA device."""
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:  # noqa: BLE001 — any failure means "no usable GPU"
        logger.debug("CUDA detection failed; assuming CPU", exc_info=True)
        return False


def _resolve_device_and_compute(
    requested_device: str | None, requested_compute: str | None
) -> tuple[str, str]:
    """Resolve the concrete (device, compute_type) for faster-whisper.

    ``requested_device`` may be ``auto`` / ``cpu`` / ``cuda`` (or None →
    ``SIDEKICK_WHISPER_DEVICE`` env → ``auto``). ``auto`` picks CUDA when a GPU
    is present, else CPU. When ``requested_compute`` is not set, a sensible
    compute type is paired with the device (``float16`` on GPU, ``int8`` on
    CPU); an explicit compute type is always honoured.
    """
    device = (
        requested_device
        or os.environ.get("SIDEKICK_WHISPER_DEVICE")
        or "auto"
    ).lower()

    if device == "cuda":
        resolved = "cuda" if _cuda_available() else "cpu"
        if resolved == "cpu":
            logger.warning(
                "device=cuda requested but no CUDA GPU detected; using CPU."
            )
        device = resolved
    elif device == "auto":
        device = "cuda" if _cuda_available() else "cpu"
    elif device != "cpu":
        logger.warning("Unknown device=%r; using CPU.", device)
        device = "cpu"

    compute = (
        requested_compute
        or os.environ.get("SIDEKICK_WHISPER_COMPUTE")
        or ("float16" if device == "cuda" else "int8")
    )
    return device, compute


def _format_ts(seconds: float) -> str:
    """Convert seconds to VTT timestamp string HH:MM:SS.mmm."""
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:06.3f}"


# Number of trailing words carried from one chunk into the next as a coherence
# prompt (5e). Bounded so the combined prompt stays well within Whisper's
# prompt token budget and a bad chunk cannot poison many later chunks.
_COHERENCE_TAIL_WORDS = 24

# Cross-speaker echo suppression (Phase 2 / C3 Tier 1). A near-duplicate line
# from the OTHER capture within this window is treated as speaker bleed (mic
# picking up loopback output, or vice-versa) and dropped.
_ECHO_WINDOW_SECONDS = 2.0
_ECHO_SIMILARITY = 0.85
_RECENT_LINES_MAXLEN = 24
# Below this length, utterances ("Yes.", "Okay", "Right") are too common and
# too short for reliable similarity, so they are never treated as echoes — both
# sides genuinely saying "yes" must survive.
_ECHO_MIN_CHARS = 12


def _combine_prompt(vocab_prompt: str | None, prev_tail: str) -> str | None:
    """Combine the domain-vocabulary prior (5b) with the previous chunk's tail (5e).

    Either part may be empty/None. Returns ``None`` when both are empty so
    Whisper keeps its default (unconditioned) behaviour.
    """
    parts = [p for p in (vocab_prompt, prev_tail) if p]
    return " ".join(parts) if parts else None


def _tail_text(text: str, max_words: int = _COHERENCE_TAIL_WORDS) -> str:
    """Return the last ``max_words`` words of ``text`` (the cross-chunk tail)."""
    words = text.split()
    return " ".join(words[-max_words:])



class SpeechRecogniser(Protocol):
    """Interface for speech-to-text backends."""

    async def transcribe_chunk(
        self,
        audio: np.ndarray,
        sample_rate: int = 16_000,
        chunk_start_offset: float = 0.0,
        initial_prompt: str | None = None,
        speaker: str = "(audio)",
    ) -> list[TranscriptLine]: ...

    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# Whisper recogniser (local, on-device, no network)
# ---------------------------------------------------------------------------


class WhisperRecogniser:
    """Local speech recognition using faster-whisper (CTranslate2).

    Runs on CPU with no API keys and no network calls. Models auto-download
    on first use into the Hugging Face cache.

    Model sizes (approximate):
      - ``base.en``   ~150 MB  fastest, ~8-10% WER
      - ``small.en``  ~470 MB  balanced, ~5-7% WER  **(default)**
      - ``medium.en`` ~1.5 GB  slower, ~4-6% WER
      - ``large-v3``  ~3.1 GB  multilingual, best accuracy
    """

    DEFAULT_MODEL = "small.en"
    DEFAULT_COMPUTE = "int8"

    def __init__(
        self,
        model_size: str | None = None,
        compute_type: str | None = None,
        device: str | None = None,
        *,
        vad_min_silence_ms: int = 500,
        no_speech_threshold: float = 0.6,
        log_prob_threshold: float = -1.0,
        compression_ratio_threshold: float = 2.4,
        echo_suppression: bool = True,
    ):
        model_size = (
            model_size
            or os.environ.get("SIDEKICK_WHISPER_MODEL")
            or self.DEFAULT_MODEL
        )
        device, compute_type = _resolve_device_and_compute(device, compute_type)
        logger.info(
            "Loading Whisper model: %s (device=%s, compute=%s)...",
            model_size,
            device,
            compute_type,
        )

        from faster_whisper import WhisperModel

        try:
            self.model = WhisperModel(
                model_size, device=device, compute_type=compute_type
            )
        except Exception as e:  # noqa: BLE001 — GPU init can fail at runtime
            if device != "cpu":
                logger.warning(
                    "Whisper init failed on device=%s (%s); falling back to CPU/int8.",
                    device,
                    e,
                )
                device, compute_type = "cpu", "int8"
                self.model = WhisperModel(
                    model_size, device=device, compute_type=compute_type
                )
            else:
                raise
        self.model_size = model_size
        self.compute_type = compute_type
        self.device = device
        # Repetition hallucination guard, keyed by speaker (C2.3). Keying by
        # speaker stops one speaker's repeated short utterance from suppressing
        # a different speaker's identical one (e.g. both sides saying "Yes.").
        # Serial transcription means these are never mutated concurrently.
        self._last_text: dict[str, str] = {}
        self._repeat_count: dict[str, int] = {}
        # Per-speaker trailing text from the previous chunk (5e). Keyed by
        # speaker so dual capture ("(me)"/"(remote)") keeps each side's
        # cross-chunk continuity independent.
        self._prev_tail: dict[str, str] = {}
        # Decode-threshold tuning (Phase 2 / C2) passed to faster-whisper to
        # cut hallucinations and edge clipping.
        self._vad_min_silence_ms = vad_min_silence_ms
        self._no_speech_threshold = no_speech_threshold
        self._log_prob_threshold = log_prob_threshold
        self._compression_ratio_threshold = compression_ratio_threshold
        # Cross-speaker echo suppression (Phase 2 / C3 Tier 1).
        self._echo_suppression = echo_suppression
        self._recent_lines: deque = deque(maxlen=_RECENT_LINES_MAXLEN)
        logger.info("Whisper model loaded (%s, device=%s).", model_size, device)

    async def transcribe_chunk(
        self,
        audio: np.ndarray,
        sample_rate: int = 16_000,
        chunk_start_offset: float = 0.0,
        initial_prompt: str | None = None,
        speaker: str = "(audio)",
    ) -> list[TranscriptLine]:
        """Transcribe a single audio chunk.

        Args:
            audio: float32 mono PCM samples in [-1, 1].
            sample_rate: sample rate of ``audio`` (informational; Whisper
                expects 16 kHz upstream).
            chunk_start_offset: seconds elapsed since the listen session
                started up to the **start** of this chunk. Added to every
                segment offset so transcript timestamps reflect the position
                within the meeting, not within the 5-second chunk.
            initial_prompt: optional domain-vocabulary hint (Phase 5b) biasing
                Whisper toward expected proper nouns/jargon. ``None`` preserves
                Whisper's default behaviour.
            speaker: speaker tag applied to every emitted line (Phase 5d).
                Defaults to ``"(audio)"`` (loopback-only). When microphone
                capture is enabled the caller passes ``"(remote)"`` for system
                audio and ``"(me)"`` for the local microphone.

        Returns:
            List of ``TranscriptLine`` with session-relative VTT timestamps.

        The CPU-bound Whisper inference runs in a worker thread via
        :func:`asyncio.to_thread` so it never blocks the asyncio event loop.
        Blocking the loop here would starve concurrent research tasks (whose
        ``wait_for`` timeouts are wall-clock and would expire) and make the
        ``status`` tool unresponsive while a chunk is transcribing.
        """
        if sample_rate != 16_000:
            logger.debug("Unusual sample_rate=%s for Whisper input", sample_rate)

        return await asyncio.to_thread(
            self._transcribe_sync, audio, chunk_start_offset, initial_prompt, speaker
        )

    def _transcribe_sync(
        self,
        audio: np.ndarray,
        chunk_start_offset: float,
        initial_prompt: str | None,
        speaker: str = "(audio)",
    ) -> list[TranscriptLine]:
        """Synchronous Whisper inference + segment filtering.

        Runs in a worker thread (see :meth:`transcribe_chunk`). Transcriptions
        are issued serially by the consume loop, so the repetition-filter state
        (``_last_text`` / ``_repeat_count``) and the cross-chunk coherence tail
        (``_prev_tail``) are never mutated concurrently.
        """
        # 5e cross-chunk coherence: prepend the previous chunk's trailing words
        # for this speaker to the vocabulary prior so Whisper has lexical
        # context across the 5-second chunk boundary (our chunk-at-a-time
        # equivalent of condition_on_previous_text). Reduces words clipped or
        # garbled at chunk edges and improves proper-noun continuity.
        prev_tail = self._prev_tail.get(speaker, "")
        effective_prompt = _combine_prompt(initial_prompt, prev_tail)

        segments, _info = self.model.transcribe(
            audio,
            beam_size=5,
            language="en",
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=getattr(self, "_vad_min_silence_ms", 500)
            ),
            no_speech_threshold=getattr(self, "_no_speech_threshold", 0.6),
            log_prob_threshold=getattr(self, "_log_prob_threshold", -1.0),
            compression_ratio_threshold=getattr(
                self, "_compression_ratio_threshold", 2.4
            ),
            initial_prompt=effective_prompt,
        )

        lines: list[TranscriptLine] = []
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue

            # Filter high no-speech probability segments (Whisper hallucination guard)
            if seg.no_speech_prob > 0.7:
                logger.debug(
                    "Dropping segment (no_speech_prob=%.2f): %s",
                    seg.no_speech_prob,
                    text,
                )
                continue

            # Repetition filter — Whisper hallucination guard (per-speaker, C2.3)
            if text == self._last_text.get(speaker, ""):
                count = self._repeat_count.get(speaker, 0) + 1
                self._repeat_count[speaker] = count
                if count >= 3:
                    logger.debug("Dropping repeated hallucination: %s", text)
                    continue
            else:
                self._last_text[speaker] = text
                self._repeat_count[speaker] = 0

            seg_offset = chunk_start_offset + seg.start
            # Cross-speaker echo guard (C3 Tier 1): drop the same utterance when
            # it bleeds into the other capture. No-op for single-source capture.
            if getattr(self, "_echo_suppression", True) and self._is_echo(
                seg_offset, speaker, text
            ):
                logger.debug("Dropping cross-speaker echo: %s", text)
                continue

            lines.append(
                TranscriptLine(
                    start=_format_ts(seg_offset),
                    end=_format_ts(chunk_start_offset + seg.end),
                    speaker=speaker,
                    text=text,
                )
            )
            recent = getattr(self, "_recent_lines", None)
            if recent is not None:
                recent.append((seg_offset, speaker, text))

        # 5e: remember this chunk's trailing words as the coherence prompt for
        # this speaker's next chunk. Only update when the chunk produced text so
        # a silent/dropped chunk does not erase the running context.
        if lines:
            self._prev_tail[speaker] = _tail_text(
                " ".join(ln.text for ln in lines)
            )

        return lines

    def _is_echo(self, offset: float, speaker: str, text: str) -> bool:
        """True if a recent line from a DIFFERENT speaker near-duplicates this.

        Guards dual (mic + loopback) capture against speaker bleed: the same
        utterance picked up by both devices would otherwise appear twice under
        different speaker tags. Same-speaker repeats are handled separately by
        the repetition guard.
        """
        from sidekick.dedup import similarity

        recent = getattr(self, "_recent_lines", None)
        if not recent or len(text) < _ECHO_MIN_CHARS:
            return False
        for rec_off, rec_speaker, rec_text in recent:
            if rec_speaker == speaker:
                continue
            if abs(offset - rec_off) > _ECHO_WINDOW_SECONDS:
                continue
            if similarity(text, rec_text) >= _ECHO_SIMILARITY:
                return True
        return False

    def close(self) -> None:
        self.model = None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_recogniser(speech_config=None) -> SpeechRecogniser:
    """Create the speech recognition backend.

    Sidekick uses local Whisper exclusively. ``backend: azure`` is no longer
    supported (see CHANGELOG v0.3.0 for the rationale). Configs that still
    request ``azure`` log a warning and fall back to Whisper.

    Args:
        speech_config: A ``SpeechConfig`` from the customer YAML. May be
            ``None`` (defaults applied).

    Returns:
        A configured ``WhisperRecogniser`` instance.
    """
    if speech_config is not None and getattr(speech_config, "backend", "whisper") not in (
        "whisper",
        "",
        None,
    ):
        logger.warning(
            "speech.backend=%r is no longer supported. Using local Whisper. "
            "Update your customer YAML to 'backend: whisper' to silence this warning.",
            speech_config.backend,
        )

    model = None
    compute = None
    device = None
    decode_kwargs: dict = {}
    if speech_config is not None:
        model = getattr(speech_config, "model", None)
        compute = getattr(speech_config, "compute_type", None)
        device = getattr(speech_config, "device", None)
        decode_kwargs = dict(
            vad_min_silence_ms=getattr(speech_config, "vad_min_silence_ms", 500),
            no_speech_threshold=getattr(speech_config, "no_speech_threshold", 0.6),
            log_prob_threshold=getattr(speech_config, "log_prob_threshold", -1.0),
            compression_ratio_threshold=getattr(
                speech_config, "compression_ratio_threshold", 2.4
            ),
            echo_suppression=getattr(speech_config, "echo_suppression", True),
        )

    logger.info("Using local Whisper backend (on-device, no network).")
    return WhisperRecogniser(
        model_size=model, compute_type=compute, device=device, **decode_kwargs
    )
