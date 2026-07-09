"""Classification + dispatch engine (Phase 2e).

Extracted from ``server.py``: the orchestration that turns a batch of
transcript lines into queued action items, runs the research/prototype
pipelines, records results, and notifies. Also the domain auto-detection
that runs once enough transcript context has accumulated.

These functions take the :class:`~sidekick.session_state.SessionState`
explicitly and (for dispatch) a ``notify`` callable, so they can be tested
with mocked components. The live audio loop in ``server`` calls
``classify_and_dispatch`` on each batch.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Callable

from sidekick.session_state import SessionState

logger = logging.getLogger("sidekick")


async def detect_domains(state: SessionState) -> None:
    """Auto-detect domains from transcript after enough context accumulates.

    Runs a fast-tier LLM call on the first ~30 transcript lines to identify
    which technology domains are being discussed. Detected domains supplement
    (not replace) the configured domains from customers.yaml.
    """
    if state.domains_detected or not state.context or not state.config:
        return

    transcript_sample = state.context.full_transcript[-30:]
    if len(transcript_sample) < 10:
        return

    from sidekick.llm import call_llm, parse_llm_json
    sample_text = "\n".join(
        f"{getattr(line, 'speaker', '?')}: {getattr(line, 'text', str(line))}"
        for line in transcript_sample
    )

    try:
        result = await call_llm(
            system_prompt=(
                "You analyse meeting transcripts to detect technology domains "
                "being discussed. Return a JSON object with a single key "
                "\"domains\" containing a list of 3-8 domain strings. "
                "Examples: \"Microsoft Fabric\", \"Power BI\", \"Dynamics 365\", "
                "\"Azure APIM\", \"Azure Service Bus\", \"Oracle\", \"PostgreSQL\", "
                "\"AWS S3\", \"Databricks\", \"Legacy Systems\", \"Cosmos DB\". "
                "Only include domains clearly mentioned or implied."
            ),
            user_prompt=f"Transcript sample:\n{sample_text}",
            json_output=True,
            tier="fast",
            timeout=8,
        )
        data = parse_llm_json(result)
        detected = data.get("domains", [])
        if detected:
            # Merge with configured domains (no duplicates)
            existing = {d.lower() for d in state.config.domains}
            new_domains = [d for d in detected if d.lower() not in existing]
            if new_domains:
                state.config.domains.extend(new_domains)
                state.context.detected_domains = new_domains
                logger.info("Auto-detected domains: %s", new_domains)
                # Invalidate grounding cache since domains changed
                state.grounding_cache = None
    except Exception as e:
        logger.debug("Domain detection failed: %s", e)

    state.domains_detected = True


async def classify_and_dispatch(
    state: SessionState,
    lines: list,
    consecutive_errors: int,
    notify: Callable[[object], None],
) -> None:
    """Send accumulated transcript lines to the classifier and dispatch results."""
    state.classify_batch_count += 1

    action_items = await state.analyst.analyse_chunk(lines)

    # Auto-detect domains after 3 batches (enough transcript context)
    if state.classify_batch_count == 3 and not state.domains_detected:
        await detect_domains(state)

    # Accuracy pipeline (Phase 1): in accuracy_mode the fast classifier only
    # produces *candidates* — a periodic deep-tier adjudicator selects the few
    # worth surfacing. Default mode enqueues immediately (unchanged behaviour).
    if _accuracy_mode(state):
        await _accumulate_and_maybe_adjudicate(state, action_items)
    else:
        for item in action_items:
            await state.queue.enqueue(item)

    results = await state.queue.process_ready(
        research=state.research,
        prototype=state.prototype,
        context=state.context,
        domains=state.config.domains if state.config else None,
        notify=notify,
    )
    for result in results:
        state.session_log.record(result)
        logger.info(
            "Sidekick output: [%s] %s",
            result.action_type,
            result.question[:60],
        )

    # Notify for new findings (sound alert + log file). Results whose lead
    # answer was already surfaced early via streaming (``early_notified``) are
    # skipped here to avoid a duplicate toast.
    for result in results:
        if not getattr(result, "early_notified", False):
            notify(result)

    # Proactive advisor (Phase 9.3): occasionally suggest a question to ask the
    # client. Opt-in and slow-cadence, so it never competes with research.
    await maybe_auto_suggest(state, notify)

    # Adapt the Whisper vocabulary prior (Phase 5b) from LLM-corrected text:
    # thread/context key_facts and research answers spell proper nouns
    # correctly even when Whisper misheard them, so feeding them back improves
    # recognition of those terms in subsequent chunks.
    if state.vocabulary is not None:
        try:
            corrected: list[str] = []
            if state.context is not None:
                corrected.extend(getattr(state.context, "key_facts", []) or [])
                for thread in (getattr(state.context, "threads", {}) or {}).values():
                    corrected.extend(getattr(thread, "key_facts", []) or [])
            for result in results:
                corrected.append(getattr(result, "question", "") or "")
                corrected.append(getattr(result, "answer", "") or "")
            state.vocabulary.update(corrected)
        except Exception as e:  # noqa: BLE001 — prior is best-effort, never fatal
            logger.debug("Vocabulary update skipped: %s", e)


async def infer_objectives(state: SessionState) -> None:
    """Infer engagement objectives from the opening transcript (Phase 1 / A2).

    Runs a fast-tier LLM call over the first few minutes when no objectives are
    set explicitly, so the relevance adjudicator has goals to score against.
    Best-effort — failure leaves objectives empty (the adjudicator then infers
    from context). Runs at most once per session.
    """
    if state.objectives_inferred or not state.context or not state.config:
        return
    sample = state.context.full_transcript[-40:]
    if len(sample) < 10:
        return

    from sidekick.llm import call_llm, parse_llm_json

    sample_text = "\n".join(
        f"{getattr(line, 'speaker', '?')}: {getattr(line, 'text', str(line))}"
        for line in sample
    )
    try:
        result = await call_llm(
            system_prompt=(
                "You infer the concrete OBJECTIVES of a consulting meeting from "
                "its opening. Return a JSON object with a single key "
                '"objectives" containing 2-4 short goal strings (what the '
                "consultant is trying to achieve or decide). Be specific and "
                "grounded in what was actually said; do not invent goals."
            ),
            user_prompt=f"Meeting opening:\n{sample_text}",
            json_output=True,
            tier="fast",
            timeout=10,
        )
        data = parse_llm_json(result)
        objs = [
            str(o).strip() for o in (data.get("objectives") or []) if str(o).strip()
        ]
        if objs:
            state.context.objectives = objs[:4]
            logger.info("Auto-inferred objectives: %s", objs[:4])
    except Exception as e:  # noqa: BLE001 — best-effort; never fatal
        logger.debug("Objective inference failed: %s", e)
    state.objectives_inferred = True


def _accuracy_mode(state: SessionState) -> bool:
    """True when the two-stage accuracy pipeline is enabled for this session."""
    sens = getattr(state.config, "sensitivity", None) if state.config else None
    return bool(getattr(sens, "accuracy_mode", False))


def _should_adjudicate(state: SessionState, has_critical: bool) -> bool:
    """Flush the candidate buffer when the interval elapses or on a hedge.

    Interval-based cadence (a ceiling, configurable via
    ``SIDEKICK_ADJUDICATE_INTERVAL`` or ``adjudicator_interval_seconds``) with
    an early flush when a critical candidate (e.g. a consultant hedge) is
    pending so urgent items surface without waiting out the interval.
    """
    if not state.pending_candidates:
        return False
    sens = state.config.sensitivity
    interval = float(
        os.environ.get(
            "SIDEKICK_ADJUDICATE_INTERVAL",
            str(getattr(sens, "adjudicator_interval_seconds", 40)),
        )
    )
    if (time.monotonic() - state.last_adjudicate_time) >= interval:
        return True
    if getattr(sens, "adjudicator_pause_flush", True) and has_critical:
        return True
    return False


async def _ensure_objectives(state: SessionState) -> None:
    """Populate ``context.objectives`` from config, else auto-infer once."""
    if getattr(state.context, "objectives", None):
        return
    if state.objectives_inferred:
        return
    cfg_objs = list(getattr(state.config, "objectives", None) or [])
    if cfg_objs:
        state.context.objectives = cfg_objs
        state.objectives_inferred = True
        return
    if state.classify_batch_count >= 3:
        await infer_objectives(state)


async def _accumulate_and_maybe_adjudicate(
    state: SessionState, action_items: list
) -> None:
    """Buffer candidates and run the deep-tier adjudicator on cadence."""
    from sidekick.analyst.adjudicator import adjudicate

    if state.last_adjudicate_time == 0.0:
        state.last_adjudicate_time = time.monotonic()

    await _ensure_objectives(state)

    state.pending_candidates.extend(action_items)
    has_critical = any(
        getattr(i, "priority_score", 0.0) >= 0.9 for i in state.pending_candidates
    )
    if not _should_adjudicate(state, has_critical):
        return

    surfaced = await adjudicate(
        state.pending_candidates,
        state.context,
        state.config,
        getattr(state, "grounding_cache", None) or "",
        objectives=list(getattr(state.context, "objectives", None) or []),
        already_surfaced=list(getattr(state, "surfaced_questions", None) or []),
    )
    state.pending_candidates = []
    state.last_adjudicate_time = time.monotonic()
    for item in surfaced:
        await state.queue.enqueue(item)
    # Remember what we surfaced so later passes don't repeat it (keep last 40).
    state.surfaced_questions.extend(
        getattr(i, "question", "") for i in surfaced if getattr(i, "question", "")
    )
    del state.surfaced_questions[:-40]


_SUGGEST_SYSTEM_PROMPT = (
    "You are a senior consulting advisor listening to a live client meeting. "
    "Suggest at most ONE high-impact question the consultant should ask the "
    "client right now to move the engagement forward. Only suggest when there "
    "is a clear, valuable gap worth asking about; otherwise return an empty "
    "question. Mirror the client's language; never rephrase into jargon. "
    "Return JSON only."
)


async def _generate_suggestion(state: SessionState) -> dict | None:
    """Fast-tier advisor pass: one question to ask the client, or None."""
    from sidekick.llm import call_llm, parse_llm_json

    ctx = state.context
    recent = list(getattr(ctx, "full_transcript", []) or [])[-30:]
    transcript_block = "\n".join(
        f"{getattr(line, 'speaker', '?')}: {getattr(line, 'text', str(line))}"
        for line in recent
    )
    objectives = ", ".join(getattr(ctx, "objectives", []) or []) or "(not yet known)"
    already = (
        "\n".join(f"- {q}" for q in state.recent_suggestions[-8:]) or "(none)"
    )
    user_prompt = (
        f"Objectives: {objectives}\n\n"
        f"Recent conversation:\n{transcript_block}\n\n"
        f"Questions already suggested (do NOT repeat these or close variants):\n"
        f"{already}\n\n"
        'Return JSON: {"question": "<one question to ask the client, or an '
        'empty string if none is worth asking right now>", '
        '"rationale": "<one short line on why it matters>", '
        '"impact": "high|medium|low"}'
    )
    raw = await asyncio.wait_for(
        call_llm(
            system_prompt=_SUGGEST_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            json_output=True,
            timeout=20.0,
            tier="fast",
        ),
        timeout=25.0,
    )
    data = parse_llm_json(raw)
    question = str(data.get("question") or "").strip()
    if not question:
        return None
    return {
        "question": question,
        "rationale": str(data.get("rationale") or "").strip(),
        "impact": str(data.get("impact") or "medium").strip().lower(),
    }


async def maybe_auto_suggest(
    state: SessionState, notify: Callable[[object], None]
) -> None:
    """Proactively surface a question to ask the client (Phase 9.3, opt-in).

    Gated by ``sensitivity.auto_suggest`` (default off). Runs on a slow cadence
    so it never competes with research for attention, generates at most one
    suggestion per pass, dedupes against recent suggestions, and emits it as a
    low-key ``suggestion`` finding. Best-effort — any failure is swallowed.
    """
    if not state.config or not state.context:
        return
    sens = getattr(state.config, "sensitivity", None)
    if not getattr(sens, "auto_suggest", False):
        return

    # Need a few exchanges before advice is meaningful.
    transcript = getattr(state.context, "full_transcript", []) or []
    if len(transcript) < 6:
        return

    interval = float(getattr(sens, "auto_suggest_interval_seconds", 120) or 120)
    now = time.monotonic()
    # Arm the timer on the first eligible pass so we don't fire immediately.
    if state.last_suggest_time == 0.0:
        state.last_suggest_time = now
        return
    if (now - state.last_suggest_time) < interval:
        return
    state.last_suggest_time = now

    try:
        suggestion = await _generate_suggestion(state)
    except Exception as e:  # noqa: BLE001 — advisory pass is never fatal
        logger.debug("auto-suggest skipped: %s", e)
        return
    if not suggestion:
        return

    from sidekick.dedup import find_duplicate

    if find_duplicate(suggestion["question"], state.recent_suggestions) is not None:
        return
    state.recent_suggestions.append(suggestion["question"])
    del state.recent_suggestions[:-20]  # keep the last 20

    from sidekick.queue.priority_queue import ActionResult

    result = ActionResult(
        question=suggestion["question"],
        action_type="suggestion",
        answer=suggestion["rationale"] or "Suggested question to ask the client.",
        confidence="medium",
        priority=suggestion["impact"] if suggestion["impact"] in
        ("high", "medium", "low") else "medium",
        rationale=suggestion["rationale"],
    )
    logger.info("Auto-suggest: %s", suggestion["question"][:80])
    notify(result)


async def name_speakers(state: SessionState) -> None:
    """Attribute transcript lines to named participants (Phase 7 / C3 Tier 2).

    Runs at ``stop`` (off the live path). Mutates ``line.speaker`` in place for
    confidently-attributed lines so the transcript, summary, and deliverables
    read with names. Best-effort — any failure leaves the source tags intact.
    """
    if not state.context or not state.config:
        return
    if not getattr(getattr(state.config, "speech", None), "speaker_naming", True):
        return
    lines = getattr(state.context, "full_transcript", None) or []
    if not lines:
        return
    try:
        from sidekick.analyst.speakers import build_roster, name_lines

        roster = build_roster(state.config, state.context)
        if not roster:
            return
        labels = await name_lines(lines, roster)
        for idx, name in labels.items():
            try:
                lines[idx].speaker = name
            except Exception:  # noqa: BLE001 — one bad line must not abort
                logger.debug("Could not set speaker on line %d", idx, exc_info=True)
        if labels:
            logger.info(
                "Speaker naming labelled %d/%d lines", len(labels), len(lines)
            )
    except Exception as e:  # noqa: BLE001 — never break stop
        logger.warning("Speaker naming failed: %s", e)


# ---------------------------------------------------------------------------
# Live audio loop (Phase 3 — extracted from server._run_listen_loop)
# ---------------------------------------------------------------------------

# Loop tuning constants. Kept module-level so tests can monkeypatch them to
# shrink timeouts / error budgets without waiting on real audio hardware.
MAX_CONSECUTIVE_ERRORS = 5
SILENCE_TIMEOUT_SECS = 60
AUDIO_POLL_SECS = 10.0


async def run_listen_loop(state: SessionState, notify: Callable[[object], None]) -> None:
    """Background loop: capture audio → transcribe → batch → classify → queue → execute.

    Thin entry point: initialise the capture stack (Whisper model + WASAPI
    loopback), then run the consume loop. Initialisation failures set
    ``state.last_error`` and return early without raising.
    """
    if not await _initialise_capture(state):
        return
    await _consume_audio(state, notify)


async def _initialise_capture(state: SessionState) -> bool:
    """Load the speech model and open the audio capture device.

    Returns ``True`` when ``state.recogniser`` and ``state.audio_capture`` are
    ready, or ``False`` (with ``state.last_error`` set) if the live
    dependencies are missing or the model fails to load.
    """
    try:
        from sidekick.transcript.audio_capture import AudioCapture
        from sidekick.transcript.speech_recogniser import create_recogniser
    except ImportError as e:
        state.last_error = f"Missing live dependencies: {e}"
        logger.error(state.last_error)
        return False

    loop = asyncio.get_running_loop()

    # 5d: optionally capture the local microphone in addition to system audio
    # so speech can be attributed to "(me)" vs "(remote)". Default off — a
    # single loopback capture tagged "(audio)" (unchanged behaviour).
    capture_mic = bool(
        getattr(getattr(state.config, "speech", None), "capture_microphone", False)
    )
    # Chunk length (Phase 2 / C2): longer chunks give Whisper more context and
    # fewer mid-utterance boundary cuts. Configurable via speech.chunk_seconds.
    chunk_seconds = float(
        getattr(getattr(state.config, "speech", None), "chunk_seconds", 5.0)
    )
    if capture_mic:
        captures = [
            AudioCapture(
                capture_mode="loopback",
                speaker_label="(remote)",
                chunk_duration=chunk_seconds,
            ),
            AudioCapture(
                capture_mode="input",
                speaker_label="(me)",
                chunk_duration=chunk_seconds,
            ),
        ]
    else:
        captures = [
            AudioCapture(
                capture_mode="loopback",
                speaker_label="(audio)",
                chunk_duration=chunk_seconds,
            )
        ]

    # 5c pre-roll: create and BEGIN audio capture *before* loading the (slow)
    # Whisper model, so the WASAPI stream buffers the opening of the meeting
    # while the model loads instead of dropping the first ~minute. The bounded
    # capture queue holds the pre-roll; once the model is ready _consume_audio
    # drains the buffered audio first. Best-effort — if begin() fails (e.g. no
    # device), start() opens capture lazily later as before.
    state.audio_captures = captures
    state.audio_capture = captures[0]  # primary handle (status/stop reference)
    for cap in captures:
        try:
            cap.begin()
            logger.info(
                "Audio capture pre-roll started for %s (buffering during model load).",
                cap.speaker_label,
            )
        except Exception as e:  # noqa: BLE001 — fall back to lazy start in _consume_audio
            logger.debug(
                "Pre-roll capture begin failed for %s (%s); will start lazily.",
                cap.speaker_label,
                e,
            )

    try:
        state.recogniser = await loop.run_in_executor(
            None, create_recogniser, state.config.speech
        )
    except Exception as e:
        state.last_error = f"Failed to load speech model: {e}"
        logger.exception(state.last_error)
        # Stop the pre-roll captures we may have started so the device/thread
        # do not leak when initialisation fails.
        for cap in captures:
            try:
                cap.stop()
            except Exception:  # noqa: BLE001 — cleanup is best-effort
                logger.debug(
                    "Pre-roll capture stop after model-load failure raised",
                    exc_info=True,
                )
        return False

    # Seed the derived Whisper vocabulary prior (Phase 5b) from the same
    # engagement inputs that feed grounding — always available at listen time.
    try:
        from sidekick.transcript.vocabulary import Vocabulary, config_seed_text

        vocab = Vocabulary()
        vocab.seed(config_seed_text(state.config))
        if isinstance(getattr(state, "grounding_cache", None), str):
            vocab.seed(state.grounding_cache)
        # Per-customer glossary outranks derived seed terms (verbatim, weighted).
        vocab.seed_terms(getattr(state.config, "glossary", None) or [])
        state.vocabulary = vocab
        logger.info("Whisper vocabulary prior seeded (%d terms).", len(vocab))
    except Exception as e:  # noqa: BLE001 — prior is best-effort, never fatal
        logger.debug("Vocabulary seed skipped: %s", e)

    devices = await loop.run_in_executor(None, state.audio_capture.list_devices)
    device_names = [d["name"] for d in devices] if devices else ["(none found)"]
    logger.info("Loopback devices: %s", ", ".join(device_names))
    return True


async def _merge_captures(captures: list):
    """Fan-in chunks from multiple audio captures into one async stream (5d).

    Each capture's ``start()`` iterator is drained by its own pump task into a
    shared queue; this generator yields ``(speaker_label, offset, chunk)`` in
    arrival order so dual loopback/microphone capture appears as a single
    interleaved stream to the consumer. Pump tasks are cancelled on exit.
    """
    merged: asyncio.Queue = asyncio.Queue()
    _DONE = object()

    async def _pump(cap):
        label = getattr(cap, "speaker_label", "(audio)")
        try:
            async for chunk in cap.start():
                await merged.put((label, getattr(cap, "last_chunk_offset", 0.0), chunk))
        finally:
            await merged.put(_DONE)

    tasks = [asyncio.create_task(_pump(c)) for c in captures]
    remaining = len(tasks)
    try:
        while remaining > 0:
            item = await merged.get()
            if item is _DONE:
                remaining -= 1
                continue
            yield item
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def _iter_capture_chunks(state: SessionState):
    """Yield ``(speaker, offset, chunk)`` from the session's audio capture(s).

    The default loopback-only path wraps a single capture, preserving the
    existing offset/timing semantics. When microphone capture is enabled (5d)
    it fans in the loopback ``(remote)`` and microphone ``(me)`` streams so each
    transcript line carries the correct speaker tag.
    """
    captures = [
        c
        for c in (getattr(state, "audio_captures", None) or [state.audio_capture])
        if c is not None
    ]
    if len(captures) == 1:
        cap = captures[0]
        label = getattr(cap, "speaker_label", "(audio)")
        # Sample-based audio position (5a): prefer the capture's per-chunk
        # offset (immune to processing backlog); fall back to an index estimate
        # when the capture does not expose one (e.g. test doubles).
        chunk_duration = getattr(cap, "chunk_duration", 5.0)
        index = 0
        async for chunk in cap.start():
            capture_offset = getattr(cap, "last_chunk_offset", None)
            if isinstance(capture_offset, (int, float)):
                offset = float(capture_offset)
            else:
                offset = index * chunk_duration
            index += 1
            yield label, offset, chunk
        return
    async for item in _merge_captures(captures):
        yield item


async def _consume_audio(state: SessionState, notify: Callable[[object], None]) -> None:
    """Consume audio chunks, transcribe, batch, and dispatch for classification.

    Transcription runs on every audio chunk (real-time). Classification is
    batched: transcribed lines accumulate for ``CLASSIFY_INTERVAL`` seconds
    before being sent to the classifier, halving LLM calls while keeping
    transcription responsive. Auto-stops after ``SILENCE_TIMEOUT_SECS`` of no
    recognised speech (audio energy alone does not reset the timer).
    """
    consecutive_errors = 0

    # Classify cadence: config value, with env var override.
    CLASSIFY_INTERVAL = float(
        os.environ.get(
            "SIDEKICK_CLASSIFY_INTERVAL",
            str(state.config.sensitivity.analyst_interval_seconds),
        )
    )

    # Transcript line buffer — accumulates between classifier calls
    pending_lines: list = []
    last_classify_time = time.monotonic()

    # Speech-based auto-stop: tracks last time Whisper returned actual words.
    # Audio energy alone (background hum, HVAC, holding music) does NOT reset
    # this timer — only recognised speech does.
    last_speech_time = time.monotonic()

    # Per-chunk audio position (5a). The capture exposes ``last_chunk_offset``
    # derived from audio actually captured, so transcript timestamps reflect
    # the position within the meeting even when transcription falls behind
    # real time (fixes the wall-clock drift that inflated a 32-min meeting to
    # "56 minutes"). The unified capture stream (5d) supplies the per-chunk
    # offset and speaker tag for both single (loopback) and dual (mic) capture.
    audio_iter = None
    try:
        audio_iter = _iter_capture_chunks(state).__aiter__()
        while True:
            # Wait for next audio chunk.  Use a short poll interval so we
            # can check the speech-based timer even when audio keeps flowing
            # (e.g. background noise with no intelligible speech).
            try:
                speaker, chunk_start_offset, audio_chunk = await asyncio.wait_for(
                    audio_iter.__anext__(), timeout=AUDIO_POLL_SECS
                )
            except asyncio.TimeoutError:
                # No audio chunk arrived — check speech timer
                if time.monotonic() - last_speech_time >= SILENCE_TIMEOUT_SECS:
                    if pending_lines:
                        await classify_and_dispatch(
                            state, pending_lines, consecutive_errors, notify
                        )
                        pending_lines.clear()
                    logger.info(
                        "No speech detected for %ds — auto-stopping.",
                        SILENCE_TIMEOUT_SECS,
                    )
                    state.last_error = (
                        f"Auto-stopped: no speech detected for {SILENCE_TIMEOUT_SECS}s. "
                        f"Call stop for the summary, or listen to start a new session."
                    )
                    break
                continue
            except StopAsyncIteration:
                break

            try:
                # Derived domain prior (5b) — adapts as the call progresses.
                initial_prompt = None
                if state.vocabulary is not None:
                    initial_prompt = state.vocabulary.initial_prompt()
                lines = await state.recogniser.transcribe_chunk(
                    audio_chunk,
                    chunk_start_offset=chunk_start_offset,
                    initial_prompt=initial_prompt,
                    speaker=speaker,
                )
                if lines:
                    last_speech_time = time.monotonic()
                    pending_lines.extend(lines)

                # Check speech-based timeout even when audio is flowing
                # (handles background hum / ambient noise with no words).
                if time.monotonic() - last_speech_time >= SILENCE_TIMEOUT_SECS:
                    if pending_lines:
                        await classify_and_dispatch(
                            state, pending_lines, consecutive_errors, notify
                        )
                        pending_lines.clear()
                    logger.info(
                        "No recognised speech for %ds (audio still active) "
                        "— auto-stopping.",
                        SILENCE_TIMEOUT_SECS,
                    )
                    state.last_error = (
                        f"Auto-stopped: no recognised speech for "
                        f"{SILENCE_TIMEOUT_SECS}s. "
                        f"Call stop for the summary, or listen to start "
                        f"a new session."
                    )
                    break

                # Classify when enough time has passed
                if not lines:
                    continue
                elapsed = time.monotonic() - last_classify_time
                if elapsed >= CLASSIFY_INTERVAL:
                    await classify_and_dispatch(
                        state, pending_lines, consecutive_errors, notify
                    )
                    pending_lines.clear()
                    last_classify_time = time.monotonic()
                    consecutive_errors = 0

            except asyncio.CancelledError:
                raise
            except Exception as chunk_err:
                consecutive_errors += 1
                logger.exception(
                    "Error processing chunk (%d/%d): %s",
                    consecutive_errors, MAX_CONSECUTIVE_ERRORS, chunk_err,
                )
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    raise RuntimeError(
                        f"Too many consecutive errors ({MAX_CONSECUTIVE_ERRORS}): {chunk_err}"
                    ) from chunk_err

    except asyncio.CancelledError:
        logger.info("Listen loop cancelled.")
    except Exception as e:
        state.last_error = f"Listen loop error: {type(e).__name__}: {e}"
        logger.exception("Listen loop error.")
    finally:
        # Clean up resources on ANY exit (auto-stop, cancel, or error).
        # Without this, the WASAPI capture thread and Whisper model leak.
        if audio_iter is not None:
            await audio_iter.aclose()
        # Stop every capture (loopback + optional microphone, 5d).
        for cap in (
            getattr(state, "audio_captures", None) or [state.audio_capture]
        ):
            if cap is not None:
                cap.stop()
        if state.audio_capture is not None:
            logger.info("Audio capture cleaned up after listen loop exit.")
        if state.recogniser is not None:
            state.recogniser.close()
            logger.info("Speech recogniser closed after listen loop exit.")
