"""Sidekick MCP server — real-time meeting co-pilot.

Seven tools, focused on live consulting value:

  listen            — capture system audio and transcribe in real-time
  suggest_questions — synthesise the meeting and recommend what to ask
  add_context       — inject live context (notes, docs, diagrams)
  research          — answer a question instantly
  prototype         — generate working code on the fly
  status            — show what Sidekick has found so far
  stop              — end session and get the summary
"""

import asyncio
import logging
import os
import re
import time

try:
    import numpy as np  # noqa: F401 — must be imported at module level to avoid import-lock deadlock with MCP stdio threads
except ImportError:
    pass  # optional [live] dependency

try:
    import pyaudiowpatch  # noqa: F401 — same deadlock guard for audio device enumeration
except ImportError:
    pass  # optional [live] dependency

from mcp.server.fastmcp import FastMCP

from sidekick.config import load_config
from sidekick.session_state import SessionState
from sidekick.analyst.classifier import TranscriptAnalyst
from sidekick.analyst.context import MeetingContext
from sidekick.queue.priority_queue import PriorityQueue
from sidekick.actions.research import ResearchPipeline
from sidekick.actions.prototype import PrototypePipeline
from sidekick.output.session_log import SessionLog
from sidekick.output import notifier
from sidekick.output.deliverables import build_deliverables, save_deliverables
from sidekick import grounding
from sidekick import engine
from sidekick.prompt_budget import clip

logger = logging.getLogger("sidekick")

# Install source for sidekick-copilot. Distribution is a public Git repo.
# Override with SIDEKICK_REPO_URL.
_REPO_URL = os.environ.get(
    "SIDEKICK_REPO_URL",
    "git+https://github.com/Kenniola/sidekick-copilot.git#subdirectory=sidekick",
)


def _install_hint(extras: str = "live") -> str:
    """Return the `uv tool install` command for the given extras."""
    return f'uv tool install "sidekick-copilot[{extras}] @ {_REPO_URL}" --force'


# ---------------------------------------------------------------------------
# Global state (lives for the lifetime of the MCP server process)
# ---------------------------------------------------------------------------

server = FastMCP("sidekick")

# All mutable session state lives in a single SessionState instance. Functions
# mutate its attributes in place (no ``global`` needed); ``_init_session``
# repopulates the components and ``_state.reset()`` clears per-session counters.
_state = SessionState()

_GROUNDING_CACHE_TTL = 300.0  # 5 minutes

# Bounds for the ``stop`` response so it always renders inline and never spills
# to an unreadable overflow file. The full deliverables pack is saved to disk;
# the chat response carries only a bounded summary + digest.
_MAX_SUMMARY_CHARS = 4000
_MAX_STOP_RESPONSE_CHARS = 8000


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _init_session(config_name: str = "default"):
    """Initialise shared session components."""

    _state.reset()

    _state.config = load_config(config_name)
    _state.context = MeetingContext(customer_name=_state.config.customer)
    # Seed explicit engagement objectives (Phase 1 / A2) so the adjudicator has
    # goals from the first pass; an add_context "goal: …" note or auto-inference
    # fills them in otherwise.
    if getattr(_state.config, "objectives", None):
        _state.context.objectives = list(_state.config.objectives)
        _state.objectives_inferred = True
    _state.analyst = TranscriptAnalyst(config=_state.config, context=_state.context)
    _state.queue = PriorityQueue(config=_state.config)
    _state.session_log = SessionLog(config=_state.config)
    _state.research = ResearchPipeline(config=_state.config)
    _state.prototype = PrototypePipeline(config=_state.config)

    # Register the resolved model chains so every call_llm(tier=…) honours the
    # configured models (YAML + SIDEKICK_MODEL_<TIER> env overrides).
    from sidekick import llm as _llm
    _llm.set_active_models(_state.config.models)


# ---------------------------------------------------------------------------
# Background processing
# ---------------------------------------------------------------------------
#
# The live audio loop (capture → transcribe → batch → classify → dispatch)
# lives in ``sidekick.engine`` (``run_listen_loop``) so it can be tested with
# fake capture/recogniser components. ``listen`` launches it as a task.


def _parse_goal_note(text: str) -> list[str]:
    """Extract objectives from a ``goal:``/``goals:`` context note, else [].

    Splits the note body on newlines and semicolons so a multi-goal note like
    ``"goal: land the S3 PoC; de-risk F64 sizing"`` becomes two objectives.
    """
    stripped = text.strip()
    low = stripped.lower()
    for prefix in ("goals:", "goal:"):
        if low.startswith(prefix):
            body = stripped[len(prefix):]
            parts = re.split(r"[\n;]+", body)
            return [p.strip(" -\u2022\t") for p in parts if p.strip(" -\u2022\t")]
    return []


def _notify(result) -> None:
    """Log a finding: resolve the configured sound and delegate to the notifier."""
    sound = (
        _state.config.notifications.sound
        if _state.config and getattr(_state.config, "notifications", None)
        else "chime"
    )
    notifier.notify(result, sound=sound)


def _get_unseen_findings() -> str:
    """Return new findings since the last tool call, then mark them seen.

    Every tool call prepends this to its output so the user sees background
    research results regardless of which tool they invoke. This solves the
    MCP push limitation — the server can't initiate messages, but it can
    piggyback findings on any response.
    """

    try:
        if not _state.session_log and not _state.context:
            return ""

        parts: list[str] = []

        # New threads
        all_threads = list(_state.context.threads.values()) if _state.context else []
        new_threads = all_threads[_state.last_surface_thread_count:]
        if new_threads:
            for t in new_threads:
                status_icon = "\u23f3" if t.status == "open" else "\u2705"
                parts.append(f"  {status_icon} New thread: **{t.topic}** ({t.status})")
                for q in t.questions[:2]:
                    parts.append(f"     \u2514\u2500 {q}")

        # New research results
        all_outputs = _state.session_log.outputs if _state.session_log else []
        new_outputs = all_outputs[_state.last_surface_output_count:]
        logger.debug(
            "_get_unseen_findings: %d total outputs, surface_count=%d, %d new, %d new threads",
            len(all_outputs), _state.last_surface_output_count, len(new_outputs), len(new_threads),
        )
        if new_outputs:
            for o in new_outputs:
                conf = o.get("confidence", "medium").upper()
                parts.append(f"  \u2705 **{o['question'][:80]}** ({conf})")
                answer = o.get("answer", "")
                if answer:
                    for line in answer.strip().split("\n"):
                        line = line.strip()
                        if line and not line.startswith("Sources"):
                            parts.append(f"     {line[:140]}")
                            break
                sources = o.get("sources", [])
                if not sources and answer:
                    sources = re.findall(r"https?://[^\s\)]+", answer)
                for src in sources[:2]:
                    parts.append(f"     \u2514\u2500 {src}")

        # Update counters
        _state.last_surface_thread_count = len(all_threads)
        _state.last_surface_output_count = len(all_outputs)

        if not parts:
            return ""

        header = f"\U0001f514 **SIDEKICK FOUND** ({len(new_outputs)} new) while you were talking:\n"
        return header + "\n".join(parts) + "\n\n---\n\n"
    except Exception:
        logger.warning("_get_unseen_findings failed", exc_info=True)
        return ""


def _build_grounding_context() -> str:
    """Build grounding context from team standards and engagement artifacts.

    Thin wrapper over :func:`sidekick.grounding.build_grounding_context` that
    supplies the current config and live context. Synchronous file I/O —
    callers should wrap with ``asyncio.to_thread`` to avoid blocking the loop.
    """
    return grounding.build_grounding_context(_state.config, _state.context)


async def _get_grounding_context_async() -> str:
    """Get grounding context, using cache if fresh, else rebuild in a thread."""

    if _state.grounding_cache and (time.time() - _state.grounding_cache_time) < _GROUNDING_CACHE_TTL:
        return _state.grounding_cache

    result = await asyncio.to_thread(_build_grounding_context)
    _state.grounding_cache = result
    _state.grounding_cache_time = time.time()
    return result


# ---------------------------------------------------------------------------
# Tool 1: listen â€” start live audio capture
# ---------------------------------------------------------------------------


@server.tool()
async def listen(config: str = "default", confirmed: bool = False) -> str:
    """Start capturing system audio and transcribing in real-time.

    Captures audio from your default speakers/headset via WASAPI loopback
    and runs the full analysis pipeline. Transcription uses local Whisper
    (configured in the customer YAML config).

    The first call (without confirmed=True) returns a consent notice.
    The agent should present this to the user and only proceed by calling
    listen again with confirmed=True once the user agrees.

    Args:
        config: Customer config name (e.g., 'acme'). Defaults to 'default'.
        confirmed: Set to True after the user consents to audio transcription.
    """

    if _state.listen_task and not _state.listen_task.done():
        return "Already listening. Call stop to end the session first."

    # --- Consent gate ---
    if not confirmed:
        return (
            "\u26a0\ufe0f Audio Transcription Consent\n"
            "\n"
            "Sidekick will capture and transcribe system audio "
            "(Teams, Zoom, etc.).\n"
            "Please confirm that all meeting participants consent "
            "to transcription being captured.\n"
            f"\nConfig: {config} | Reply yes to start."
        )

    # Audio modules are imported at top-of-function to validate they're
    # installed. The heavy C-extension imports (numpy, ctranslate2) must
    # already be in sys.modules from module-level imports above — otherwise
    # Python's import lock deadlocks with MCP's stdio reader thread.
    try:
        from sidekick.transcript.audio_capture import AudioCapture  # noqa: F401
        from sidekick.transcript.speech_recogniser import create_recogniser  # noqa: F401
    except ImportError as e:
        return (
            f"Missing live dependencies: {e}\n"
            f"Reinstall with live extras: "
            f"{_install_hint('live')}"
        )

    _init_session(config)

    # Archive any prior session's live alerts so the feed starts clean for this
    # meeting and the alerts file cannot grow unbounded (Phase 5 / 5.3).
    notifier.rotate_alerts()

    backend_label = "Whisper (local)"

    # Enumerate loopback devices (pyaudiowpatch is pre-imported at module
    # level to avoid import-lock deadlock).
    try:
        from sidekick.transcript.audio_capture import AudioCapture as _AC
        devices = _AC().list_devices()
        # Shorten device names: strip driver details and [Loopback] suffix
        device_names = []
        for d in (devices or []):
            n = d["name"].replace(" [Loopback]", "")
            n = n.split(" (")[0].strip() if " (" in n else n.strip()
            device_names.append(n)
        if not device_names:
            device_names = ["none found"]
    except Exception:
        device_names = ["unavailable"]

    # Start the background loop — model loading and audio capture happen
    # there so this tool returns instantly.
    _state.listen_task = asyncio.create_task(engine.run_listen_loop(_state, _notify))

    # Best-effort pre-warm of the primary LLM connection so the first
    # classifier/research call doesn't pay DNS + TLS setup cost.
    from sidekick import llm as _llm
    asyncio.create_task(_llm.prewarm())

    domains = " \u00b7 ".join(_state.config.domains)
    devices_str = " \u00b7 ".join(device_names)

    return (
        f"{_state.config.customer} \u2014 \U0001f399\ufe0f live ({backend_label})\n"
        f"\n"
        f"Config: {config}.yaml \u00b7 Domains: {domains}\n"
        f"Devices: {devices_str}\n"
        f"\n"
        f"\U0001f7e2 Loading model and starting audio capture...\n"
        f"\n"
        f"`suggest_questions` \u00b7 `add_context` \u00b7 `research` \u00b7 `prototype` \u00b7 `status` \u00b7 `stop`"
    )


# ---------------------------------------------------------------------------
# Tool 2: suggest_questions â€” consultant advisor
# ---------------------------------------------------------------------------


@server.tool()
async def suggest_questions() -> str:
    """Synthesise the meeting and recommend high-impact questions to ask the client.

    Uses deep chain-of-thought reasoning with grounding from team standards
    and past engagement artifacts. Categorised as:
    clarify, probe, challenge, scope, stakeholder, risk, or next_step.
    """
    if not _state.context:
        return "No active session. Start with: listen"

    if len(_state.context.full_transcript) < 3:
        return "Not enough transcript yet \u2014 need a few exchanges first."

    from sidekick.analyst.prompts import CONSULTANT_ADVISOR_PROMPT
    from sidekick.llm import call_llm, parse_llm_json

    # Build a rich context block with key facts and open questions
    key_facts_str = ""
    if _state.context.key_facts:
        key_facts_str = "\nKey facts established:\n" + "\n".join(
            f"  - {f}" for f in _state.context.key_facts[-10:]
        )
    open_q_str = ""
    if _state.context.open_questions:
        open_q_str = "\nOpen questions (unanswered):\n" + "\n".join(
            f"  - {q['question']}" for q in _state.context.open_questions[-5:]
        )

    context_block = (
        f"Customer: {_state.config.customer}\n"
        f"Domains: {', '.join(_state.config.domains)}\n"
        f"Elapsed: {_state.context.elapsed_minutes:.0f} minutes\n"
        f"Phase: {_state.context.current_phase}\n"
        f"Transcript lines: {len(_state.context.full_transcript)}"
        f"{key_facts_str}{open_q_str}"
    )

    recent = _state.context.full_transcript[-50:]
    transcript_block = "\n".join(
        f"[{getattr(line, 'start', '?')}] {getattr(line, 'speaker', '?')}: {getattr(line, 'text', str(line))}"
        for line in recent
    )
    # Token-budget each block so a long meeting can't blow the deep model's
    # context window or inflate latency. Keep the most recent transcript tail;
    # keep the head of the (less time-sensitive) threads/research/grounding.
    transcript_block = clip(transcript_block, 6000, keep="tail")

    threads_block = _state.context.format_threads()
    threads_block = clip(threads_block, 2000, keep="head")

    research_block = "(none yet)"
    if _state.session_log and _state.session_log.outputs:
        research_parts = []
        for o in _state.session_log.outputs[-10:]:
            research_parts.append(f"[{o['action_type']}] {o['question']}: {o['answer'][:150]}")
        research_block = "\n".join(research_parts)
    research_block = clip(research_block, 2500, keep="head")

    grounding_block = await _get_grounding_context_async()
    grounding_block = clip(grounding_block, 4000, keep="head")

    prompt = CONSULTANT_ADVISOR_PROMPT.format(
        context_block=context_block,
        transcript_block=transcript_block,
        threads_block=threads_block,
        research_block=research_block,
        grounding_block=grounding_block,
        phase=_state.context.current_phase,
    )

    advisor_system_prompt = (
        "You are a senior consulting advisor with deep expertise in "
        f"{', '.join(_state.config.domains)}. "
        "Think carefully through the reasoning chain before generating "
        "questions. Return JSON only."
    )

    async def _suggest(tier: str, attempt_timeout: float, overall_timeout: float) -> str:
        # ``timeout`` on call_llm is the *per-HTTP-attempt* budget; the
        # ``asyncio.wait_for`` caps the whole call (incl. retries/backoff) with
        # an overall wall-clock budget so a throttled upstream can't stall.
        return await asyncio.wait_for(
            call_llm(
                system_prompt=advisor_system_prompt,
                user_prompt=prompt,
                json_output=True,
                timeout=attempt_timeout,
                tier=tier,
            ),
            timeout=overall_timeout,
        )

    try:
        # Prefer the deep (Opus) advisor for the richest reasoning, but cap it
        # tightly. When the background research loop is hammering the same
        # Copilot endpoint, the deep model gets throttled (429) and would
        # otherwise time out with nothing to show. On timeout, fall back to the
        # fast tier (gpt-4o-mini) — a different model deployment that is far
        # less likely to be rate-limited — so the tool always returns usable
        # suggestions, degrading quality gracefully instead of failing.
        try:
            response_text = await _suggest("deep", attempt_timeout=30.0, overall_timeout=40.0)
        except asyncio.TimeoutError:
            logger.warning(
                "suggest_questions deep tier timed out; falling back to fast tier"
            )
            response_text = await _suggest("fast", attempt_timeout=20.0, overall_timeout=25.0)

        data = parse_llm_json(response_text)
    except asyncio.TimeoutError:
        logger.warning("suggest_questions timed out (deep + fast fallback)")
        return (
            "Suggestions timed out (the model is busy — likely throttled while "
            "research is also running). Try `suggest_questions` again in a moment."
        )
    except Exception as e:
        logger.exception("suggest_questions LLM call failed")
        return f"Failed to generate suggestions: {type(e).__name__}: {e}"

    parts = []

    synthesis = data.get("synthesis", "")
    if synthesis:
        parts.append(f"\U0001f4cb {synthesis}")
        parts.append("")

    # Show corrections first — these are urgent
    corrections = data.get("corrections", [])
    if corrections:
        parts.append("\u26a0\ufe0f CORRECTIONS:")
        for corr in corrections:
            parts.append(f"  \U0001f534 {corr}")
        parts.append("")

    questions = data.get("questions", [])
    if questions:
        parts.append("SUGGESTED QUESTIONS:")
        for i, q in enumerate(questions, 1):
            impact = q.get("impact", "medium")
            icon = "\U0001f534" if impact == "high" else "\U0001f7e1"
            category = q.get("category", "?")
            parts.append(f"  {icon} {i}. [{category}] {q.get('question', '?')}")
            rationale = q.get("rationale", "")
            if rationale:
                parts.append(f"     \u21b3 {rationale}")
            builds_on = q.get("builds_on", "")
            if builds_on:
                parts.append(f"     \U0001f4ac Based on: \"{builds_on}\"")
            parts.append("")
    else:
        parts.append("No questions to suggest at this point.")

    observations = data.get("observations", [])
    if observations:
        parts.append("OBSERVATIONS:")
        for obs in observations:
            parts.append(f"  \U0001f441 {obs}")

    return _get_unseen_findings() + "\n".join(parts)


# ---------------------------------------------------------------------------
# Tool 3: add_context — inject live context
# ---------------------------------------------------------------------------


@server.tool()
async def add_context(
    content: str = "",
    file_path: str = "",
    image_path: str = "",
) -> str:
    """Inject live context into the meeting session (notes, docs, diagrams).

    Use during a call to feed Sidekick additional information it can't hear,
    such as architecture diagrams shown on screen, document links, or
    decisions made in chat.

    Args:
        content: Free-text context (paste notes, architecture decisions, etc.)
        file_path: Path to a document to ingest (md, txt, json, yaml)
        image_path: Path to a screenshot or diagram image (png, jpg) —
                    processed via vision LLM to extract a text description.
    """
    if not _state.context:
        return "No active session. Start with: listen"

    if not content and not file_path and not image_path:
        return (
            "Provide at least one input:\n"
            "  content=\"your notes here\"\n"
            "  file_path=\"path/to/doc.md\"\n"
            "  image_path=\"path/to/diagram.png\""
        )

    added: list[str] = []

    # 1. Free-text content
    if content:
        _state.context.context_documents.append(content)
        added.append(f"Text note ({len(content)} chars)")
        # A "goal:"-prefixed note sets the engagement objectives the Phase 1
        # adjudicator scores relevance against (highest-priority source).
        goals = _parse_goal_note(content)
        if goals:
            _state.context.objectives = goals
            _state.objectives_inferred = True
            added.append(f"Objectives set ({len(goals)})")

    # 2. File content
    if file_path:
        from pathlib import Path as _Path
        fp = _Path(file_path)
        if not fp.exists():
            return f"File not found: {file_path}"
        allowed = {".md", ".txt", ".json", ".yaml", ".yml", ".csv", ".sql"}
        if fp.suffix.lower() not in allowed:
            return f"Unsupported file type: {fp.suffix}. Supported: {', '.join(sorted(allowed))}"
        try:
            text = fp.read_text(encoding="utf-8")
            # Cap at 4000 chars to stay within context limits
            if len(text) > 4000:
                text = text[:4000] + "\n... (truncated)"
            _state.context.context_documents.append(f"--- {fp.name} ---\n{text}")
            added.append(f"File: {fp.name} ({len(text)} chars)")
        except Exception as e:
            return f"Error reading {fp.name}: {e}"

    # 3. Image — extract description via vision LLM
    if image_path:
        import base64
        from pathlib import Path as _Path
        from sidekick.llm import call_llm_vision

        ip = _Path(image_path)
        if not ip.exists():
            return f"Image not found: {image_path}"
        allowed_img = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
        if ip.suffix.lower() not in allowed_img:
            return f"Unsupported image type: {ip.suffix}. Supported: {', '.join(sorted(allowed_img))}"
        try:
            image_bytes = ip.read_bytes()
            # Cap at 10MB
            if len(image_bytes) > 10 * 1024 * 1024:
                return f"Image too large ({len(image_bytes) // 1024 // 1024}MB). Max 10MB."
            image_b64 = base64.b64encode(image_bytes).decode()

            description = await call_llm_vision(
                system_prompt=(
                    "You are a technical diagram analyst. Describe this image "
                    "precisely, extracting: components, data flows, technologies, "
                    "integration points, labels, and any text visible. "
                    "Be specific — mention product names, service names, and "
                    "connection types exactly as shown."
                ),
                user_prompt=(
                    "Describe this architecture diagram or screenshot from a "
                    "customer meeting. Extract all technical details."
                ),
                image_b64=image_b64,
                timeout=30.0,
            )

            _state.context.context_documents.append(
                f"--- Image: {ip.name} ---\n{description}"
            )
            added.append(f"Image: {ip.name} (extracted {len(description)} chars)")
        except Exception as e:
            return f"Error processing image {ip.name}: {e}"

    summary = ", ".join(added)
    total = len(_state.context.context_documents)
    return (
        _get_unseen_findings()
        + f"Context added: {summary}\n"
        + f"Total context documents: {total}"
    )


# ---------------------------------------------------------------------------
# Tool 4: research — instant answers
# ---------------------------------------------------------------------------


@server.tool()
async def research(question: str, depth: str = "medium") -> str:
    """Research a question instantly. Searches MS Learn and workspace docs.

    Args:
        question: The question to research.
        depth: 'quick' (fast lookup), 'medium' (multi-source), 'deep' (thorough).
    """
    pipeline = _state.research or ResearchPipeline()

    result = await pipeline.execute_direct(
        question=question,
        depth=depth,
        context=_state.context,
        tier="deep" if depth == "deep" else "standard",
        domains=_state.config.domains if _state.config else None,
    )

    return _get_unseen_findings() + result.format()

# ---------------------------------------------------------------------------
# Tool 5: prototype â€” generate code on the fly
# ---------------------------------------------------------------------------


@server.tool()
async def prototype(
    description: str,
    type: str = "notebook",
    columns: str = "",
) -> str:
    """Generate working code on the fly during the meeting.

    Args:
        description: What the prototype should do.
        type: 'notebook' (PySpark), 'sql' (T-SQL), 'dax' (measures), 'pipeline'.
        columns: Optional comma-separated column list.
    """
    config = _state.config or load_config("default")
    pipeline = _state.prototype or PrototypePipeline(config=config)

    result = await pipeline.execute_direct(
        description=description,
        prototype_type=type,
        columns=columns,
        context=_state.context,
    )
    return _get_unseen_findings() + result.format()


# ---------------------------------------------------------------------------
# Tool 6: status â€” what has Sidekick found so far?
# ---------------------------------------------------------------------------


@server.tool()
async def status() -> str:
    """Show what Sidekick has found — new threads, research results, and errors.

    Call this anytime to see incremental updates since your last check.
    Also shows the full session overview.
    """

    if not _state.context:
        return "No active session. Start with: listen"

    # Session header
    if _state.audio_capture and _state.audio_capture.is_capturing:
        mode_label = "🎙️ live (Whisper)"
    else:
        mode_label = "session active"

    parts = [
        f"{_state.config.customer} — {mode_label} — {_state.context.elapsed_minutes:.0f} min — {len(_state.context.full_transcript)} lines",
    ]

    # Surface errors immediately
    if _state.last_error:
        parts.append(f"\n⚠️ ERROR: {_state.last_error}")
        parts.append("Try: stop, then listen again.\n")

    # In-progress queue items
    if _state.queue:
        in_progress = _state.queue.get_in_progress()
        if in_progress:
            parts.append("")
            parts.append("RESEARCHING:")
            for item in in_progress:
                parts.append(f"  ⏳ {item.item.question[:80]}")

    # Full thread summary
    all_threads = list(_state.context.threads.values()) if _state.context else []
    if all_threads:
        parts.append("")
        parts.append("ALL THREADS:")
        for t in all_threads:
            status_icon = "⏳" if t.status == "open" else "✅" if t.status == "resolved" else "🚫"
            parts.append(f"  {status_icon} {t.topic} ({t.status})")

    # Output count
    total_outputs = len(_state.session_log.outputs) if _state.session_log else 0
    in_progress_count = len(_state.queue.get_in_progress()) if _state.queue else 0
    if total_outputs or in_progress_count:
        phase_suffix = f" Still on the {_state.context.current_phase} topic." if hasattr(_state.context, 'current_phase') else ""
        parts.append(f"\n{total_outputs} research completed, {in_progress_count} in progress.{phase_suffix}")

    if len(parts) == 1 and not _state.last_error:
        parts.append("Listening... no threads detected yet.")

    # Prepend any new findings since last tool call
    return _get_unseen_findings() + "\n".join(parts)


# ---------------------------------------------------------------------------
# Tool 7: stop â€” end session with summary
# ---------------------------------------------------------------------------


@server.tool()
async def stop(deliverables: bool = True) -> str:
    """End the session and get a full meeting summary.

    Stops audio capture, generates a structured summary of all threads,
    research results, and action items, and saves the session log.

    Args:
        deliverables: When True (default), also generate a customer-ready
            deliverables pack (draft follow-up email, action-item table, and
            a "couldn't answer live" research batch) and save it alongside the
            summary. Set False to skip the extra LLM call.
    """

    # Stop audio capture FIRST — this signals the capture thread to exit
    # cleanly before we cancel the listen task. Cancelling the task while
    # the capture thread is still writing to PyAudio can cause a C-level crash.
    # Note: engine.run_listen_loop's finally block may have already called
    # stop() and close() — these methods are safe to call multiple times.
    if _state.audio_capture:
        _state.audio_capture.stop()

    if _state.listen_task and not _state.listen_task.done():
        # Give the loop a few seconds to exit naturally via the sentinel
        try:
            await asyncio.wait_for(_state.listen_task, timeout=5.0)
        except asyncio.TimeoutError:
            _state.listen_task.cancel()
            try:
                await _state.listen_task
            except asyncio.CancelledError:
                pass
    elif _state.listen_task and _state.listen_task.done():
        # Task already finished (e.g. auto-stop) — retrieve any exception
        # so it doesn't go unhandled.
        if not _state.listen_task.cancelled():
            exc = _state.listen_task.exception()
            if exc:
                logger.warning("Listen task had unhandled exception: %s", exc)

    if _state.recogniser:
        _state.recogniser.close()

    summary = "No active session."
    saved_files: list[str] = []
    deliverables_digest = ""
    deliverables_path = ""
    if _state.session_log and _state.context:
        # Attribute transcript lines to named participants first (Phase 7) so
        # the summary, transcript, and deliverables all read with names.
        await engine.name_speakers(_state)
        summary = _state.session_log.generate_summary(_state.context)
        path = _state.session_log.save_to_disk()
        if path:
            saved_files.append(str(path))
        # Export transcript and markdown summary
        tp = _state.session_log.save_transcript(_state.context)
        if tp:
            saved_files.append(str(tp))
        mp = _state.session_log.save_markdown_summary(_state.context)
        if mp:
            saved_files.append(str(mp))

        # Post-call deliverables pack (email draft + action items + follow-up
        # research batch). Wrapped so a failure never breaks the summary.
        if deliverables and _state.config:
            try:
                pack = await build_deliverables(
                    _state.session_log, _state.context, _state.config
                )
                # Always persist the FULL pack on an explicit stop (force) so
                # there is a readable file to open even when auto_save is off.
                # The chat response inlines only a bounded digest — the full
                # email can run to several KB, which overflows the tool-result
                # buffer and spills to a file the agent cannot read.
                dp = save_deliverables(
                    pack.full_markdown(), _state.config, force=True
                )
                if dp:
                    deliverables_path = str(dp)
                    saved_files.append(deliverables_path)
                    # Toast the saved file so it isn't lost in a long stop
                    # response — the extension offers an "Open File" action.
                    sound = (
                        _state.config.notifications.sound
                        if getattr(_state.config, "notifications", None)
                        else "chime"
                    )
                    notifier.write_deliverables_alert(deliverables_path, sound=sound)
                deliverables_digest = pack.inline_digest(deliverables_path)
            except Exception:
                logger.exception("Deliverables generation failed")
                deliverables_digest = ""

    # Lead with a prominent banner so the deliverables file is never lost even
    # if the chat surface truncates the (long) inline content below.
    banner = ""
    if deliverables_path:
        banner = (
            f"\U0001f4e6 **Post-call deliverables saved:** `{deliverables_path}`\n"
            f"(Draft follow-up email + action-item table + follow-up research "
            f"batch \u2014 open that file for the full pack; a preview is below.)"
            f"\n\n---\n\n"
        )

    # Bound the session summary so a long transcript/thread list can't push the
    # whole stop response past the chat inline limit.
    summary = clip(summary, _MAX_SUMMARY_CHARS, keep="head")

    if saved_files:
        summary += "\n\n**Saved files:**\n" + "\n".join(
            f"- {f}" for f in saved_files
        )

    if deliverables_digest:
        summary += "\n\n" + deliverables_digest

    # Reset state
    _state.listen_task = None
    _state.audio_capture = None
    _state.recogniser = None
    _state.last_error = None

    # Final belt-and-braces bound: the banner (with the saved path) sits at the
    # front so it always survives even if the tail is clipped.
    return clip(
        _get_unseen_findings() + banner + summary,
        _MAX_STOP_RESPONSE_CHARS,
        keep="head",
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


async def main():
    """Run the Sidekick MCP server over stdio."""
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    await server.run_stdio_async()


if __name__ == "__main__":
    asyncio.run(main())
