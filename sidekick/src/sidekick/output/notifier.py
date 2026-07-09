"""Finding notifier — audible alert, MCP-channel log line, and audit JSONL.

Extracted from ``server._notify`` (Phase 2b) so the side-effecting bits are
testable in isolation. The server resolves the configured sound and delegates
here.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("sidekick")

# Findings audit trail. Computed lazily (see ``_default_alerts_dir``) so tests
# can target a temp directory without import-time HOME binding.
_ALERTS_SUBPATH = (".sidekick", "live")

# Max length of the one-line answer carried into the toast. Long enough to be
# useful, short enough that the OS notification doesn't truncate it.
_ANSWER_MAX_CHARS = 160

_URL_RE = re.compile(r"https?://\S+")


def _default_alerts_dir() -> Path:
    """Return ~/.sidekick/live (resolved fresh so HOME changes are honoured)."""
    return Path.home().joinpath(*_ALERTS_SUBPATH)


def _stable_id(action_type: str, question: str) -> str:
    """Feed key that stays stable across enrichment (Phase 5 / 5.4).

    Re-research wraps the question as ``[ENRICHED] <q> (previous answer: …)``;
    stripping that wrapper keeps the ``id`` equal to the original so the feed
    supersedes the row in place instead of stacking duplicates.
    """
    q = (question or "").strip()
    if q.startswith("[ENRICHED] "):
        q = q[len("[ENRICHED] "):]
    idx = q.rfind(" (previous answer:")
    if idx != -1:
        q = q[:idx]
    return f"{action_type}:{q.strip()[:40]}"


def rotate_alerts(alerts_dir: Path | None = None) -> None:
    """Archive the current alerts.jsonl at session start (Phase 5 / 5.3).

    Keeps the live feed scoped to the current meeting and stops the file
    growing without bound. The prior file is moved to ``live/archive/``.
    """
    target_dir = alerts_dir if alerts_dir is not None else _default_alerts_dir()
    alerts_file = target_dir / "alerts.jsonl"
    try:
        if alerts_file.exists() and alerts_file.stat().st_size > 0:
            archive_dir = target_dir / "archive"
            archive_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            alerts_file.replace(archive_dir / f"alerts_{ts}.jsonl")
    except Exception:
        logger.debug("Failed to rotate alerts file", exc_info=True)


def _one_line_answer(result, limit: int = _ANSWER_MAX_CHARS) -> str:
    """Derive a single-line answer for the toast from ``result.answer``.

    Takes the text before any ``Sources:`` section, collapses it to the first
    non-empty line, trims a trailing URL fragment, and clips to *limit* chars
    (on a word boundary, with an ellipsis). Returns ``""`` when there is no
    answer body (e.g. a prototype result), so callers can fall back to the
    question summary.
    """
    answer = (getattr(result, "answer", "") or "").strip()
    if not answer:
        return ""

    # Drop the trailing "Sources:" / "Sources [HIGH]:" block — toast shows the
    # source link separately.
    body = re.split(r"\n\s*Sources?\b", answer, maxsplit=1, flags=re.IGNORECASE)[0]

    # First non-empty line is the lead answer (research prompts lead with it).
    first_line = next((ln.strip() for ln in body.splitlines() if ln.strip()), "")
    if not first_line:
        return ""

    if len(first_line) <= limit:
        return first_line

    clipped = first_line[:limit].rsplit(" ", 1)[0].rstrip()
    return f"{clipped or first_line[:limit].rstrip()}\u2026"


def _first_source_url(result) -> str:
    """Return the first http(s) URL for the finding, else ``""``.

    Prefers the structured ``result.sources`` list (formatted as
    ``"<title> \u2014 <URL>"``). Many results built by the direct
    ``research``/``prototype`` paths leave that list empty and carry their
    citations inline in the ``answer`` text's ``Sources:`` block instead, so
    we fall back to scanning the answer. Without this fallback the toast's
    "Open Source" button never appears even when the answer cites URLs.
    """
    for src in getattr(result, "sources", []) or []:
        match = _URL_RE.search(str(src))
        if match:
            return match.group(0).rstrip(").,;")
    # Fallback: first URL anywhere in the answer body (mirrors the auto-surface
    # preamble, which already regex-extracts sources from the answer text).
    match = _URL_RE.search(getattr(result, "answer", "") or "")
    if match:
        return match.group(0).rstrip(").,;")
    return ""


def play_sound(sound: str = "chime") -> None:
    """Play the configured notification sound. Windows-only; no-op elsewhere.

    ``sound`` accepts: ``silent`` (no sound), ``chime`` (default ``MB_OK``),
    ``asterisk``, ``exclamation``, or ``beep`` (legacy 800 Hz / 200 ms tone).
    All failures (no audio device, non-Windows) are swallowed.
    """
    try:
        if sys.platform != "win32":
            return
        import winsound

        if sound == "silent":
            return
        if sound == "beep":
            # Legacy raw tone — 800 Hz, 200 ms (softer than the old 1 kHz/300 ms).
            winsound.Beep(800, 200)
            return
        # MessageBeep variants respect the Windows Notification volume slider.
        style_map = {
            "chime": winsound.MB_OK,
            "asterisk": winsound.MB_ICONASTERISK,
            "exclamation": winsound.MB_ICONEXCLAMATION,
        }
        winsound.MessageBeep(style_map.get(sound, winsound.MB_OK))
    except Exception:
        pass  # Not on Windows or no sound device — skip silently.


def write_alert(result, alerts_dir: Path | None = None) -> None:
    """Append a finding to ``<alerts_dir>/alerts.jsonl`` (audit trail)."""
    target_dir = alerts_dir if alerts_dir is not None else _default_alerts_dir()
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        alert = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": result.action_type,
            "summary": result.question[:120],
            "answer": _one_line_answer(result),
            "answer_full": (getattr(result, "answer", "") or "").strip()[:1500],
            "source": _first_source_url(result),
            "confidence": getattr(result, "confidence", "medium"),
            "priority": getattr(result, "priority", "medium"),
            # Stable key for feed supersede/dedup (survives enrichment).
            "id": _stable_id(
                result.action_type, getattr(result, "question", "") or ""
            ),
            "rationale": getattr(result, "rationale", "") or "",
            "thread_id": getattr(result, "thread_id", "") or "",
        }
        with open(target_dir / "alerts.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(alert) + "\n")
    except Exception:
        logger.debug("Failed to write alert file", exc_info=True)


def write_deliverables_alert(
    path, sound: str = "chime", alerts_dir: Path | None = None
) -> None:
    """Append a 'deliverables ready' alert so the notify extension can toast it.

    Post-call deliverables are saved to disk on ``stop``, but the file is easy
    to miss in a long ``stop`` response. This writes a high-priority alert
    carrying the file ``path`` so the ``sidekick-notify`` extension can raise a
    toast with an "Open File" action, closing the discoverability gap.
    """
    target_dir = alerts_dir if alerts_dir is not None else _default_alerts_dir()
    play_sound(sound)
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        alert = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "deliverables",
            "summary": "Post-call deliverables ready",
            "answer": f"Saved to {Path(path).name}",
            "answer_full": "",
            "source": "",
            "file": str(path),
            "confidence": "high",
            "priority": "high",
            "id": f"deliverables:{Path(path).name}",
            "rationale": "",
            "thread_id": "",
        }
        with open(target_dir / "alerts.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(alert) + "\n")
    except Exception:
        logger.debug("Failed to write deliverables alert", exc_info=True)


def notify(result, sound: str = "chime", alerts_dir: Path | None = None) -> None:
    """Log a finding: audible alert + MCP-channel log line + audit JSONL.

    The user sees findings via the auto-surface preamble on their next tool
    call, so the audible alert is intentionally subtle.
    """
    play_sound(sound)

    icon = {"research": "\U0001f50d", "prototype": "\U0001f6e0"}.get(
        result.action_type, "\U0001f4cb"
    )
    logger.info(
        "%s FINDING [%s]: %s", icon, result.action_type, result.question[:80]
    )

    write_alert(result, alerts_dir=alerts_dir)
