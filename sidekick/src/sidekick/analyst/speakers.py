"""LLM speaker-naming (Phase 7 / C3 Tier 2).

Single-channel capture tags every line ``(audio)`` (or ``(remote)``/``(me)``
under dual capture), so a multi-person meeting collapses to one speaker. This
module runs a best-effort, *post-call* pass that attributes transcript lines to
named participants using conversational cues — self-introductions, direct
address, and turn-taking — from the known roster.

It is deliberately partial and conservative: only lines the model is confident
about are labelled; everything else keeps its source tag. Every failure
degrades to "no labels" so the transcript is never lost. Runs off the live
path (at ``stop``), so its latency and cost never affect the meeting.
"""

from __future__ import annotations

import logging
from typing import Awaitable, Callable

from sidekick.llm import call_llm, parse_llm_json
from sidekick.prompt_budget import clip

logger = logging.getLogger(__name__)

LLMFn = Callable[..., Awaitable[str]]

# Per line, and per batch, so a long meeting stays bounded in cost.
_LINE_CHARS = 240
_BATCH_SIZE = 150
_MAX_BATCHES = 8

SPEAKER_SYSTEM_PROMPT = """\
You attribute meeting-transcript lines to named speakers. You are given the \
KNOWN PARTICIPANTS (with roles) and a numbered transcript. Use explicit cues \
only — self-introductions ("I'm Sam"), direct address ("Alex, what do you \
think?"), names mentioned, and turn-taking — to decide who spoke each line.

Rules:
- Only assign a name when you are reasonably confident. OMIT lines you are \
unsure about — do NOT guess.
- Prefer names from the KNOWN PARTICIPANTS list; you may add a name only when \
someone is clearly introduced by name in the transcript.
- Keep names exactly as given in the roster.

Return JSON ONLY, no prose:
{"labels": {"<line index>": "<speaker name>"}}"""


def build_roster(config, context) -> dict[str, str]:
    """Build a ``{name: role}`` roster from config + detected participants.

    Roles are ``"consultant"`` or ``"client"``. Config names take precedence;
    participants identified during the meeting fill in the rest.
    """
    roster: dict[str, str] = {}
    for n in getattr(config, "consultant_names", None) or []:
        name = str(n).strip()
        if name:
            roster[name] = "consultant"
    for n in getattr(config, "client_names", None) or []:
        name = str(n).strip()
        if name:
            roster.setdefault(name, "client")
    for n, role in (getattr(context, "participants", None) or {}).items():
        name = str(n).strip()
        if name:
            roster.setdefault(name, role or "client")
    return roster


async def name_lines(
    lines,
    roster: dict[str, str],
    *,
    llm_fn: LLMFn = call_llm,
) -> dict[int, str]:
    """Return ``{line_index: speaker_name}`` for confidently-attributed lines.

    Batches the transcript so long meetings stay bounded. Returns ``{}`` when
    there is nothing to work with or on any failure.
    """
    if not lines or not roster:
        return {}

    roster_str = ", ".join(f"{n} ({r})" for n, r in roster.items())
    labels: dict[int, str] = {}
    valid_names = set(roster)

    for b, start in enumerate(range(0, len(lines), _BATCH_SIZE)):
        if b >= _MAX_BATCHES:
            break
        batch = lines[start:start + _BATCH_SIZE]
        numbered = "\n".join(
            f"{i}: {clip(getattr(ln, 'text', '') or '', _LINE_CHARS, keep='head')}"
            for i, ln in enumerate(batch)
        )
        try:
            raw = await llm_fn(
                system_prompt=SPEAKER_SYSTEM_PROMPT,
                user_prompt=(
                    f"KNOWN PARTICIPANTS: {roster_str}\n\n"
                    f"TRANSCRIPT (index: text):\n{numbered}"
                ),
                json_output=True,
                tier="fast",
                timeout=30.0,
            )
            data = parse_llm_json(raw)
        except Exception as e:  # noqa: BLE001 — never lose the transcript
            logger.warning("Speaker naming batch %d failed: %s", b, e)
            continue

        for k, v in (data.get("labels") or {}).items():
            try:
                idx = int(k)
            except (TypeError, ValueError):
                continue
            name = str(v).strip()
            # Accept roster names, or a newly-introduced name the model returns.
            if not name or not (0 <= idx < len(batch)):
                continue
            if name in valid_names or name not in ("", "?", "unknown"):
                labels[start + idx] = name

    return labels
