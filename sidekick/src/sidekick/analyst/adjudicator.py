"""Relevance adjudicator (Phase 1 / A1).

Second stage of the two-stage accuracy pipeline. The fast per-chunk classifier
is a high-recall *candidate detector*; this deep-tier pass decides which few
candidates are genuinely worth surfacing to the consultant right now, judged
against the engagement's objectives and the team's standards. Each surfaced
item carries a one-line rationale.

The LLM call is injected (``llm_fn``) so tests run offline, and every failure
degrades gracefully to a deterministic threshold-and-cap filter, so the live
loop is never blocked by an adjudicator error.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Awaitable, Callable

from sidekick.analyst.classifier import ActionItem
from sidekick.dedup import find_duplicate
from sidekick.llm import call_llm, parse_llm_json
from sidekick.prompt_budget import clip

logger = logging.getLogger(__name__)

LLMFn = Callable[..., Awaitable[str]]

ADJUDICATOR_SYSTEM_PROMPT = """\
You are a senior consulting advisor acting as a RELEVANCE ADJUDICATOR for a \
live meeting co-pilot. A fast first-pass classifier has proposed CANDIDATE \
questions/topics from the transcript. Your job is to select ONLY the few that \
are genuinely worth surfacing to the consultant RIGHT NOW, judged against the \
engagement's OBJECTIVES and the team's standards.

Reason through each candidate before deciding:
1. RELEVANCE — does it materially advance a stated OBJECTIVE or de-risk the \
engagement? Discard generically-interesting-but-off-goal items.
2. REDUNDANCY — merge near-duplicates and refine verbose candidates into one \
crisp, well-formed question. Preserve the client's exact system/product/team \
names; never generalise specifics away.
2b. SESSION MEMORY — you are given the questions ALREADY SURFACED earlier this \
session. Do NOT surface any candidate that repeats or merely rephrases one of \
them, even if the wording, scope, or emphasis differs slightly (e.g. "is there \
an advantage to X" vs "what advantage does X give this project" are the SAME \
question). The consultant has already seen that point — surface only a \
genuinely NEW angle that adds information beyond what was already asked.
3. VALUE — prefer questions that uncover the real problem, unstated \
assumptions, risks, or that the consultant must answer to move forward.
4. PRECISION over recall — it is better to surface 1 excellent item than 3 \
mediocre ones. Surfacing noise erodes the consultant's trust.

For every item you surface, give a one-line RATIONALE tying it to a specific \
objective or risk, and a priority_score in [0, 1].

Return JSON ONLY, no prose:
{
  "surfaced": [
    {
      "index": <int index of the primary candidate>,
      "question": "<refined, specific question>",
      "priority_score": 0.0,
      "rationale": "<why this matters, tied to an objective/risk>",
      "merged_indices": [<other candidate indices merged in>]
    }
  ]
}
Surface AT MOST the requested number of items. Omit anything below the \
requested threshold. If nothing clears the bar, return an empty list."""


def _fallback_filter(
    candidates: list[ActionItem], threshold: float, max_surfaced: int
) -> list[ActionItem]:
    """Deterministic threshold + cap used when the LLM pass is unavailable."""
    ranked = sorted(
        candidates, key=lambda i: getattr(i, "priority_score", 0.0), reverse=True
    )
    kept = [i for i in ranked if getattr(i, "priority_score", 0.0) >= threshold]
    return kept[:max_surfaced]


# Lexical backstop for the LLM's semantic session-memory. Set below the strict
# 0.8 dedup default so obvious rephrasings of an already-surfaced question are
# still caught; the LLM handles the harder semantic cases.
_SURFACED_DUP_THRESHOLD = 0.72


def _drop_already_surfaced(
    surfaced: list[ActionItem], already_surfaced: list[str] | None
) -> list[ActionItem]:
    """Drop items that lexically repeat a question surfaced earlier this session."""
    if not already_surfaced:
        return surfaced
    out: list[ActionItem] = []
    for item in surfaced:
        if find_duplicate(
            item.question, already_surfaced, threshold=_SURFACED_DUP_THRESHOLD
        ) is not None:
            logger.info(
                "Suppressed already-surfaced (lexical): %s", item.question[:60]
            )
            continue
        out.append(item)
    return out


def _candidate_block(candidates: list[ActionItem]) -> str:
    """Enumerate candidates for the prompt (index → question/type/score)."""
    lines = []
    for i, c in enumerate(candidates):
        lines.append(
            f"[{i}] ({c.type}, score={getattr(c, 'priority_score', 0.0):.2f}) "
            f"{c.question}"
        )
    return "\n".join(lines)


def _build_user_prompt(
    candidates: list[ActionItem],
    context,
    config,
    grounding: str,
    objectives: list[str],
    *,
    threshold: float,
    max_surfaced: int,
    already_surfaced: list[str] | None = None,
) -> str:
    """Assemble the token-budgeted adjudication prompt."""
    customer = getattr(config, "customer", "the customer")
    objectives_block = (
        "\n".join(f"- {o}" for o in objectives)
        if objectives
        else "(none stated — infer from context; be conservative)"
    )
    threads_block = (
        context.format_threads() if hasattr(context, "format_threads") else "(none)"
    )
    buffer_block = ""
    if hasattr(context, "format_recent_buffer"):
        buffer_block = context.format_recent_buffer()
    grounding_block = clip(grounding or "(none)", 4000, keep="head")
    surfaced_block = (
        "\n".join(f"- {q}" for q in (already_surfaced or [])[-40:])
        if already_surfaced
        else "(nothing yet)"
    )

    return (
        f"Customer: {customer}\n\n"
        f"ENGAGEMENT OBJECTIVES:\n{objectives_block}\n\n"
        f"ALREADY SURFACED THIS SESSION (do NOT repeat or rephrase these):\n"
        f"{surfaced_block}\n\n"
        f"ACTIVE THREADS:\n{threads_block}\n\n"
        f"RECENT CONVERSATION:\n{clip(buffer_block or '(none)', 3000, keep='tail')}\n\n"
        f"TEAM STANDARDS / PAST-ENGAGEMENT GROUNDING:\n{grounding_block}\n\n"
        f"CANDIDATES (from the fast classifier):\n"
        f"{_candidate_block(candidates)}\n\n"
        f"Surface at most {max_surfaced} items. Omit anything below a "
        f"priority_score of {threshold}, and omit anything already surfaced "
        f"this session. Return JSON only."
    )


def _parse_surfaced(
    raw: str,
    candidates: list[ActionItem],
    threshold: float,
    max_surfaced: int,
) -> list[ActionItem]:
    """Map the LLM's surfaced entries back onto (copies of) the candidates."""
    data = parse_llm_json(raw)
    out: list[ActionItem] = []
    seen: set[str] = set()
    for entry in data.get("surfaced", []) or []:
        if not isinstance(entry, dict):
            continue
        try:
            idx = int(entry.get("index"))
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx >= len(candidates):
            continue
        base = candidates[idx]
        try:
            score = float(entry.get("priority_score", base.priority_score))
        except (TypeError, ValueError):
            score = getattr(base, "priority_score", 0.0)
        if score < threshold:
            continue
        question = str(entry.get("question") or base.question).strip()
        key = question.lower()
        if not question or key in seen:
            continue
        seen.add(key)
        rationale = entry.get("rationale")
        out.append(
            replace(
                base,
                question=question,
                priority_score=score,
                rationale=(str(rationale).strip() if rationale else None),
            )
        )
    out.sort(key=lambda i: i.priority_score, reverse=True)
    return out[:max_surfaced]


async def adjudicate(
    candidates: list[ActionItem],
    context,
    config,
    grounding: str = "",
    *,
    objectives: list[str] | None = None,
    already_surfaced: list[str] | None = None,
    llm_fn: LLMFn = call_llm,
) -> list[ActionItem]:
    """Select the few candidates genuinely worth surfacing (deep-tier pass).

    Returns a re-scored, de-duplicated, rationale-annotated subset capped at
    ``sensitivity.max_surfaced_per_pass`` and gated at
    ``sensitivity.surface_threshold``. Questions already surfaced this session
    (``already_surfaced``) are suppressed — primarily by the LLM's semantic
    judgement, with a lexical backstop. Falls back to a deterministic
    threshold+cap filter on any error.
    """
    if not candidates:
        return []

    sens = getattr(config, "sensitivity", None)
    threshold = float(getattr(sens, "surface_threshold", 0.7))
    max_surfaced = int(getattr(sens, "max_surfaced_per_pass", 3))
    objectives = objectives or list(getattr(context, "objectives", None) or [])

    try:
        prompt = _build_user_prompt(
            candidates,
            context,
            config,
            grounding,
            objectives,
            threshold=threshold,
            max_surfaced=max_surfaced,
            already_surfaced=already_surfaced,
        )
        raw = await llm_fn(
            system_prompt=ADJUDICATOR_SYSTEM_PROMPT,
            user_prompt=prompt,
            json_output=True,
            tier="deep",
            timeout=45.0,
        )
        surfaced = _parse_surfaced(raw, candidates, threshold, max_surfaced)
        surfaced = _drop_already_surfaced(surfaced, already_surfaced)
        logger.info(
            "Adjudicated %d candidate(s) → %d surfaced", len(candidates), len(surfaced)
        )
        return surfaced
    except Exception as e:  # noqa: BLE001 — never block the live loop
        logger.warning(
            "Adjudicator failed (%s); falling back to threshold filter", e
        )
        return _drop_already_surfaced(
            _fallback_filter(candidates, threshold, max_surfaced), already_surfaced
        )
