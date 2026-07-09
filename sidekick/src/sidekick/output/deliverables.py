"""Post-call deliverables — customer-ready follow-up email, action-item table,
and a "couldn't answer live" research batch.

Phase 4b. Turns the artefacts Sidekick already has (transcript, threads,
research answers, action items) into a single markdown deliverable produced on
``stop``. The email draft is LLM-generated (deep tier); the action-item table
and follow-up batch are deterministic so they are unit-testable without a
model and never block on a network call.

The LLM call is injected (``llm_fn``) so tests run offline, and every failure
degrades gracefully — ``generate_deliverables`` always returns a string so the
``stop`` summary is never broken by a deliverables error.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable

from sidekick.config import SidekickConfig, get_output_dir
from sidekick.llm import call_llm
from sidekick.prompt_budget import clip

logger = logging.getLogger(__name__)

LLMFn = Callable[..., Awaitable[str]]

# How much of the (LLM-drafted, variable-length) email to inline in the chat
# ``stop`` response. The full email always lives in the saved file; the inline
# copy is a preview so the whole ``stop`` response stays well under the chat
# tool-result buffer and never spills to an unreadable overflow file.
_EMAIL_PREVIEW_CHARS = 700

_EMAIL_SYSTEM_PROMPT = (
    "You are a senior consultant drafting a concise, professional follow-up "
    "email to a customer after a meeting. Write in British English. Be warm "
    "but businesslike. Structure: a one-line thank-you, a short 'what we "
    "discussed' summary (3-5 bullets), agreed next steps, and a sign-off "
    "placeholder. Do NOT invent commitments, dates, or names that are not in "
    "the supplied context. Output the email body only — no preamble, no "
    "markdown fences."
)


def _action_item_table(context) -> str:
    """Render ``context.action_items`` as a markdown table (deterministic)."""
    items = getattr(context, "action_items", None) or []
    if not items:
        return "_No action items captured during the session._"

    rows = ["| # | Action | Owner | Due |", "|---|--------|-------|-----|"]
    for i, ai in enumerate(items, 1):
        if isinstance(ai, dict):
            desc = ai.get("description", "").strip() or "(unspecified)"
            owner = (ai.get("owner") or "\u2014").strip()
            due = (ai.get("due") or "\u2014").strip()
        else:
            desc, owner, due = str(ai), "\u2014", "\u2014"
        # Keep table cells single-line.
        desc = desc.replace("\n", " ").replace("|", "\\|")
        rows.append(f"| {i} | {desc} | {owner} | {due} |")
    return "\n".join(rows)


_FRAGMENT_TAIL_RE = re.compile(
    r"\b(and|but|or|because|so|as|that|which|the|a|an|to|of|for|with)"
    r"\s*[.,\u2026]?$",
    re.IGNORECASE,
)


def _clean_followups(questions: list[str], *, limit: int = 12) -> list[str]:
    """Filter the follow-up list to well-formed, de-duplicated questions (6.3).

    Drops incomplete fragments (trailing conjunctions/prepositions, too short)
    and near-duplicates so the customer-facing follow-up reads cleanly.
    """
    seen: set[str] = set()
    kept: list[str] = []
    for q in questions:
        t = (q or "").strip()
        key = t.lower().rstrip("?.! ")
        if not t or key in seen:
            continue
        # A well-formed follow-up either ends with '?' or is a reasonably
        # complete statement (>= 6 words) that doesn't trail off mid-clause.
        if not t.endswith("?"):
            if len(t.split()) < 6 or _FRAGMENT_TAIL_RE.search(t):
                continue
        seen.add(key)
        kept.append(t)
        if len(kept) >= limit:
            break
    return kept


def _unanswered_research_batch(session_log, context) -> str:
    """List questions/threads not resolved live, as a follow-up checklist.

    A question is considered "answered live" if its text matches a recorded
    research/prototype output. Open or blocked threads are also surfaced.
    """
    researched = {
        (o.get("question") or "").strip().lower()
        for o in (getattr(session_log, "outputs", None) or [])
    }

    pending: list[str] = []
    for q in getattr(context, "open_questions", None) or []:
        text = (q.get("question") or "").strip()
        if text and text.lower() not in researched:
            pending.append(text)
    # Clean the customer-facing follow-up list: drop garbled fragments and
    # near-duplicates, cap the length (Phase 6 / 6.3).
    pending = _clean_followups(pending)

    open_threads = [
        t.topic
        for t in (getattr(context, "threads", None) or {}).values()
        if getattr(t, "status", "") in ("open", "blocked")
    ]

    if not pending and not open_threads:
        return "_Everything raised was addressed live \u2014 nothing outstanding._"

    parts: list[str] = []
    if pending:
        parts.append("Questions raised but not answered live (run `research` on each):")
        parts.extend(f"- [ ] {q}" for q in pending)
    if open_threads:
        if parts:
            parts.append("")
        parts.append("Open threads needing follow-up:")
        parts.extend(f"- [ ] {topic}" for topic in open_threads)
    return "\n".join(parts)


def _email_context_block(session_log, context, config) -> str:
    """Assemble a compact, token-budgeted context block for the email prompt."""
    customer = getattr(config, "customer", "the customer")

    facts = getattr(context, "key_facts", None) or []
    facts_block = "\n".join(f"- {f}" for f in facts[-12:]) or "(none captured)"

    threads = getattr(context, "threads", None) or {}
    topics_block = (
        "\n".join(
            f"- {t.topic} ({getattr(t, 'status', 'open')})"
            for t in threads.values()
        )
        or "(none)"
    )

    research_block = "(none)"
    outputs = getattr(session_log, "outputs", None) or []
    research = [o for o in outputs if o.get("action_type") == "research"]
    if research:
        research_block = "\n".join(
            f"- Q: {o.get('question', '')}\n  A: {(o.get('answer') or '')[:200]}"
            for o in research[-8:]
        )

    block = (
        f"Customer: {customer}\n\n"
        f"Key facts established:\n{facts_block}\n\n"
        f"Topics discussed:\n{topics_block}\n\n"
        f"Questions researched live:\n{research_block}"
    )
    return clip(block, 6000, keep="head")


async def _draft_email(session_log, context, config, llm_fn: LLMFn) -> str:
    """LLM-draft the follow-up email. Degrades to a placeholder on failure."""
    context_block = _email_context_block(session_log, context, config)
    try:
        body = await llm_fn(
            system_prompt=_EMAIL_SYSTEM_PROMPT,
            user_prompt=(
                "Draft the follow-up email based only on this meeting "
                f"context:\n\n{context_block}"
            ),
            tier="deep",
            timeout=45.0,
        )
        return body.strip()
    except Exception as e:  # noqa: BLE001 — never break the stop summary
        logger.warning("Deliverables email draft failed: %s", e)
        return (
            "_Email draft unavailable (LLM call failed). Key facts and topics "
            "are summarised above; draft manually._"
        )


@dataclass
class DeliverablesPack:
    """The three deliverables sections, kept separate so the caller can render
    the full pack (saved to disk) or a compact digest (inlined in chat).
    """

    customer: str
    email: str
    actions: str
    follow_up: str

    def full_markdown(self) -> str:
        """The complete deliverables pack — written to the saved ``.md`` file."""
        return "\n".join(
            [
                f"# Post-Call Deliverables \u2014 {self.customer}",
                "",
                "## Draft Follow-up Email",
                "",
                self.email,
                "",
                "## Action Items",
                "",
                self.actions,
                "",
                "## Follow-up Research Batch",
                "",
                self.follow_up,
                "",
            ]
        )

    def inline_digest(
        self,
        saved_path: str | None = None,
        *,
        email_preview_chars: int = _EMAIL_PREVIEW_CHARS,
    ) -> str:
        """A compact, size-bounded version for the chat ``stop`` response.

        Inlining the full pack (a real email runs to several KB) overflows the
        chat tool-result buffer, which spills it to a file the agent can't read
        — so the deliverables never render. This keeps the short, immediately
        actionable sections (action items + follow-up batch) verbatim and shows
        only a preview of the long email, pointing at the saved file for the
        rest.
        """
        email_preview = clip(self.email, email_preview_chars, keep="head")
        truncated = len(email_preview) < len(self.email)
        if truncated:
            pointer = (
                f"_\u2026email truncated \u2014 full draft in `{saved_path}`._"
                if saved_path
                else "_\u2026email truncated \u2014 see the saved deliverables file._"
            )
        else:
            pointer = ""

        parts = [
            f"# Post-Call Deliverables \u2014 {self.customer}",
            "",
            "## Draft Follow-up Email (preview)",
            "",
            email_preview,
        ]
        if pointer:
            parts += ["", pointer]
        parts += [
            "",
            "## Action Items",
            "",
            self.actions,
            "",
            "## Follow-up Research Batch",
            "",
            self.follow_up,
            "",
        ]
        return "\n".join(parts)


async def build_deliverables(
    session_log,
    context,
    config,
    *,
    llm_fn: LLMFn = call_llm,
) -> DeliverablesPack:
    """Assemble the deliverables sections for a finished session.

    Returns a :class:`DeliverablesPack` so the caller can persist the full
    markdown and inline only a compact digest.
    """
    customer = getattr(config, "customer", "the customer")
    email = await _draft_email(session_log, context, config, llm_fn)
    actions = _action_item_table(context)
    follow_up = _unanswered_research_batch(session_log, context)
    return DeliverablesPack(customer, email, actions, follow_up)


async def generate_deliverables(
    session_log,
    context,
    config,
    *,
    llm_fn: LLMFn = call_llm,
) -> str:
    """Produce the full markdown deliverables block for a finished session."""
    pack = await build_deliverables(session_log, context, config, llm_fn=llm_fn)
    return pack.full_markdown()


def save_deliverables(
    content: str, config: SidekickConfig, *, force: bool = False
) -> Path | None:
    """Write the deliverables markdown to the customer output directory.

    ``force=True`` persists even when ``output.auto_save`` is off — used on an
    explicit ``stop`` so there is always a file to open, since the inline chat
    copy is only a preview.
    """
    if not force and not getattr(config.output, "auto_save", False):
        return None
    output_dir = get_output_dir(config.customer)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"deliverables_{timestamp}.md"
    path.write_text(content, encoding="utf-8")
    logger.info("Deliverables saved to %s", path)
    return path
