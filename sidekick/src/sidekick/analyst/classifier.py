"""Transcript analyst — LLM-powered classification of meeting transcript chunks."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields

from sidekick.analyst.context import MeetingContext
from sidekick.analyst.prompts import build_analyst_system_prompt
from sidekick.config import SidekickConfig
from sidekick.dedup import similarity
from sidekick.llm import call_llm, parse_llm_json

logger = logging.getLogger(__name__)


@dataclass
class ActionItem:
    """A classified action item from transcript analysis."""

    question: str
    type: str                     # research, prototype, roadmap, sizing, diagnostic, action_item, none
    complexity: str               # simple, medium, complex
    priority: str                 # critical, high, medium, low, skip
    priority_score: float = 0.0
    already_answered: bool = False
    consultant_answer_correct: bool | None = None
    correction_needed: bool = False
    correction_detail: str | None = None
    related_to: str | None = None
    relationship_type: str | None = None
    missing_context: str | None = None
    suggest_ask_client: str | None = None
    context_used: list[str] = field(default_factory=list)
    batch_with: str | None = None
    # One-line justification tying the item to an engagement objective/risk,
    # set by the Phase 1 relevance adjudicator. None for plain classifier items.
    rationale: str | None = None


@dataclass
class AnalystResponse:
    """Parsed LLM analyst response."""

    items: list[ActionItem]
    phase: str = "core"
    threads_update: list[dict] = field(default_factory=list)

    # Fields accepted by ActionItem — used to filter out unexpected LLM keys
    _ACTION_ITEM_FIELDS = {f.name for f in fields(ActionItem)}

    @classmethod
    def from_json(cls, text: str) -> AnalystResponse:
        data = parse_llm_json(text)

        # Build ActionItems, filtering out any unexpected keys from the LLM
        items = []
        for raw in data.get("items", []):
            filtered = {k: v for k, v in raw.items() if k in cls._ACTION_ITEM_FIELDS}
            # Ensure required fields exist
            if "question" not in filtered or "type" not in filtered:
                logger.warning("Skipping item missing required fields: %s", raw)
                continue
            filtered.setdefault("complexity", "medium")
            filtered.setdefault("priority", "medium")
            try:
                items.append(ActionItem(**filtered))
            except Exception:
                logger.exception("Failed to create ActionItem from: %s", filtered)

        return cls(
            items=items,
            phase=data.get("phase", "core"),
            threads_update=data.get("threads_update", []),
        )


class TranscriptAnalyst:
    """LLM-powered meeting transcript analyser."""

    def __init__(self, config: SidekickConfig, context: MeetingContext | None = None):
        self.config = config
        self.context = context or MeetingContext()
        self.system_prompt = build_analyst_system_prompt(config)

    async def analyse_chunk(self, chunk: list) -> list[ActionItem]:
        """Analyse new transcript lines against meeting context."""
        # Update context with new lines
        self.context.add_lines(chunk)

        prompt = self._build_prompt(chunk)

        response_text = await call_llm(
            system_prompt=self.system_prompt,
            user_prompt=prompt,
            json_output=True,
            tier="fast",
        )

        response = AnalystResponse.from_json(response_text)

        # Update phase
        if response.phase:
            self.context.current_phase = response.phase

        # Update threads from LLM response
        for thread_data in response.threads_update:
            tid = thread_data.get("thread_id", "")
            if not tid:
                continue

            # The LLM frequently mints a fresh thread_id for a topic it has
            # already opened in an earlier chunk, which previously created a
            # duplicate thread on every mention. Resolve an unknown id to an
            # existing thread with a near-duplicate topic before deciding to
            # create a new one.
            if tid not in self.context.threads:
                match = self._find_thread_by_topic(
                    thread_data.get("topic", tid)
                )
                if match is not None:
                    tid = match

            if tid in self.context.threads:
                t = self.context.threads[tid]
                t.status = thread_data.get("status", t.status)
                t.last_active_at = thread_data.get("last_active_at", t.last_active_at)
                t.questions.extend(thread_data.get("questions", []))
                t.key_facts.extend(thread_data.get("key_facts", []))
            else:
                from sidekick.analyst.context import TopicThread
                self.context.threads[tid] = TopicThread(
                    thread_id=tid,
                    topic=thread_data.get("topic", tid),
                    started_at=thread_data.get("started_at", ""),
                    last_active_at=thread_data.get("last_active_at", ""),
                    status=thread_data.get("status", "open"),
                    questions=thread_data.get("questions", []),
                    key_facts=thread_data.get("key_facts", []),
                )

        # Filter items based on threshold
        items = []
        for item in response.items:
            if item.priority_score >= self.config.sensitivity.trigger_threshold:
                items.append(item)

        # Record decisions in context
        self.context.record_decisions(items)

        logger.info(
            "Analysed %d lines → %d triggers (phase: %s)",
            len(chunk),
            len(items),
            response.phase,
        )

        return items

    # Topics this close are treated as the same thread. Topics are short
    # descriptive phrases, so a high threshold avoids merging genuinely
    # distinct themes while still catching verbatim/near-verbatim repeats.
    _TOPIC_DEDUP_THRESHOLD = 0.8

    def _find_thread_by_topic(self, topic: str) -> str | None:
        """Return the id of an existing thread whose topic matches ``topic``.

        Guards against the analyst LLM minting a fresh ``thread_id`` for a
        topic it already opened, which would otherwise create duplicate
        threads. Returns the best match at/above the dedup threshold, else None.
        """
        if not topic:
            return None
        best_tid: str | None = None
        best_score = 0.0
        for existing_tid, thread in self.context.threads.items():
            score = similarity(topic, thread.topic)
            if score > best_score:
                best_score = score
                best_tid = existing_tid
        return best_tid if best_score >= self._TOPIC_DEDUP_THRESHOLD else None

    def _build_prompt(self, chunk: list) -> str:
        chunk_text = "\n".join(
            f"[{line.start}] {line.speaker}: {line.text}" for line in chunk
        )
        trigger_text = "\n".join(
            f"- {t.pattern} → {t.action} ({t.grounding})"
            for t in self.config.triggers.client_topics
        )

        # Include injected context summaries if available
        injected_context = ""
        if self.context.context_documents:
            recent_docs = self.context.context_documents[-3:]
            summaries = []
            for doc in recent_docs:
                # First 200 chars of each document
                summaries.append(doc[:200])
            injected_context = "\nINJECTED CONTEXT (from add_context):\n" + "\n---\n".join(summaries)

        return f"""MEETING STATE:
Customer: {self.context.customer_name}
Duration so far: {self.context.elapsed_minutes:.0f} minutes
Domains: {', '.join(self.config.domains)}

PARTICIPANTS:
Consultants: {', '.join(self.config.consultant_names)}
Client-side: {', '.join(self.context.identified_clients) or '(identifying...)'}

ACTIVE THREADS:
{self.context.format_threads()}

OPEN QUESTIONS:
{self.context.format_open_questions()}

ANSWERED QUESTIONS:
{self.context.format_answered_questions()}

RECENT CONVERSATION BUFFER (last ~3 minutes):
{self.context.format_recent_buffer()}{injected_context}

NEW TRANSCRIPT CHUNK (last {self.config.sensitivity.analyst_interval_seconds}s):
{chunk_text}

CUSTOM TRIGGERS:
{trigger_text or '(none configured)'}

Analyse this chunk and return your assessment."""


