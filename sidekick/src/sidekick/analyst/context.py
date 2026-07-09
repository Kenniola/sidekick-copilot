"""Meeting context accumulator — tracks threads, questions, and meeting state."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class TranscriptLine:
    """Imported type alias for transcript lines."""
    start: str
    end: str
    speaker: str
    text: str


@dataclass
class TopicThread:
    """A conversation thread within the meeting."""

    thread_id: str
    topic: str
    started_at: str
    last_active_at: str
    status: str = "open"            # open, answered, blocked, closed
    key_facts: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    sidekick_outputs: list[str] = field(default_factory=list)


@dataclass
class MeetingContext:
    """Accumulated state for the entire meeting."""

    customer_name: str = ""
    start_time: datetime | None = None
    participants: dict[str, str] = field(default_factory=dict)  # name → role

    # Rolling buffer: last ~3 minutes of raw transcript
    recent_buffer: deque = field(default_factory=lambda: deque(maxlen=200))

    # All transcript lines (full meeting)
    full_transcript: list = field(default_factory=list)

    # Structured state
    threads: dict[str, TopicThread] = field(default_factory=dict)
    open_questions: list[dict] = field(default_factory=list)
    answered_questions: list[dict] = field(default_factory=list)
    action_items: list[dict] = field(default_factory=list)
    key_facts: list[str] = field(default_factory=list)

    # Injected context — documents, notes, and image descriptions added live
    context_documents: list[str] = field(default_factory=list)

    # Engagement objectives (Phase 1 / A2). Set from an ``add_context
    # "goal: …"`` note, seeded from config, or auto-inferred from the opening
    # minutes. The relevance adjudicator scores candidates against these.
    objectives: list[str] = field(default_factory=list)

    # Auto-detected domains from transcript (supplements config domains)
    detected_domains: list[str] = field(default_factory=list)

    # Phase tracking
    current_phase: str = "opening"  # opening, core, deepdive, wrapup

    @property
    def elapsed_minutes(self) -> float:
        if not self.start_time:
            return 0.0
        delta = datetime.now(timezone.utc) - self.start_time
        return delta.total_seconds() / 60.0

    @property
    def identified_clients(self) -> list[str]:
        return [
            name for name, role in self.participants.items()
            if role == "client"
        ]

    def add_lines(self, lines) -> None:
        """Add new transcript lines and update context."""
        if not self.start_time:
            self.start_time = datetime.now(timezone.utc)

        for line in lines:
            self.full_transcript.append(line)
            self.recent_buffer.append(line)

    def record_decisions(self, items) -> None:
        """Record analyst decisions into meeting context."""
        for item in items:
            itype = getattr(item, "type", "unknown")
            q = {
                "question": getattr(item, "question", str(item)),
                "type": itype,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self.open_questions.append(q)
            # Capture action_item classifications so the deliverables table is
            # populated (Phase 6 / 6.2). Deduped by description.
            if itype == "action_item":
                desc = (getattr(item, "question", "") or "").strip()
                if desc and not any(
                    a.get("description") == desc for a in self.action_items
                ):
                    self.action_items.append({"description": desc})

    def mark_answered(self, question: str) -> None:
        """Move a question from open to answered."""
        for i, q in enumerate(self.open_questions):
            if q["question"] == question:
                q["answered_at"] = datetime.now(timezone.utc).isoformat()
                self.answered_questions.append(q)
                self.open_questions.pop(i)
                return

    def format_threads(self) -> str:
        if not self.threads:
            return "(no threads yet)"
        parts = []
        for t in self.threads.values():
            parts.append(f"[{t.status}] {t.topic}")
            for q in t.questions:
                parts.append(f"  └─ {q}")
        return "\n".join(parts)

    def format_open_questions(self) -> str:
        if not self.open_questions:
            return "(none)"
        return "\n".join(
            f"- {q['question']} ({q['type']})" for q in self.open_questions
        )

    def format_answered_questions(self) -> str:
        if not self.answered_questions:
            return "(none)"
        return "\n".join(
            f"- {q['question']}" for q in self.answered_questions[-5:]
        )

    def format_recent_buffer(self) -> str:
        if not self.recent_buffer:
            return "(empty)"
        return "\n".join(
            f"[{line.start}] {line.speaker}: {line.text}"
            for line in self.recent_buffer
        )
