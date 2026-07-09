"""Session state container for the MCP server (Phase 2d).

Collects the ~15 module-level globals that previously lived in ``server.py``
into a single dataclass instance. Functions mutate ``_state`` attributes
in place, which removes the need for ``global`` declarations and makes the
server's mutable state explicit and easy to reset between sessions.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from sidekick.config import SidekickConfig
    from sidekick.analyst.classifier import TranscriptAnalyst
    from sidekick.analyst.context import MeetingContext
    from sidekick.queue.priority_queue import PriorityQueue
    from sidekick.actions.research import ResearchPipeline
    from sidekick.actions.prototype import PrototypePipeline
    from sidekick.output.session_log import SessionLog


@dataclass
class SessionState:
    """Mutable state for the lifetime of an MCP server session."""

    # Core session components
    config: "SidekickConfig | None" = None
    context: "MeetingContext | None" = None
    analyst: "TranscriptAnalyst | None" = None
    queue: "PriorityQueue | None" = None
    session_log: "SessionLog | None" = None
    research: "ResearchPipeline | None" = None
    prototype: "PrototypePipeline | None" = None

    # Tier 2 — live audio capture
    audio_capture: object | None = None        # AudioCapture instance (primary)
    audio_captures: list | None = None         # all captures (loopback + mic, 5d)
    recogniser: object | None = None           # SpeechRecogniser instance
    listen_task: "asyncio.Task | None" = None

    # Derived Whisper vocabulary prior (Phase 5b) — seeded from config/grounding
    # and adapted in-session from LLM-corrected key_facts/research.
    vocabulary: object | None = None           # transcript.vocabulary.Vocabulary

    # Error tracking for background loops
    last_error: str | None = None

    # Delta tracking — unified counter for all tools
    last_surface_output_count: int = 0
    last_surface_thread_count: int = 0

    # Domain auto-detection — runs after first 3 classifier batches
    classify_batch_count: int = 0
    domains_detected: bool = False

    # Two-stage accuracy pipeline (Phase 1). In accuracy_mode, fast-tier
    # candidates accumulate here between deep-tier adjudicator passes.
    pending_candidates: list = field(default_factory=list)
    last_adjudicate_time: float = 0.0        # monotonic clock of last pass
    objectives_inferred: bool = False        # one-shot objective inference guard

    # Proactive advisor (Phase 9.3). Slow-cadence "question to ask the client"
    # suggestions, deduped against the recently suggested list.
    last_suggest_time: float = 0.0           # monotonic clock of last suggestion
    recent_suggestions: list = field(default_factory=list)

    # Session memory of questions the adjudicator has already surfaced, so
    # later passes don't re-surface the same point in different words.
    surfaced_questions: list = field(default_factory=list)

    # Grounding context cache — avoids re-reading files on every call
    grounding_cache: str | None = None
    grounding_cache_time: float = 0.0

    def reset(self) -> None:
        """Reset the per-session counters and caches (not the components)."""
        self.last_error = None
        self.last_surface_output_count = 0
        self.last_surface_thread_count = 0
        self.grounding_cache = None
        self.grounding_cache_time = 0.0
        self.classify_batch_count = 0
        self.domains_detected = False
