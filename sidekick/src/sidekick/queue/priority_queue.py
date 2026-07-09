"""Three-lane priority queue with merging and expiry."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable

from sidekick.analyst.classifier import ActionItem
from sidekick.config import SidekickConfig
from sidekick.dedup import find_duplicate

logger = logging.getLogger(__name__)


# Synthesis tier per item complexity. Mirrors ``PriorityQueue._route`` so the
# chosen model's latency fits the lane's wall-clock timeout (fast=15s,
# standard=30s, deep=90s). Keep in sync with ``_route``.
_COMPLEXITY_TIER: dict[str, str] = {
    "simple": "fast",
    "medium": "standard",
}

# Enrichment restraint (Phase 6 / 6.4): a duplicate question answered within
# this window is not re-researched — the fresh answer already stands.
_ENRICH_COOLDOWN_SECONDS = 90


@dataclass
class ActionResult:
    """Result produced by an action pipeline."""

    question: str
    action_type: str
    answer: str = ""
    sources: list[str] = field(default_factory=list)
    confidence: str = "medium"
    priority: str = "medium"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # Set when an early "lead answer" was already surfaced via the notify
    # callback during streaming, so the caller skips a duplicate final notify.
    early_notified: bool = False
    # Why this was surfaced (from the Phase 1 adjudicator), carried into the
    # notification for display in the feed. Empty for non-adjudicated results.
    rationale: str = ""

    def format(self) -> str:
        return self.answer


@dataclass
class QueueItem:
    """Wrapper around ActionItem with queue metadata."""

    item: ActionItem
    enqueued_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = "pending"   # pending, running, done, expired
    merged_items: list[ActionItem] = field(default_factory=list)

    @property
    def id(self) -> str:
        return f"{self.item.type}:{self.item.question[:40]}"

    def merge_with(self, other: ActionItem) -> None:
        self.merged_items.append(other)
        logger.info("Merged '%s' into '%s'", other.question[:40], self.id)


class AsyncLane:
    """A concurrency-limited execution lane."""

    def __init__(self, max_concurrent: int, timeout: int):
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.items: list[QueueItem] = []

    async def submit(self, item: ActionItem) -> None:
        qi = QueueItem(item=item)
        self.items.append(qi)

    def get_pending(self) -> list[QueueItem]:
        return [qi for qi in self.items if qi.status == "pending"]

    def get_running(self) -> list[QueueItem]:
        return [qi for qi in self.items if qi.status == "running"]

    def expire_before(self, cutoff: datetime) -> None:
        for qi in self.items:
            if qi.status == "pending" and qi.enqueued_at < cutoff:
                qi.status = "expired"
                logger.info("Expired stale item: %s", qi.id)


class PriorityQueue:
    """Three-lane priority queue with merging and expiry.

    Lane design:
    - Fast:     max 3 concurrent, 15s timeout — simple lookups
    - Standard: max 2 concurrent, 30s timeout — multi-source research
    - Deep:     max 1 concurrent, 90s timeout — complex prototypes
    """

    def __init__(self, config: SidekickConfig):
        self.fast_lane = AsyncLane(
            max_concurrent=config.queue.fast_lane_max, timeout=15,
        )
        self.standard_lane = AsyncLane(
            max_concurrent=config.queue.standard_lane_max, timeout=30,
        )
        self.deep_lane = AsyncLane(
            max_concurrent=config.queue.deep_lane_max, timeout=90,
        )
        self.completed: list[ActionResult] = []
        self.config = config

    async def enqueue(self, item: ActionItem) -> None:
        """Add an action item to the appropriate lane.

        Checks for:
        1. Merge candidates in pending/running queue items
        2. Semantic duplicates against recently completed outputs
           (enriches with new context instead of skipping)
        """
        # Check for merge candidates in queue
        existing = self._find_merge_candidate(item)
        if existing:
            existing.merge_with(item)
            return

        # Check for semantic duplicates against completed outputs
        duplicate_of = self._find_completed_duplicate(item)
        if duplicate_of:
            # Enrichment restraint (6.4): if we answered this same question very
            # recently, skip re-research entirely — the fresh answer already
            # stands. Only re-research once it has gone stale so evolving
            # context can refresh it.
            age = (
                datetime.now(timezone.utc) - duplicate_of.timestamp
            ).total_seconds()
            if age < _ENRICH_COOLDOWN_SECONDS:
                logger.info(
                    "Duplicate within cooldown (%.0fs) — skipping re-research: %s",
                    age,
                    item.question[:60],
                )
                return
            # Mark as a context-enrichment re-research
            item.question = (
                f"[ENRICHED] {item.question} "
                f"(previous answer: {duplicate_of.answer[:200]})"
            )
            logger.info(
                "Duplicate detected, enriching with new context: %s",
                item.question[:60],
            )

        lane = self._route(item)
        await lane.submit(item)
        logger.info(
            "Enqueued [%s/%s]: %s",
            item.complexity,
            item.priority,
            item.question[:60],
        )

    def _effective_answer_tier(self) -> str:
        """Resolve the synthesis tier policy (Phase 4 / A3).

        ``answer_tier: "deep"`` forces deep-model answers; in ``accuracy_mode``
        the default (``"auto"``) is also treated as deep — trading latency for
        accuracy. Otherwise answers stay complexity-routed as before.
        """
        sens = getattr(self.config, "sensitivity", None)
        if getattr(sens, "answer_tier", "auto") == "deep":
            return "deep"
        if getattr(sens, "accuracy_mode", False):
            return "deep"
        return "auto"

    def _route(self, item: ActionItem) -> AsyncLane:
        # Deep-answer policy routes everything to the deep lane so the slower
        # model gets the 90s budget (mirrors the tier chosen in _execute).
        if self._effective_answer_tier() == "deep":
            return self.deep_lane
        if item.complexity == "simple":
            return self.fast_lane
        elif item.complexity == "medium":
            return self.standard_lane
        else:
            return self.deep_lane

    def _find_merge_candidate(self, item: ActionItem) -> QueueItem | None:
        for lane in [self.fast_lane, self.standard_lane, self.deep_lane]:
            for queued in lane.items:
                if queued.status in ("pending", "running"):
                    if (
                        item.batch_with == queued.id
                        or item.related_to == queued.id
                    ):
                        return queued
        return None

    def _find_completed_duplicate(self, item: ActionItem) -> ActionResult | None:
        """Return a recent completed output that is a near-duplicate of ``item``.

        Uses a deterministic local similarity check (no LLM round-trip) against
        the last 10 completed questions. Returns the matching ActionResult when
        similarity meets the threshold, else None.
        """
        if not self.completed:
            return None

        # Only check last 10 completed items
        recent = self.completed[-10:]
        idx = find_duplicate(item.question, [r.question for r in recent])
        if idx is not None:
            return recent[idx]
        return None

    async def process_ready(
        self, research, prototype, context, domains: list[str] | None = None,
        notify: Callable[[object], None] | None = None,
    ) -> list[ActionResult]:
        """Process pending items across all lanes. Returns completed results.

        When *notify* is supplied, research-type items stream their synthesis
        and surface the lead answer early via *notify*; the corresponding
        result is flagged ``early_notified`` so the caller skips a duplicate
        final notification. When *notify* is ``None`` the behaviour is the
        original non-streaming path.
        """
        results: list[ActionResult] = []

        for lane in [self.fast_lane, self.standard_lane, self.deep_lane]:
            pending = lane.get_pending()
            running_count = len(lane.get_running())

            for qi in pending:
                if running_count >= lane.max_concurrent:
                    break

                qi.status = "running"
                running_count += 1

                try:
                    result = await asyncio.wait_for(
                        self._execute(qi, research, prototype, context, domains, notify),
                        timeout=lane.timeout,
                    )
                    # Carry the adjudicator's rationale onto the result so the
                    # notification/feed can show why it was surfaced.
                    result.rationale = getattr(qi.item, "rationale", "") or ""
                    qi.status = "done"
                    results.append(result)
                    self.completed.append(result)
                except asyncio.TimeoutError:
                    qi.status = "expired"
                    logger.warning("Timeout on: %s", qi.id)
                except Exception:
                    qi.status = "expired"
                    logger.exception("Error processing: %s", qi.id)

        # Expire stale items
        await self.expire_stale()

        return results

    async def _execute(
        self, qi: QueueItem, research, prototype, context,
        domains: list[str] | None = None,
        notify: Callable[[object], None] | None = None,
    ) -> ActionResult:
        """Route a queue item to the correct action pipeline."""
        item = qi.item
        action_type = item.type
        # Match the synthesis tier to the lane budget so the model latency fits
        # the wall-clock timeout. Hardcoding "deep" (the slowest model) forced
        # the 15s fast lane and 30s standard lane to run claude-opus, which
        # could not finish in time and expired with zero outputs. The mapping
        # mirrors ``_route``: simple→fast, medium→standard, else→deep. When the
        # deep-answer policy is active (Phase 4 / A3) every answer uses deep.
        if self._effective_answer_tier() == "deep":
            tier = "deep"
        else:
            tier = _COMPLEXITY_TIER.get(item.complexity, "deep")

        # Opt-in self-critique (Phase 4 / A3): draft → critique → refine.
        self_critique = bool(
            getattr(getattr(self.config, "sensitivity", None), "self_critique", False)
        )

        # When notify is provided, surface the lead answer early via a
        # streaming callback. ``early_fired`` records whether it fired so the
        # final ActionResult can be flagged to avoid a duplicate notify.
        early_fired = {"v": False}

        def _make_on_lead(result_type: str):
            if notify is None:
                return None

            def _on_lead(lead: str) -> None:
                early_fired["v"] = True
                notify(ActionResult(
                    question=item.question,
                    action_type=result_type,
                    answer=lead,
                    sources=[],
                    confidence="medium",
                    priority=item.priority,
                    early_notified=True,
                    rationale=getattr(item, "rationale", "") or "",
                ))

            return _on_lead

        if action_type == "research":
            result = await research.execute_direct(
                question=item.question,
                depth="medium" if item.complexity != "simple" else "quick",
                context=context,
                tier=tier,
                domains=domains,
                on_lead=_make_on_lead("research"),
                self_critique=self_critique,
            )
            return ActionResult(
                question=item.question,
                action_type="research",
                answer=result.format() if hasattr(result, "format") else str(result),
                sources=getattr(result, "sources", []),
                confidence=getattr(result, "confidence", "medium"),
                priority=item.priority,
                early_notified=early_fired["v"],
            )
        elif action_type in ("roadmap", "sizing", "diagnostic", "action_item"):
            # Route through research pipeline — these benefit from
            # grounded answers rather than placeholder responses
            result = await research.execute_direct(
                question=item.question,
                depth="medium",
                context=context,
                tier=tier,
                domains=domains,
                on_lead=_make_on_lead(action_type),
                self_critique=self_critique,
            )
            return ActionResult(
                question=item.question,
                action_type=action_type,
                answer=result.format() if hasattr(result, "format") else str(result),
                sources=getattr(result, "sources", []),
                confidence=getattr(result, "confidence", "medium"),
                priority=item.priority,
                early_notified=early_fired["v"],
            )
        elif action_type == "prototype":
            result = await prototype.execute_direct(
                description=item.question,
                prototype_type="notebook",
                context=context,
            )
            return ActionResult(
                question=item.question,
                action_type="prototype",
                answer=result.format() if hasattr(result, "format") else str(result),
                priority=item.priority,
            )
        else:
            # Unknown type — still route through research as a fallback
            result = await research.execute_direct(
                question=item.question,
                depth="quick",
                context=context,
                tier=tier,
                domains=domains,
            )
            return ActionResult(
                question=item.question,
                action_type=action_type,
                answer=result.format() if hasattr(result, "format") else str(result),
                sources=getattr(result, "sources", []),
                priority=item.priority,
            )

    async def expire_stale(self, minutes: int = 5) -> None:
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        for lane in [self.fast_lane, self.standard_lane, self.deep_lane]:
            lane.expire_before(cutoff)

    def get_in_progress(self) -> list[QueueItem]:
        result = []
        for lane in [self.fast_lane, self.standard_lane, self.deep_lane]:
            result.extend(lane.get_running())
        return result

    def format_status(self) -> str:
        parts = []
        for name, lane in [
            ("Fast", self.fast_lane),
            ("Standard", self.standard_lane),
            ("Deep", self.deep_lane),
        ]:
            pending = len(lane.get_pending())
            running = len(lane.get_running())
            if pending or running:
                parts.append(f"  {name}: {running} running, {pending} pending")
        return "\n".join(parts) if parts else "  (empty)"
