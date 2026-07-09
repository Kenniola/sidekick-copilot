"""Tests for the three-lane priority queue (Phase 3).

Covers lane routing, in-queue merging, deterministic dedup/enrichment against
completed outputs, concurrency caps, per-lane timeout handling, pipeline error
handling, and stale-item expiry. Pipelines are faked so the tests are fast and
deterministic (lane timeouts are shrunk where a timeout is being exercised).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from sidekick.analyst.classifier import ActionItem
from sidekick.queue.priority_queue import ActionResult, PriorityQueue


# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


def _make_queue(
    fast=3, standard=2, deep=1, answer_tier="auto", accuracy_mode=False,
    self_critique=False,
) -> PriorityQueue:
    cfg = SimpleNamespace(
        queue=SimpleNamespace(
            fast_lane_max=fast,
            standard_lane_max=standard,
            deep_lane_max=deep,
            stale_expiry_minutes=5,
        ),
        sensitivity=SimpleNamespace(
            answer_tier=answer_tier,
            accuracy_mode=accuracy_mode,
            self_critique=self_critique,
        ),
    )
    return PriorityQueue(config=cfg)


def _item(question="Throughput limit?", type="research", complexity="simple",
          priority="high", **kwargs) -> ActionItem:
    return ActionItem(
        question=question, type=type, complexity=complexity,
        priority=priority, **kwargs,
    )


class _FakeResult:
    def __init__(self, answer="answer", sources=None, confidence="high"):
        self.answer = answer
        self.sources = sources or []
        self.confidence = confidence

    def format(self) -> str:
        return self.answer


class _FakeResearch:
    def __init__(self, result=None, sleep=0.0, exc=None):
        self.result = result or _FakeResult()
        self.sleep = sleep
        self.exc = exc
        self.calls = 0

    async def execute_direct(self, **kwargs):
        self.calls += 1
        if self.sleep:
            await asyncio.sleep(self.sleep)
        if self.exc:
            raise self.exc
        return self.result


class _FakePrototype:
    def __init__(self, result=None):
        self.result = result or _FakeResult(answer="# code")

    async def execute_direct(self, **kwargs):
        return self.result


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


class TestRouting:
    def test_simple_routes_to_fast_lane(self):
        q = _make_queue()
        assert q._route(_item(complexity="simple")) is q.fast_lane

    def test_medium_routes_to_standard_lane(self):
        q = _make_queue()
        assert q._route(_item(complexity="medium")) is q.standard_lane

    def test_complex_routes_to_deep_lane(self):
        q = _make_queue()
        assert q._route(_item(complexity="complex")) is q.deep_lane

    def test_deep_answer_tier_routes_all_to_deep(self):
        q = _make_queue(answer_tier="deep")
        assert q._route(_item(complexity="simple")) is q.deep_lane
        assert q._route(_item(complexity="medium")) is q.deep_lane

    def test_accuracy_mode_routes_all_to_deep(self):
        q = _make_queue(accuracy_mode=True)
        assert q._route(_item(complexity="simple")) is q.deep_lane

    def test_effective_answer_tier_resolution(self):
        assert _make_queue()._effective_answer_tier() == "auto"
        assert _make_queue(answer_tier="deep")._effective_answer_tier() == "deep"
        assert _make_queue(accuracy_mode=True)._effective_answer_tier() == "deep"

    @pytest.mark.asyncio
    async def test_enqueue_places_item_in_routed_lane(self):
        q = _make_queue()
        await q.enqueue(_item(complexity="medium"))
        assert len(q.standard_lane.items) == 1
        assert len(q.fast_lane.items) == 0


# ---------------------------------------------------------------------------
# Synthesis tier selection (must match the lane budget)
# ---------------------------------------------------------------------------


class _TierCapturingResearch:
    """Fake research pipeline that records the ``tier`` it was called with."""

    def __init__(self, result=None):
        self.result = result or _FakeResult()
        self.tier = None
        self.self_critique = None

    async def execute_direct(self, **kwargs):
        self.tier = kwargs.get("tier")
        self.self_critique = kwargs.get("self_critique")
        return self.result


class TestTierSelection:
    """Regression: research items must use a tier that fits their lane timeout.

    Hardcoding ``tier="deep"`` (claude-opus, the slowest model) for every item
    caused 15s fast-lane and 30s standard-lane research to exceed the wall-clock
    timeout and expire with zero outputs. The tier must mirror ``_route``.
    """

    @pytest.mark.asyncio
    async def test_simple_uses_fast_tier(self):
        q = _make_queue()
        research = _TierCapturingResearch()
        await q.enqueue(_item(complexity="simple"))
        await q.process_ready(research, _FakePrototype(), context=None)
        assert research.tier == "fast"

    @pytest.mark.asyncio
    async def test_medium_uses_standard_tier(self):
        q = _make_queue()
        research = _TierCapturingResearch()
        await q.enqueue(_item(complexity="medium"))
        await q.process_ready(research, _FakePrototype(), context=None)
        assert research.tier == "standard"

    @pytest.mark.asyncio
    async def test_complex_uses_deep_tier(self):
        q = _make_queue()
        research = _TierCapturingResearch()
        await q.enqueue(_item(complexity="complex"))
        await q.process_ready(research, _FakePrototype(), context=None)
        assert research.tier == "deep"

    @pytest.mark.asyncio
    async def test_deep_answer_tier_forces_deep_synthesis(self):
        q = _make_queue(answer_tier="deep")
        research = _TierCapturingResearch()
        await q.enqueue(_item(complexity="simple"))
        await q.process_ready(research, _FakePrototype(), context=None)
        assert research.tier == "deep"

    @pytest.mark.asyncio
    async def test_accuracy_mode_forces_deep_synthesis(self):
        q = _make_queue(accuracy_mode=True)
        research = _TierCapturingResearch()
        await q.enqueue(_item(complexity="medium"))
        await q.process_ready(research, _FakePrototype(), context=None)
        assert research.tier == "deep"

    @pytest.mark.asyncio
    async def test_self_critique_flag_passed_to_research(self):
        q = _make_queue(self_critique=True)
        research = _TierCapturingResearch()
        await q.enqueue(_item(complexity="medium"))
        await q.process_ready(research, _FakePrototype(), context=None)
        assert research.self_critique is True

    @pytest.mark.asyncio
    async def test_self_critique_off_by_default(self):
        q = _make_queue()
        research = _TierCapturingResearch()
        await q.enqueue(_item(complexity="medium"))
        await q.process_ready(research, _FakePrototype(), context=None)
        assert research.self_critique is False


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------


class TestMerging:
    @pytest.mark.asyncio
    async def test_batch_with_merges_into_existing(self):
        q = _make_queue()
        first = _item(question="Throughput limit?", complexity="medium")
        await q.enqueue(first)
        first_id = q.standard_lane.items[0].id
        second = _item(
            question="follow up", complexity="medium",
            batch_with=first_id,
        )
        await q.enqueue(second)
        # Merged, not added as a new queue item.
        assert len(q.standard_lane.items) == 1
        assert len(q.standard_lane.items[0].merged_items) == 1

    @pytest.mark.asyncio
    async def test_related_to_merges_into_existing(self):
        q = _make_queue()
        first = _item(question="Capacity sizing?", complexity="complex")
        await q.enqueue(first)
        first_id = q.deep_lane.items[0].id
        second = _item(
            question="related q", complexity="complex",
            related_to=first_id,
        )
        await q.enqueue(second)
        assert len(q.deep_lane.items) == 1
        assert len(q.deep_lane.items[0].merged_items) == 1


# ---------------------------------------------------------------------------
# Dedup / enrichment against completed outputs
# ---------------------------------------------------------------------------


class TestDedup:
    @pytest.mark.asyncio
    async def test_near_duplicate_is_enriched(self):
        q = _make_queue()
        q.completed.append(
            ActionResult(
                question="What is the Fabric capacity throughput limit?",
                action_type="research",
                answer="It is governed by the SKU CU allocation.",
                # Older than the enrichment cooldown so it re-researches (6.4).
                timestamp=datetime.now(timezone.utc) - timedelta(seconds=120),
            )
        )
        dup = _item(question="What is the Fabric capacity throughput limit?")
        await q.enqueue(dup)
        queued = q.fast_lane.items[0].item
        assert queued.question.startswith("[ENRICHED]")
        assert "previous answer" in queued.question

    @pytest.mark.asyncio
    async def test_distinct_question_is_not_enriched(self):
        q = _make_queue()
        q.completed.append(
            ActionResult(
                question="What is the Fabric capacity throughput limit?",
                action_type="research",
                answer="SKU CU allocation.",
            )
        )
        fresh = _item(question="How do I configure OneLake shortcuts to S3?")
        await q.enqueue(fresh)
        assert not q.fast_lane.items[0].item.question.startswith("[ENRICHED]")


# ---------------------------------------------------------------------------
# process_ready — concurrency, success, timeout, errors
# ---------------------------------------------------------------------------


class TestProcessReady:
    @pytest.mark.asyncio
    async def test_research_result_recorded_in_completed(self):
        q = _make_queue()
        await q.enqueue(_item())
        research = _FakeResearch(_FakeResult(answer="A", sources=["http://x"]))
        results = await q.process_ready(
            research=research, prototype=None, context=None,
        )
        assert len(results) == 1
        assert results[0].action_type == "research"
        assert results[0].answer == "A"
        assert q.completed[-1] is results[0]
        assert q.fast_lane.items[0].status == "done"

    @pytest.mark.asyncio
    async def test_prototype_routes_to_prototype_pipeline(self):
        q = _make_queue()
        await q.enqueue(_item(question="build a notebook", type="prototype",
                              complexity="complex"))
        proto = _FakePrototype(_FakeResult(answer="# generated"))
        results = await q.process_ready(
            research=None, prototype=proto, context=None,
        )
        assert len(results) == 1
        assert results[0].action_type == "prototype"
        assert results[0].answer == "# generated"

    @pytest.mark.asyncio
    async def test_concurrency_cap_limits_processed_per_call(self):
        q = _make_queue(fast=2)
        for i in range(3):
            await q.enqueue(_item(question=f"Q{i}"))
        research = _FakeResearch()
        results = await q.process_ready(
            research=research, prototype=None, context=None,
        )
        assert len(results) == 2
        assert len(q.fast_lane.get_pending()) == 1

    @pytest.mark.asyncio
    async def test_timeout_marks_item_expired(self):
        q = _make_queue()
        q.fast_lane.timeout = 0.01
        await q.enqueue(_item())
        research = _FakeResearch(sleep=0.1)
        results = await q.process_ready(
            research=research, prototype=None, context=None,
        )
        assert results == []
        assert q.fast_lane.items[0].status == "expired"

    @pytest.mark.asyncio
    async def test_pipeline_exception_marks_item_expired(self):
        q = _make_queue()
        await q.enqueue(_item())
        research = _FakeResearch(exc=RuntimeError("pipeline boom"))
        results = await q.process_ready(
            research=research, prototype=None, context=None,
        )
        assert results == []
        assert q.fast_lane.items[0].status == "expired"

    @pytest.mark.asyncio
    async def test_notify_surfaces_lead_early_and_flags_result(self):
        # When notify is passed, a research item that streams a lead answer
        # fires notify early and the final result is flagged early_notified
        # so the caller skips a duplicate notification.
        class _StreamingResearch(_FakeResearch):
            async def execute_direct(self, **kwargs):
                self.calls += 1
                on_lead = kwargs.get("on_lead")
                if on_lead is not None:
                    on_lead("Lead answer.")
                return self.result

        q = _make_queue()
        await q.enqueue(_item())
        research = _StreamingResearch(_FakeResult(answer="Full answer."))
        notified = []
        results = await q.process_ready(
            research=research, prototype=None, context=None,
            notify=notified.append,
        )
        assert len(results) == 1
        assert results[0].early_notified is True
        assert len(notified) == 1
        assert notified[0].answer == "Lead answer."
        assert notified[0].early_notified is True

    @pytest.mark.asyncio
    async def test_no_notify_leaves_result_unflagged(self):
        # Default path (notify=None) never fires early and leaves the result
        # unflagged — byte-identical to prior behaviour.
        q = _make_queue()
        await q.enqueue(_item())
        research = _FakeResearch(_FakeResult(answer="A"))
        results = await q.process_ready(
            research=research, prototype=None, context=None,
        )
        assert results[0].early_notified is False



# ---------------------------------------------------------------------------
# Expiry
# ---------------------------------------------------------------------------


class TestExpiry:
    @pytest.mark.asyncio
    async def test_stale_pending_item_expires(self):
        q = _make_queue()
        await q.enqueue(_item())
        q.fast_lane.items[0].enqueued_at = (
            datetime.now(timezone.utc) - timedelta(minutes=10)
        )
        await q.expire_stale(minutes=5)
        assert q.fast_lane.items[0].status == "expired"

    @pytest.mark.asyncio
    async def test_fresh_item_does_not_expire(self):
        q = _make_queue()
        await q.enqueue(_item())
        await q.expire_stale(minutes=5)
        assert q.fast_lane.items[0].status == "pending"


def _lane_total(q) -> int:
    return (
        len(q.fast_lane.items) + len(q.standard_lane.items) + len(q.deep_lane.items)
    )


class TestEnrichmentCooldown:
    """Phase 6 / 6.4: a very recent duplicate is not re-researched."""

    @pytest.mark.asyncio
    async def test_recent_duplicate_skipped(self):
        q = _make_queue()
        q.completed.append(
            ActionResult(
                question="What is F64 capacity?",
                action_type="research",
                answer="A",
                timestamp=datetime.now(timezone.utc),
            )
        )
        await q.enqueue(_item(question="What is F64 capacity?"))
        assert _lane_total(q) == 0  # within cooldown → skipped

    @pytest.mark.asyncio
    async def test_stale_duplicate_is_enriched(self):
        q = _make_queue()
        q.completed.append(
            ActionResult(
                question="What is F64 capacity?",
                action_type="research",
                answer="A",
                timestamp=datetime.now(timezone.utc) - timedelta(seconds=120),
            )
        )
        item = _item(question="What is F64 capacity?")
        await q.enqueue(item)
        assert _lane_total(q) == 1
        assert item.question.startswith("[ENRICHED]")
