"""Tests for the relevance adjudicator (Phase 1 / A1).

The deep-tier LLM call is injected so the suite runs offline. Covers the
select/merge/re-score/cap/threshold logic and the graceful fallback.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from sidekick.analyst import adjudicator as adj
from sidekick.analyst.classifier import ActionItem


def _item(question, score, type_="research"):
    return ActionItem(
        question=question,
        type=type_,
        complexity="medium",
        priority="high",
        priority_score=score,
    )


def _config(threshold=0.7, max_surfaced=3):
    return SimpleNamespace(
        customer="Acme",
        sensitivity=SimpleNamespace(
            surface_threshold=threshold, max_surfaced_per_pass=max_surfaced
        ),
    )


def _context(objectives=None):
    return SimpleNamespace(
        objectives=list(objectives or []),
        format_threads=lambda: "(none)",
        format_recent_buffer=lambda: "recent buffer text",
    )


def _llm_returning(payload):
    async def _fn(**kwargs):
        return json.dumps(payload)

    return _fn


async def _boom_llm(**kwargs):
    raise RuntimeError("deep tier down")


class TestAdjudicate:
    @pytest.mark.asyncio
    async def test_empty_candidates_short_circuits(self):
        async def _must_not_call(**kwargs):
            raise AssertionError("llm should not be called for empty candidates")

        out = await adj.adjudicate([], _context(), _config(), llm_fn=_must_not_call)
        assert out == []

    @pytest.mark.asyncio
    async def test_surfaces_refined_item_with_rationale(self):
        candidates = [_item("verbose q0", 0.6), _item("q1", 0.8)]
        payload = {
            "surfaced": [
                {
                    "index": 1,
                    "question": "Refined, specific question about F64?",
                    "priority_score": 0.9,
                    "rationale": "Directly de-risks the sizing objective.",
                }
            ]
        }
        out = await adj.adjudicate(
            candidates, _context(["size F64"]), _config(), "grounding",
            llm_fn=_llm_returning(payload),
        )
        assert len(out) == 1
        assert out[0].question == "Refined, specific question about F64?"
        assert out[0].priority_score == 0.9
        assert out[0].rationale == "Directly de-risks the sizing objective."
        assert out[0].type == "research"  # preserved from the base candidate

    @pytest.mark.asyncio
    async def test_below_threshold_dropped(self):
        candidates = [_item("q0", 0.8)]
        payload = {"surfaced": [{"index": 0, "priority_score": 0.5}]}  # < 0.7
        out = await adj.adjudicate(
            candidates, _context(), _config(), llm_fn=_llm_returning(payload)
        )
        assert out == []

    @pytest.mark.asyncio
    async def test_capped_to_max_surfaced(self):
        candidates = [_item(f"q{i}", 0.8) for i in range(4)]
        payload = {
            "surfaced": [
                {"index": i, "question": f"kept {i}", "priority_score": 0.8}
                for i in range(4)
            ]
        }
        out = await adj.adjudicate(
            candidates, _context(), _config(max_surfaced=2),
            llm_fn=_llm_returning(payload),
        )
        assert len(out) == 2

    @pytest.mark.asyncio
    async def test_duplicate_questions_deduped(self):
        candidates = [_item("q0", 0.8), _item("q1", 0.8)]
        payload = {
            "surfaced": [
                {"index": 0, "question": "Same question", "priority_score": 0.9},
                {"index": 1, "question": "same question", "priority_score": 0.85},
            ]
        }
        out = await adj.adjudicate(
            candidates, _context(), _config(), llm_fn=_llm_returning(payload)
        )
        assert len(out) == 1

    @pytest.mark.asyncio
    async def test_invalid_index_skipped(self):
        candidates = [_item("q0", 0.8)]
        payload = {
            "surfaced": [
                {"index": 99, "priority_score": 0.9},   # out of range
                {"index": 0, "priority_score": 0.9},
            ]
        }
        out = await adj.adjudicate(
            candidates, _context(), _config(), llm_fn=_llm_returning(payload)
        )
        assert len(out) == 1
        assert out[0].question == "q0"

    @pytest.mark.asyncio
    async def test_sorted_by_score_desc(self):
        candidates = [_item("q0", 0.8), _item("q1", 0.8)]
        payload = {
            "surfaced": [
                {"index": 0, "question": "low", "priority_score": 0.72},
                {"index": 1, "question": "high", "priority_score": 0.95},
            ]
        }
        out = await adj.adjudicate(
            candidates, _context(), _config(), llm_fn=_llm_returning(payload)
        )
        assert [o.question for o in out] == ["high", "low"]


class TestSessionMemory:
    @pytest.mark.asyncio
    async def test_already_surfaced_injected_into_prompt(self):
        seen = {}

        async def _capture(**kwargs):
            seen["prompt"] = kwargs["user_prompt"]
            return json.dumps({"surfaced": []})

        await adj.adjudicate(
            [_item("q0", 0.9)],
            _context(),
            _config(),
            already_surfaced=["earlier GitHub-vs-ADO question"],
            llm_fn=_capture,
        )
        assert "ALREADY SURFACED THIS SESSION" in seen["prompt"]
        assert "earlier GitHub-vs-ADO question" in seen["prompt"]

    @pytest.mark.asyncio
    async def test_lexical_backstop_drops_repeat(self):
        q = "Is there a concrete advantage to GitHub over ADO for this project"
        candidates = [_item(q, 0.9)]
        payload = {"surfaced": [{"index": 0, "question": q, "priority_score": 0.9}]}
        out = await adj.adjudicate(
            candidates,
            _context(),
            _config(),
            already_surfaced=[q],
            llm_fn=_llm_returning(payload),
        )
        assert out == []

    @pytest.mark.asyncio
    async def test_new_angle_not_dropped(self):
        candidates = [_item("What is the Dataverse one-workspace limit?", 0.9)]
        payload = {
            "surfaced": [
                {
                    "index": 0,
                    "question": "What is the Dataverse one-workspace limit?",
                    "priority_score": 0.9,
                }
            ]
        }
        out = await adj.adjudicate(
            candidates,
            _context(),
            _config(),
            already_surfaced=["Is there a concrete advantage to GitHub over ADO?"],
            llm_fn=_llm_returning(payload),
        )
        assert len(out) == 1

    @pytest.mark.asyncio
    async def test_backstop_applies_on_fallback_too(self):
        q = "Concrete advantage to GitHub over ADO for this Fabric project"
        candidates = [_item(q, 0.9)]
        out = await adj.adjudicate(
            candidates,
            _context(),
            _config(),
            already_surfaced=[q],
            llm_fn=_boom_llm,
        )
        assert out == []


class TestFallback:
    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_to_threshold_filter(self):
        candidates = [_item("q0", 0.9), _item("q1", 0.6), _item("q2", 0.4)]
        out = await adj.adjudicate(
            candidates, _context(), _config(), llm_fn=_boom_llm
        )
        # Only the >= 0.7 candidate survives; capped and ordered by score.
        assert [o.question for o in out] == ["q0"]

    @pytest.mark.asyncio
    async def test_fallback_respects_cap_and_order(self):
        candidates = [_item("q0", 0.75), _item("q1", 0.95), _item("q2", 0.85)]
        out = await adj.adjudicate(
            candidates, _context(), _config(max_surfaced=2), llm_fn=_boom_llm
        )
        assert [o.question for o in out] == ["q1", "q2"]

    def test_fallback_filter_direct(self):
        candidates = [_item("a", 0.9), _item("b", 0.5)]
        kept = adj._fallback_filter(candidates, threshold=0.7, max_surfaced=3)
        assert [k.question for k in kept] == ["a"]
