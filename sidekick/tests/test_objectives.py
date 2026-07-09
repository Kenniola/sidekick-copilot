"""Tests for engagement objectives (Phase 1 / A2).

Covers the three objective sources: an ``add_context "goal: …"`` note
(`server._parse_goal_note`), explicit config seeding + auto-inference
(`engine._ensure_objectives`), and the fast-tier inference itself
(`engine.infer_objectives`).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sidekick import engine, server
from sidekick.session_state import SessionState


class TestParseGoalNote:
    def test_single_goal(self):
        assert server._parse_goal_note("goal: land the S3 PoC") == ["land the S3 PoC"]

    def test_plural_prefix(self):
        assert server._parse_goal_note("Goals: de-risk F64") == ["de-risk F64"]

    def test_multi_goal_split_on_semicolon(self):
        out = server._parse_goal_note("goal: land the S3 PoC; de-risk F64 sizing")
        assert out == ["land the S3 PoC", "de-risk F64 sizing"]

    def test_split_on_newlines_and_bullets(self):
        out = server._parse_goal_note("goals:\n- land PoC\n- de-risk F64")
        assert out == ["land PoC", "de-risk F64"]

    def test_non_goal_note_returns_empty(self):
        assert server._parse_goal_note("just some meeting notes") == []

    def test_case_insensitive_prefix(self):
        assert server._parse_goal_note("GOAL: ship it") == ["ship it"]


class TestEnsureObjectives:
    @pytest.mark.asyncio
    async def test_noop_when_already_set(self, monkeypatch):
        called = {"infer": False}

        async def _fake_infer(state):
            called["infer"] = True

        monkeypatch.setattr(engine, "infer_objectives", _fake_infer)
        s = SessionState()
        s.context = SimpleNamespace(objectives=["existing"])
        s.config = SimpleNamespace(objectives=["cfg"])
        await engine._ensure_objectives(s)
        assert s.context.objectives == ["existing"]
        assert called["infer"] is False

    @pytest.mark.asyncio
    async def test_seeds_from_config(self):
        s = SessionState()
        s.context = SimpleNamespace(objectives=[])
        s.config = SimpleNamespace(objectives=["land PoC", "de-risk F64"])
        await engine._ensure_objectives(s)
        assert s.context.objectives == ["land PoC", "de-risk F64"]
        assert s.objectives_inferred is True

    @pytest.mark.asyncio
    async def test_auto_infers_after_three_batches(self, monkeypatch):
        called = {"infer": False}

        async def _fake_infer(state):
            called["infer"] = True

        monkeypatch.setattr(engine, "infer_objectives", _fake_infer)
        s = SessionState()
        s.context = SimpleNamespace(objectives=[])
        s.config = SimpleNamespace(objectives=[])
        s.classify_batch_count = 3
        await engine._ensure_objectives(s)
        assert called["infer"] is True

    @pytest.mark.asyncio
    async def test_does_not_infer_before_three_batches(self, monkeypatch):
        called = {"infer": False}

        async def _fake_infer(state):
            called["infer"] = True

        monkeypatch.setattr(engine, "infer_objectives", _fake_infer)
        s = SessionState()
        s.context = SimpleNamespace(objectives=[])
        s.config = SimpleNamespace(objectives=[])
        s.classify_batch_count = 1
        await engine._ensure_objectives(s)
        assert called["infer"] is False


class TestInferObjectives:
    @pytest.mark.asyncio
    async def test_infers_from_transcript(self, monkeypatch):
        s = SessionState()
        s.context = SimpleNamespace(
            full_transcript=[
                SimpleNamespace(speaker="A", text=f"line {i}") for i in range(12)
            ],
            objectives=[],
        )
        s.config = SimpleNamespace()

        async def _fake_llm(**kwargs):
            return '{"objectives": ["land the S3 PoC", "de-risk F64 sizing"]}'

        monkeypatch.setattr("sidekick.llm.call_llm", _fake_llm)
        await engine.infer_objectives(s)
        assert s.context.objectives == ["land the S3 PoC", "de-risk F64 sizing"]
        assert s.objectives_inferred is True

    @pytest.mark.asyncio
    async def test_noop_when_too_few_lines(self, monkeypatch):
        s = SessionState()
        s.context = SimpleNamespace(full_transcript=["x"] * 5, objectives=[])
        s.config = SimpleNamespace()
        await engine.infer_objectives(s)
        assert s.context.objectives == []
        assert s.objectives_inferred is False  # not enough context yet

    @pytest.mark.asyncio
    async def test_llm_failure_is_swallowed(self, monkeypatch):
        s = SessionState()
        s.context = SimpleNamespace(
            full_transcript=[
                SimpleNamespace(speaker="A", text=f"line {i}") for i in range(12)
            ],
            objectives=[],
        )
        s.config = SimpleNamespace()

        async def _boom(**kwargs):
            raise RuntimeError("fast tier down")

        monkeypatch.setattr("sidekick.llm.call_llm", _boom)
        await engine.infer_objectives(s)
        assert s.context.objectives == []
        assert s.objectives_inferred is True  # marked done so it won't retry
