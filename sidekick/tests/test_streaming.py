"""Tests for streaming synthesis (Phase 4c).

Covers ``llm.stream_llm`` provider routing / retry / fallback and the research
pipeline's streaming ``on_lead`` path. The provider functions in
``_STREAM_PROVIDERS`` are faked (async generators) for fast, deterministic
control, mirroring ``test_llm_routing``. ``_BACKOFF_BASE`` is zeroed so retries
don't sleep.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest

from sidekick import llm
from sidekick.actions import research as research_mod
from sidekick.actions.research import ResearchPipeline, _lead_answer


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    monkeypatch.setattr(llm, "_BACKOFF_BASE", 0)
    monkeypatch.setattr(llm, "_active_models", None)


def _http_error(status: int) -> httpx.HTTPStatusError:
    req = httpx.Request("POST", "https://example.test/chat")
    resp = httpx.Response(status, request=req)
    return httpx.HTTPStatusError(f"status {status}", request=req, response=resp)


def _fake_provider(*deltas):
    """Build an async-generator provider that yields the given deltas."""

    async def _gen(model, system_prompt, user_prompt, timeout):
        for d in deltas:
            yield d

    return _gen


def _failing_provider(exc, *, after=()):
    """Provider that yields ``after`` deltas then raises ``exc``."""

    async def _gen(model, system_prompt, user_prompt, timeout):
        for d in after:
            yield d
        raise exc

    return _gen


async def _collect(agen):
    return [d async for d in agen]


# ---------------------------------------------------------------------------
# stream_llm — chain resolution
# ---------------------------------------------------------------------------


class TestStreamChainResolution:
    @pytest.mark.asyncio
    async def test_uses_builtin_chain_when_no_active_models(self, monkeypatch):
        captured = {}

        async def cap(model, *a):
            captured["model"] = model
            yield "ok"

        monkeypatch.setitem(llm._STREAM_PROVIDERS, "copilot", cap)
        out = await _collect(llm.stream_llm("s", "u", tier="fast"))
        assert out == ["ok"]
        assert captured["model"] == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_explicit_chain_overrides_tier(self, monkeypatch):
        captured = {}

        async def cap(model, *a):
            captured["model"] = model
            yield "x"

        monkeypatch.setitem(llm._STREAM_PROVIDERS, "copilot", cap)
        await _collect(
            llm.stream_llm("s", "u", tier="deep", chain=[("copilot", "explicit")])
        )
        assert captured["model"] == "explicit"


# ---------------------------------------------------------------------------
# stream_llm — yielding, retry, fallback
# ---------------------------------------------------------------------------


class TestStreamLLM:
    @pytest.mark.asyncio
    async def test_yields_all_deltas_in_order(self, monkeypatch):
        monkeypatch.setitem(
            llm._STREAM_PROVIDERS, "copilot", _fake_provider("Hel", "lo ", "world")
        )
        out = await _collect(llm.stream_llm("s", "u", tier="fast"))
        assert "".join(out) == "Hello world"

    @pytest.mark.asyncio
    async def test_retries_then_succeeds_before_yielding(self, monkeypatch):
        calls = {"n": 0}

        async def flaky(model, system_prompt, user_prompt, timeout):
            calls["n"] += 1
            if calls["n"] == 1:
                raise _http_error(429)
                yield  # pragma: no cover
            yield "ok"

        monkeypatch.setitem(llm._STREAM_PROVIDERS, "copilot", flaky)
        out = await _collect(llm.stream_llm("s", "u", tier="fast"))
        assert out == ["ok"]
        assert calls["n"] == 2

    @pytest.mark.asyncio
    async def test_falls_back_to_next_provider_on_4xx(self, monkeypatch):
        monkeypatch.setitem(
            llm._STREAM_PROVIDERS, "copilot", _failing_provider(_http_error(400))
        )
        monkeypatch.setitem(
            llm._STREAM_PROVIDERS, "github_models", _fake_provider("from-gh")
        )
        out = await _collect(
            llm.stream_llm(
                "s", "u",
                chain=[("copilot", "a"), ("github_models", "b")],
            )
        )
        assert out == ["from-gh"]

    @pytest.mark.asyncio
    async def test_midstream_failure_after_yield_reraises(self, monkeypatch):
        # Provider yields one delta then fails — must NOT retry/fallback
        # (would duplicate already-emitted text); the error propagates.
        monkeypatch.setitem(
            llm._STREAM_PROVIDERS,
            "copilot",
            _failing_provider(_http_error(500), after=["partial"]),
        )
        monkeypatch.setitem(
            llm._STREAM_PROVIDERS, "github_models", _fake_provider("never")
        )
        got = []
        with pytest.raises(httpx.HTTPStatusError):
            async for d in llm.stream_llm(
                "s", "u", chain=[("copilot", "a"), ("github_models", "b")]
            ):
                got.append(d)
        assert got == ["partial"]

    @pytest.mark.asyncio
    async def test_all_providers_fail_raises_runtimeerror(self, monkeypatch):
        monkeypatch.setitem(
            llm._STREAM_PROVIDERS, "copilot", _failing_provider(_http_error(503))
        )
        with pytest.raises(RuntimeError, match="All LLM providers failed"):
            await _collect(llm.stream_llm("s", "u", chain=[("copilot", "a")]))


# ---------------------------------------------------------------------------
# research — lead extraction helper
# ---------------------------------------------------------------------------


class TestLeadAnswer:
    def test_strips_sources_block(self):
        text = "DirectLake is the default.\n\nSources:\n- https://x"
        assert _lead_answer(text) == "DirectLake is the default."

    def test_handles_singular_source(self):
        text = "Answer here.\nSource: https://x"
        assert _lead_answer(text) == "Answer here."

    def test_returns_full_text_when_no_sources(self):
        assert _lead_answer("Just an answer.") == "Just an answer."

    def test_strips_whitespace(self):
        assert _lead_answer("  padded  \n") == "padded"


# ---------------------------------------------------------------------------
# research — streaming synthesis path
# ---------------------------------------------------------------------------


def _agent():
    return ResearchPipeline.__new__(ResearchPipeline)


class TestSynthesiseStreaming:
    @pytest.mark.asyncio
    async def test_fires_on_lead_once_before_sources(self, monkeypatch):
        async def fake_stream(system_prompt, user_prompt, tier):
            for d in [
                "DirectLake loads Delta directly. ",
                "It avoids import refresh.\n",
                "Sources:\n- https://learn.microsoft.com",
            ]:
                yield d

        monkeypatch.setattr(research_mod, "stream_llm", fake_stream)
        leads = []
        full = await _agent()._synthesise_streaming(
            "sys", "usr", "deep", leads.append
        )
        assert len(leads) == 1
        assert "DirectLake loads Delta directly." in leads[0]
        assert "Sources" not in leads[0]
        assert "Sources:" in full

    @pytest.mark.asyncio
    async def test_fires_on_complete_lead_sentence_without_sources(self, monkeypatch):
        async def fake_stream(system_prompt, user_prompt, tier):
            yield "The answer is forty-two and that settles the matter. "
            yield "More detail follows here."

        monkeypatch.setattr(research_mod, "stream_llm", fake_stream)
        leads = []
        full = await _agent()._synthesise_streaming(
            "sys", "usr", "deep", leads.append
        )
        assert len(leads) == 1
        assert leads[0].startswith("The answer is forty-two")
        assert full.endswith("here.")

    @pytest.mark.asyncio
    async def test_falls_back_to_call_llm_on_stream_failure(self, monkeypatch):
        async def boom_stream(system_prompt, user_prompt, tier):
            raise RuntimeError("stream down")
            yield  # pragma: no cover

        monkeypatch.setattr(research_mod, "stream_llm", boom_stream)
        monkeypatch.setattr(
            research_mod,
            "call_llm",
            AsyncMock(return_value="Fallback answer.\nSources:\n- https://x"),
        )
        leads = []
        full = await _agent()._synthesise_streaming(
            "sys", "usr", "deep", leads.append
        )
        assert full.startswith("Fallback answer.")
        assert leads == ["Fallback answer."]

    @pytest.mark.asyncio
    async def test_on_lead_exception_does_not_break_synthesis(self, monkeypatch):
        async def fake_stream(system_prompt, user_prompt, tier):
            yield "A complete sentence that is plenty long enough here. "
            yield "tail"

        monkeypatch.setattr(research_mod, "stream_llm", fake_stream)

        def angry(_lead):
            raise ValueError("callback boom")

        full = await _agent()._synthesise_streaming("sys", "usr", "deep", angry)
        assert full.endswith("tail")
