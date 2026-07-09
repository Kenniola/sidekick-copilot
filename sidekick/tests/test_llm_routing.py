"""Tests for LLM tier routing, retry, and provider fallback (Phase 3).

The fallback/retry orchestration in ``call_llm`` is tested by faking the
provider functions in ``_PROVIDERS`` (fast, deterministic), and the per-provider
HTTP request shape is tested by faking the pooled httpx client. ``_BACKOFF_BASE``
is zeroed so retries don't sleep, and ``_get_gh_token`` is stubbed so no ``gh``
subprocess runs.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest

from sidekick import llm
from sidekick.config import ModelsConfig


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    """Zero the retry backoff and clear active models for every test."""
    monkeypatch.setattr(llm, "_BACKOFF_BASE", 0)
    monkeypatch.setattr(llm, "_active_models", None)


def _http_error(status: int) -> httpx.HTTPStatusError:
    req = httpx.Request("POST", "https://example.test/chat")
    resp = httpx.Response(status, request=req)
    return httpx.HTTPStatusError(f"status {status}", request=req, response=resp)


# ---------------------------------------------------------------------------
# Chain resolution / tier routing
# ---------------------------------------------------------------------------


class TestChainResolution:
    @pytest.mark.asyncio
    async def test_tier_uses_builtin_chain_when_no_active_models(self, monkeypatch):
        captured = {}

        async def cap(model, *a):
            captured["model"] = model
            return "ok"

        monkeypatch.setitem(llm._PROVIDERS, "copilot", cap)
        await llm.call_llm("s", "u", tier="fast")
        # _TIER_CONFIG["fast"][0] == ("copilot", "gpt-4o-mini")
        assert captured["model"] == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_active_models_override_builtin_chain(self, monkeypatch):
        captured = {}

        async def cap(model, *a):
            captured["model"] = model
            return "ok"

        monkeypatch.setattr(llm, "_active_models", ModelsConfig(fast=["copilot:my-model"]))
        monkeypatch.setitem(llm._PROVIDERS, "copilot", cap)
        await llm.call_llm("s", "u", tier="fast")
        assert captured["model"] == "my-model"

    @pytest.mark.asyncio
    async def test_explicit_chain_overrides_tier(self, monkeypatch):
        captured = {}

        async def cap(model, *a):
            captured["model"] = model
            return "ok"

        monkeypatch.setitem(llm._PROVIDERS, "copilot", cap)
        await llm.call_llm("s", "u", tier="deep", chain=[("copilot", "explicit")])
        assert captured["model"] == "explicit"

    @pytest.mark.asyncio
    async def test_unknown_provider_is_skipped(self, monkeypatch):
        async def ok(model, *a):
            return "from-copilot"

        monkeypatch.setitem(llm._PROVIDERS, "copilot", ok)
        out = await llm.call_llm(
            "s", "u", chain=[("ghost", "x"), ("copilot", "m")],
        )
        assert out == "from-copilot"


# ---------------------------------------------------------------------------
# Retry & fallback
# ---------------------------------------------------------------------------


class TestRetryAndFallback:
    @pytest.mark.asyncio
    async def test_retries_then_succeeds_on_429(self, monkeypatch):
        state = {"n": 0}

        async def flaky(model, *a):
            state["n"] += 1
            if state["n"] == 1:
                raise _http_error(429)
            return "recovered"

        monkeypatch.setitem(llm._PROVIDERS, "copilot", flaky)
        out = await llm.call_llm("s", "u", chain=[("copilot", "m")])
        assert out == "recovered"
        assert state["n"] == 2

    @pytest.mark.asyncio
    async def test_retries_then_succeeds_on_500(self, monkeypatch):
        state = {"n": 0}

        async def flaky(model, *a):
            state["n"] += 1
            if state["n"] < 3:
                raise _http_error(503)
            return "ok500"

        monkeypatch.setitem(llm._PROVIDERS, "copilot", flaky)
        out = await llm.call_llm("s", "u", chain=[("copilot", "m")])
        assert out == "ok500"
        assert state["n"] == 3

    @pytest.mark.asyncio
    async def test_connection_error_exhausts_retries_then_falls_back(self, monkeypatch):
        calls = []

        async def fail(model, *a):
            calls.append(("copilot", model))
            raise httpx.ConnectError("no route")

        async def ok(model, *a):
            calls.append(("gm", model))
            return "fallback-ok"

        monkeypatch.setitem(llm._PROVIDERS, "copilot", fail)
        monkeypatch.setitem(llm._PROVIDERS, "github_models", ok)
        out = await llm.call_llm(
            "s", "u", chain=[("copilot", "m1"), ("github_models", "m2")],
        )
        assert out == "fallback-ok"
        # First provider retried the full budget before fallback.
        assert sum(1 for c in calls if c[0] == "copilot") == llm._MAX_RETRIES

    @pytest.mark.asyncio
    async def test_non_retryable_4xx_skips_to_next_provider(self, monkeypatch):
        calls = []

        async def bad_request(model, *a):
            calls.append("copilot")
            raise _http_error(400)

        async def ok(model, *a):
            return "second-provider"

        monkeypatch.setitem(llm._PROVIDERS, "copilot", bad_request)
        monkeypatch.setitem(llm._PROVIDERS, "github_models", ok)
        out = await llm.call_llm(
            "s", "u", chain=[("copilot", "m1"), ("github_models", "m2")],
        )
        assert out == "second-provider"
        # 4xx (non-429) is not retried — single attempt then fall through.
        assert calls == ["copilot"]

    @pytest.mark.asyncio
    async def test_all_providers_fail_raises_runtime_error(self, monkeypatch):
        async def fail(model, *a):
            raise httpx.ConnectError("down")

        monkeypatch.setitem(llm._PROVIDERS, "copilot", fail)
        monkeypatch.setitem(llm._PROVIDERS, "github_models", fail)
        with pytest.raises(RuntimeError, match="All LLM providers failed"):
            await llm.call_llm(
                "s", "u", chain=[("copilot", "m1"), ("github_models", "m2")],
            )


# ---------------------------------------------------------------------------
# Provider HTTP request shape (mock httpx)
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, content="hello", status=200):
        self._content = content
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise _http_error(self.status_code)

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


class _FakeClient:
    def __init__(self, response):
        self._response = response
        self.posted = None

    async def post(self, url, headers=None, json=None, timeout=None):
        self.posted = {"url": url, "headers": headers, "json": json}
        return self._response


class TestProviderRequests:
    @pytest.mark.asyncio
    async def test_copilot_request_shape(self, monkeypatch):
        monkeypatch.setattr(llm, "_get_gh_token", AsyncMock(return_value="tok"))
        client = _FakeClient(_FakeResponse("copilot-content"))
        monkeypatch.setattr(llm, "_get_copilot_client", lambda timeout: client)

        out = await llm._call_copilot("gpt-4o-mini", "sys", "usr", False, 30.0)

        assert out == "copilot-content"
        assert client.posted["url"] == llm._COPILOT_URL
        assert client.posted["headers"]["Authorization"] == "Bearer tok"
        assert client.posted["headers"]["Copilot-Integration-Id"] == "vscode-chat"
        assert client.posted["json"]["model"] == "gpt-4o-mini"
        assert "response_format" not in client.posted["json"]

    @pytest.mark.asyncio
    async def test_copilot_json_output_sets_response_format(self, monkeypatch):
        monkeypatch.setattr(llm, "_get_gh_token", AsyncMock(return_value="tok"))
        client = _FakeClient(_FakeResponse())
        monkeypatch.setattr(llm, "_get_copilot_client", lambda timeout: client)

        await llm._call_copilot("m", "sys", "usr", True, 30.0)

        assert client.posted["json"]["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_github_models_request_shape(self, monkeypatch):
        monkeypatch.setattr(llm, "_get_gh_token", AsyncMock(return_value="tok"))
        client = _FakeClient(_FakeResponse("gm-content"))
        monkeypatch.setattr(llm, "_get_github_models_client", lambda timeout: client)

        out = await llm._call_github_models("gpt-4.1-mini", "sys", "usr", False, 30.0)

        assert out == "gm-content"
        assert client.posted["url"] == llm._GITHUB_MODELS_URL
        assert "Copilot-Integration-Id" not in client.posted["headers"]

    @pytest.mark.asyncio
    async def test_provider_raises_on_http_error(self, monkeypatch):
        monkeypatch.setattr(llm, "_get_gh_token", AsyncMock(return_value="tok"))
        client = _FakeClient(_FakeResponse(status=500))
        monkeypatch.setattr(llm, "_get_copilot_client", lambda timeout: client)

        with pytest.raises(httpx.HTTPStatusError):
            await llm._call_copilot("m", "sys", "usr", False, 30.0)
