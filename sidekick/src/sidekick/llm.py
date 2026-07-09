"""LLM client — tiered model routing with retry, backoff, and fallback.

Tiers map call-site intent to the right model:

  fast     → gpt-4o-mini via Copilot API    (classifier, quick decisions)
  standard → claude-sonnet-4.5 via Copilot  (research, synthesis, suggest)
  deep     → claude-opus-4.7 via Copilot    (complex research, prototypes)

Fallback: GitHub Models (gpt-4.1-mini, gpt-4.1) via gh auth token.

Auth:
  Both endpoints use the same GitHub token from `gh auth token`.
  The token is refreshed every 30 minutes to handle expiry during
  long-running meeting sessions.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from sidekick.config import ModelsConfig

logger = logging.getLogger(__name__)


def parse_llm_json(text: str):
    """Parse a JSON value from an LLM response, tolerating markdown fences.

    Models inconsistently wrap JSON. This unifies the handling previously
    duplicated across classifier, priority_queue, and server, accepting:

      - a bare object/array:           ``{"a": 1}``
      - a fenced block:                ``` ```json\\n{...}\\n``` ```
      - a bare fence:                  ``` ```\\n{...}\\n``` ```
      - leading/trailing whitespace
      - a stray ``json`` language tag: ``json\\n{...}``

    Returns the parsed value (typically a ``dict``). Raises
    ``json.JSONDecodeError`` if the cleaned text is not valid JSON.
    """
    cleaned = text.strip()

    # Strip a leading code fence (```json or ```), dropping the fence line.
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]

    # Strip a trailing code fence.
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    cleaned = cleaned.strip()

    # Some models still prefix the body with a bare "json" tag. Only strip it
    # when the remainder actually starts a JSON value, so legitimate content
    # is never corrupted (top-level JSON always starts with { or [).
    if cleaned[:4].lower() == "json":
        candidate = cleaned[4:].lstrip()
        if candidate[:1] in ("{", "["):
            cleaned = candidate

    return json.loads(cleaned)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

# Copilot API (primary) — uses Copilot Enterprise subscription
_COPILOT_URL = "https://api.githubcopilot.com/chat/completions"

# GitHub Models (fallback) — free tier with gh token
_GITHUB_MODELS_URL = "https://models.inference.ai.azure.com/chat/completions"

# Tier → (provider, model) fallback chains.
# These are the built-in code defaults used when no explicit chain is passed
# to call_llm (e.g. by tests or the vision path). The user-facing source of
# truth is config.ModelsConfig / configs/default.yaml — keep the two in sync.
_TIER_CONFIG: dict[str, list[tuple[str, str]]] = {
    "fast": [
        ("copilot", "gpt-4o-mini"),
        ("github_models", "gpt-4.1-mini"),
    ],
    "standard": [
        ("copilot", "claude-opus-4.8"),
        ("copilot", "claude-sonnet-4.5"),
        ("copilot", "gpt-4.1"),
        ("github_models", "gpt-4.1-mini"),
    ],
    "deep": [
        ("copilot", "claude-opus-4.8"),
        ("copilot", "claude-opus-4.7"),
        ("copilot", "claude-opus-4.6"),
        ("copilot", "gpt-4.1"),
        ("github_models", "DeepSeek-R1"),
    ],
}

# Retry settings
_MAX_RETRIES = 3
_BACKOFF_BASE = 1.0  # seconds: 1, 2, 4

# Active per-tier model config, registered at session init via
# set_active_models(). When set, call_llm resolves tier → chain from it
# (honouring YAML config + SIDEKICK_MODEL_<TIER> env overrides). When unset
# (tests, standalone use), call_llm falls back to _TIER_CONFIG.
_active_models: "ModelsConfig | None" = None


def set_active_models(models: "ModelsConfig | None") -> None:
    """Register the active model config so call_llm resolves chains from it."""
    global _active_models
    _active_models = models

# ---------------------------------------------------------------------------
# GitHub token management (gh auth token, refreshed periodically)
# ---------------------------------------------------------------------------

_gh_token: str = ""
_gh_token_acquired: float = 0.0
_GH_TOKEN_REFRESH_SECS = 1800  # refresh every 30 min


async def _get_gh_token() -> str:
    """Get a GitHub token, refreshing via `gh auth token` when stale.

    Priority:
      1. GITHUB_TOKEN env var (if set explicitly)
      2. `gh auth token` subprocess (uses keyring-cached credential)
    """
    global _gh_token, _gh_token_acquired

    # Env var takes priority (e.g. in CI)
    env_token = os.environ.get("GITHUB_TOKEN", "")
    if env_token:
        return env_token

    # Return cached token if fresh enough
    if _gh_token and (time.time() - _gh_token_acquired) < _GH_TOKEN_REFRESH_SECS:
        return _gh_token

    # Acquire via gh CLI
    import shutil
    import sys

    gh_cmd = "gh.exe" if sys.platform == "win32" else "gh"
    gh_path = shutil.which(gh_cmd) or shutil.which("gh")
    if not gh_path:
        raise RuntimeError(
            "Cannot acquire GitHub token: GITHUB_TOKEN not set and "
            "gh CLI not found on PATH."
        )

    proc = await asyncio.create_subprocess_exec(
        gh_path, "auth", "token",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"gh auth token failed: {stderr.decode().strip()}"
        )
    _gh_token = stdout.decode().strip()
    _gh_token_acquired = time.time()
    logger.debug("GitHub token acquired via gh auth token")
    return _gh_token


# ---------------------------------------------------------------------------
# Shared HTTP clients — reused across calls to avoid per-request TLS overhead
# ---------------------------------------------------------------------------

_copilot_client: httpx.AsyncClient | None = None
_github_models_client: httpx.AsyncClient | None = None


def _get_copilot_client(timeout: float) -> httpx.AsyncClient:
    """Return a shared httpx client for the Copilot API."""
    global _copilot_client
    if _copilot_client is None or _copilot_client.is_closed:
        _copilot_client = httpx.AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )
    return _copilot_client


def _get_github_models_client(timeout: float) -> httpx.AsyncClient:
    """Return a shared httpx client for the GitHub Models API."""
    global _github_models_client
    if _github_models_client is None or _github_models_client.is_closed:
        _github_models_client = httpx.AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )
    return _github_models_client


async def prewarm() -> None:
    """Best-effort warm-up of the primary LLM path.

    Acquires the GitHub token (amortises the `gh auth token` subprocess cost)
    and opens a pooled TLS connection to the Copilot host so the first real
    call skips DNS + TCP + TLS setup. All failures are swallowed — this is a
    latency optimisation, not a dependency.
    """
    try:
        await _get_gh_token()
    except Exception as e:  # noqa: BLE001 — best-effort only
        logger.debug("prewarm: token acquisition skipped (%s)", e)
        return

    try:
        client = _get_copilot_client(timeout=10.0)
        # A GET to the host establishes the keep-alive connection that the
        # POST /chat/completions call reuses. Status is irrelevant.
        await client.get("https://api.githubcopilot.com/", timeout=10.0)
        logger.debug("prewarm: Copilot connection established")
    except Exception as e:  # noqa: BLE001 — best-effort only
        logger.debug("prewarm: connection warm-up skipped (%s)", e)


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------


async def _call_copilot(
    model: str,
    system_prompt: str,
    user_prompt: str,
    json_output: bool,
    timeout: float,
) -> str:
    """Call via GitHub Copilot API (Enterprise subscription)."""
    token = await _get_gh_token()

    body: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if json_output:
        body["response_format"] = {"type": "json_object"}

    client = _get_copilot_client(timeout)
    resp = await client.post(
        _COPILOT_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Copilot-Integration-Id": "vscode-chat",
        },
        json=body,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


async def _call_github_models(
    model: str,
    system_prompt: str,
    user_prompt: str,
    json_output: bool,
    timeout: float,
) -> str:
    """Call via GitHub Models API (free tier, broader model catalog)."""
    token = await _get_gh_token()

    body: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if json_output:
        body["response_format"] = {"type": "json_object"}

    client = _get_github_models_client(timeout)
    resp = await client.post(
        _GITHUB_MODELS_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


_PROVIDERS = {
    "copilot": _call_copilot,
    "github_models": _call_github_models,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def call_llm(
    system_prompt: str,
    user_prompt: str,
    json_output: bool = False,
    timeout: float = 60.0,
    tier: str = "standard",
    chain: list[tuple[str, str]] | None = None,
) -> str:
    """Call an LLM with automatic tier routing, retry, and fallback.

    Tiers:
      - 'fast'     — gpt-4o-mini (classifier, quick decisions)
      - 'standard' — claude-sonnet-4.5 (research, synthesis, suggest)
      - 'deep'     — claude-opus-4.7 (complex research, prototypes)

    Each tier has a fallback chain: Copilot API → GitHub Models.

    Retry: up to 3 attempts per provider with exponential backoff (1s, 2s, 4s).
    On 429 or 5xx, retries the same provider then falls through to the next.

    Args:
        system_prompt: System-level instructions for the LLM.
        user_prompt: User-level prompt (the task to perform).
        json_output: If True, request JSON response format.
        timeout: Request timeout in seconds.
        tier: 'fast', 'standard', or 'deep'. Used to pick the built-in
            fallback chain when *chain* is not supplied.
        chain: Explicit ``[(provider, model), …]`` fallback chain, typically
            from ``config.models.chain(tier)``. Overrides *tier* when given,
            letting models be configured in YAML / env without code changes.

    Returns:
        The LLM's response text.
    """
    if chain is None:
        if _active_models is not None:
            chain = _active_models.chain(tier)
        else:
            chain = _TIER_CONFIG.get(tier, _TIER_CONFIG["standard"])
    last_error: Exception | None = None

    for provider_name, model in chain:
        provider_fn = _PROVIDERS.get(provider_name)
        if not provider_fn:
            continue

        for attempt in range(_MAX_RETRIES):
            try:
                logger.debug(
                    "LLM call: tier=%s provider=%s model=%s attempt=%d",
                    tier, provider_name, model, attempt + 1,
                )
                content = await provider_fn(
                    model, system_prompt, user_prompt, json_output, timeout
                )
                logger.debug("LLM response (%d chars)", len(content))
                return content

            except httpx.HTTPStatusError as e:
                last_error = e
                status = e.response.status_code
                if status == 429 or status >= 500:
                    wait = _BACKOFF_BASE * (2 ** attempt)
                    logger.warning(
                        "LLM %s/%s returned %d, retry in %.1fs (attempt %d/%d)",
                        provider_name, model, status, wait, attempt + 1, _MAX_RETRIES,
                    )
                    await asyncio.sleep(wait)
                    continue
                # 4xx other than 429 — don't retry, fall through
                logger.warning(
                    "LLM %s/%s returned %d, skipping to next provider",
                    provider_name, model, status,
                )
                break

            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_error = e
                wait = _BACKOFF_BASE * (2 ** attempt)
                logger.warning(
                    "LLM %s/%s connection error: %s, retry in %.1fs",
                    provider_name, model, type(e).__name__, wait,
                )
                await asyncio.sleep(wait)
                continue

            except Exception as e:
                last_error = e
                logger.warning(
                    "LLM %s/%s unexpected error: %s",
                    provider_name, model, e,
                )
                break

    raise RuntimeError(
        f"All LLM providers failed for tier={tier!r}. Last error: {last_error}"
    )


# ---------------------------------------------------------------------------
# Streaming API — yields content deltas for perceived-latency wins
# ---------------------------------------------------------------------------
#
# MCP stdio returns a single tool result, so streaming cannot push partial
# tokens into the Copilot Chat window. The value is on the *background*
# research path: stream_llm lets the engine surface a finding's lead answer
# (answer-card toast) as soon as it is generated, rather than waiting for the
# full synthesis + sources block. See SIDEKICK_ASSESSMENT.md §8b.


async def _stream_openai_compatible(client, url, headers, body, timeout):
    """Yield ``delta.content`` strings from an OpenAI-compatible SSE stream."""
    async with client.stream(
        "POST", url, headers=headers, json=body, timeout=timeout
    ) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            delta = (
                obj.get("choices", [{}])[0].get("delta", {}).get("content")
            )
            if delta:
                yield delta


async def _stream_copilot(model, system_prompt, user_prompt, timeout):
    """Stream via the GitHub Copilot API (Enterprise subscription)."""
    token = await _get_gh_token()
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": True,
    }
    client = _get_copilot_client(timeout)
    async for delta in _stream_openai_compatible(
        client,
        _COPILOT_URL,
        {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Copilot-Integration-Id": "vscode-chat",
        },
        body,
        timeout,
    ):
        yield delta


async def _stream_github_models(model, system_prompt, user_prompt, timeout):
    """Stream via the GitHub Models API (free tier)."""
    token = await _get_gh_token()
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": True,
    }
    client = _get_github_models_client(timeout)
    async for delta in _stream_openai_compatible(
        client,
        _GITHUB_MODELS_URL,
        {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        body,
        timeout,
    ):
        yield delta


_STREAM_PROVIDERS = {
    "copilot": _stream_copilot,
    "github_models": _stream_github_models,
}


async def stream_llm(
    system_prompt: str,
    user_prompt: str,
    timeout: float = 60.0,
    tier: str = "standard",
    chain: list[tuple[str, str]] | None = None,
):
    """Stream an LLM completion, yielding content deltas as they arrive.

    Mirrors :func:`call_llm`'s tier routing, retry, and provider fallback.
    Retry/fallback applies only *before* the first delta is emitted — once any
    output has been yielded, a mid-stream failure propagates (re-running a
    provider would duplicate already-yielded text). Callers that need a
    guaranteed full result should fall back to :func:`call_llm` on error.

    Yields:
        ``str`` content deltas in order.

    Raises:
        RuntimeError: if every provider fails before yielding any output.
    """
    if chain is None:
        if _active_models is not None:
            chain = _active_models.chain(tier)
        else:
            chain = _TIER_CONFIG.get(tier, _TIER_CONFIG["standard"])
    last_error: Exception | None = None

    for provider_name, model in chain:
        provider_fn = _STREAM_PROVIDERS.get(provider_name)
        if not provider_fn:
            continue

        for attempt in range(_MAX_RETRIES):
            yielded = False
            try:
                async for delta in provider_fn(
                    model, system_prompt, user_prompt, timeout
                ):
                    yielded = True
                    yield delta
                return

            except httpx.HTTPStatusError as e:
                last_error = e
                if yielded:
                    raise  # partial output emitted — don't risk duplication
                status = e.response.status_code
                if status == 429 or status >= 500:
                    await asyncio.sleep(_BACKOFF_BASE * (2 ** attempt))
                    continue
                break  # non-retryable 4xx → next provider

            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_error = e
                if yielded:
                    raise
                await asyncio.sleep(_BACKOFF_BASE * (2 ** attempt))
                continue

            except Exception as e:
                last_error = e
                if yielded:
                    raise
                break

    raise RuntimeError(
        f"All LLM providers failed (stream) for tier={tier!r}. "
        f"Last error: {last_error}"
    )


# ---------------------------------------------------------------------------
# Vision API — multimodal image analysis
# ---------------------------------------------------------------------------


async def call_llm_vision(
    system_prompt: str,
    user_prompt: str,
    image_b64: str,
    timeout: float = 30.0,
) -> str:
    """Send an image to a vision-capable model for analysis.

    Uses gpt-4o-mini on Copilot API (supports vision natively).
    Falls back to gpt-4o-mini on GitHub Models.

    Args:
        system_prompt: System-level instructions.
        user_prompt: Text prompt to accompany the image.
        image_b64: Base64-encoded PNG image.
        timeout: Request timeout in seconds.

    Returns:
        The model's description/analysis of the image.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_b64}",
                        "detail": "high",
                    },
                },
            ],
        },
    ]

    token = await _get_gh_token()
    body = {"model": "gpt-4o-mini", "messages": messages, "max_tokens": 2048}

    # Try Copilot API first
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                _COPILOT_URL,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                    "Copilot-Integration-Id": "vscode-chat",
                },
                json=body,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    except Exception as e:
        logger.warning("Copilot vision call failed: %s, trying GitHub Models", e)

    # Fallback: GitHub Models
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            _GITHUB_MODELS_URL,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json=body,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
