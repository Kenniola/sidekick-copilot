"""Tests for config-driven LLM model chains (Phase 1)."""

from __future__ import annotations

import pytest

from sidekick.config import (
    ModelsConfig,
    _DEFAULT_MODEL_CHAINS,
    _parse_config,
    _parse_model_chain,
)


class TestParseModelChain:
    def test_provider_model_split(self):
        assert _parse_model_chain(["copilot:gpt-4o-mini"]) == [
            ("copilot", "gpt-4o-mini")
        ]

    def test_defaults_to_copilot_when_no_separator(self):
        assert _parse_model_chain(["gpt-4.1"]) == [("copilot", "gpt-4.1")]

    def test_skips_blank_entries(self):
        assert _parse_model_chain(["", "  ", "copilot:x"]) == [("copilot", "x")]

    def test_strips_whitespace(self):
        assert _parse_model_chain([" copilot : gpt-4o-mini "]) == [
            ("copilot", "gpt-4o-mini")
        ]

    def test_model_name_with_colon_preserved(self):
        # Only the first colon separates provider from model.
        assert _parse_model_chain(["github_models:org:model"]) == [
            ("github_models", "org:model")
        ]


class TestModelsConfigDefaults:
    def test_default_tiers_match_canonical_chains(self):
        cfg = ModelsConfig()
        assert cfg.fast == _DEFAULT_MODEL_CHAINS["fast"]
        assert cfg.standard == _DEFAULT_MODEL_CHAINS["standard"]
        assert cfg.deep == _DEFAULT_MODEL_CHAINS["deep"]

    def test_chain_resolves_tuples(self):
        cfg = ModelsConfig()
        assert cfg.chain("fast") == [
            ("copilot", "gpt-4o-mini"),
            ("github_models", "gpt-4.1-mini"),
        ]

    def test_unknown_tier_falls_back_to_standard(self):
        cfg = ModelsConfig()
        assert cfg.chain("nonexistent") == cfg.chain("standard")


class TestEnvOverride:
    def test_env_override_replaces_chain(self, monkeypatch):
        monkeypatch.setenv(
            "SIDEKICK_MODEL_DEEP", "copilot:claude-opus-4.8,copilot:gpt-4.1"
        )
        cfg = ModelsConfig()
        assert cfg.chain("deep") == [
            ("copilot", "claude-opus-4.8"),
            ("copilot", "gpt-4.1"),
        ]

    def test_env_override_does_not_affect_other_tiers(self, monkeypatch):
        monkeypatch.setenv("SIDEKICK_MODEL_FAST", "copilot:tiny")
        cfg = ModelsConfig()
        assert cfg.chain("fast") == [("copilot", "tiny")]
        assert cfg.chain("standard") == cfg.chain("standard")  # unchanged

    def test_blank_env_override_ignored(self, monkeypatch):
        monkeypatch.setenv("SIDEKICK_MODEL_FAST", "   ")
        cfg = ModelsConfig()
        assert cfg.chain("fast") == [
            ("copilot", "gpt-4o-mini"),
            ("github_models", "gpt-4.1-mini"),
        ]


class TestParseConfigModels:
    def test_missing_models_block_uses_defaults(self):
        cfg = _parse_config({})
        assert cfg.models.deep == _DEFAULT_MODEL_CHAINS["deep"]

    def test_yaml_models_override(self):
        raw = {
            "models": {
                "deep": ["copilot:claude-opus-4.8", "github_models:DeepSeek-R1"],
            }
        }
        cfg = _parse_config(raw)
        assert cfg.models.chain("deep") == [
            ("copilot", "claude-opus-4.8"),
            ("github_models", "DeepSeek-R1"),
        ]
        # Untouched tiers keep defaults.
        assert cfg.models.fast == _DEFAULT_MODEL_CHAINS["fast"]

    def test_empty_models_lists_fall_back_to_defaults(self):
        cfg = _parse_config({"models": {"fast": []}})
        assert cfg.models.fast == _DEFAULT_MODEL_CHAINS["fast"]


class TestLlmIntegration:
    @pytest.mark.asyncio
    async def test_set_active_models_drives_call_llm_chain(self, monkeypatch):
        from sidekick import llm

        captured: list[tuple[str, str]] = []

        async def fake_copilot(model, *a, **k):
            captured.append(("copilot", model))
            return "ok"

        monkeypatch.setitem(llm._PROVIDERS, "copilot", fake_copilot)
        try:
            llm.set_active_models(
                ModelsConfig(standard=["copilot:my-custom-model"])
            )
            result = await llm.call_llm("sys", "user", tier="standard")
            assert result == "ok"
            assert captured == [("copilot", "my-custom-model")]
        finally:
            llm.set_active_models(None)

    @pytest.mark.asyncio
    async def test_explicit_chain_overrides_active_models(self, monkeypatch):
        from sidekick import llm

        captured: list[tuple[str, str]] = []

        async def fake_copilot(model, *a, **k):
            captured.append(("copilot", model))
            return "ok"

        monkeypatch.setitem(llm._PROVIDERS, "copilot", fake_copilot)
        try:
            llm.set_active_models(ModelsConfig(standard=["copilot:configured"]))
            await llm.call_llm(
                "sys", "user", tier="standard", chain=[("copilot", "explicit")]
            )
            assert captured == [("copilot", "explicit")]
        finally:
            llm.set_active_models(None)
