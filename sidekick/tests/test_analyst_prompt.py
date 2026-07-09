"""Tests for the analyst system-prompt builder.

``build_analyst_system_prompt`` returns the shared base prompt, optionally with
an appended block of per-engagement speech-to-text corrections drawn from
``config.stt_corrections`` so customer-specific jargon is un-mangled on top of
the built-in general examples.
"""

from __future__ import annotations

from types import SimpleNamespace

from sidekick.analyst.prompts import ANALYST_SYSTEM_PROMPT, build_analyst_system_prompt


class TestBuildAnalystSystemPrompt:
    def test_none_config_returns_base(self):
        assert build_analyst_system_prompt(None) == ANALYST_SYSTEM_PROMPT

    def test_no_corrections_returns_base_unchanged(self):
        cfg = SimpleNamespace(stt_corrections={})
        assert build_analyst_system_prompt(cfg) == ANALYST_SYSTEM_PROMPT

    def test_corrections_appended(self):
        cfg = SimpleNamespace(stt_corrections={"on lake": "OneLake"})
        prompt = build_analyst_system_prompt(cfg)
        assert prompt.startswith(ANALYST_SYSTEM_PROMPT)
        assert "ENGAGEMENT-SPECIFIC SPEECH-TO-TEXT CORRECTIONS" in prompt
        assert '"on lake" → "OneLake"' in prompt

    def test_multiple_corrections_each_rendered(self):
        cfg = SimpleNamespace(
            stt_corrections={"on lake": "OneLake", "data verse": "Dataverse"}
        )
        prompt = build_analyst_system_prompt(cfg)
        assert '"on lake" → "OneLake"' in prompt
        assert '"data verse" → "Dataverse"' in prompt


class TestPrecisionRules:
    """Phase 6 / 6.1: the base prompt must instruct against low-precision surfacing."""

    def test_excludes_consultant_own_words(self):
        assert "CONSULTANT'S OWN" in ANALYST_SYSTEM_PROMPT

    def test_excludes_statements_of_intent(self):
        assert "statements of intent" in ANALYST_SYSTEM_PROMPT

    def test_excludes_garbled_fragments(self):
        assert "garbled or incomplete fragments" in ANALYST_SYSTEM_PROMPT
