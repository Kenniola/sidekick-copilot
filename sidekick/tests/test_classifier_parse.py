"""Tests for the analyst response JSON parser (Phase 3).

``AnalystResponse.from_json`` turns the classifier LLM's (occasionally messy)
JSON into ActionItems, tolerating markdown fences, filtering unknown keys, and
skipping malformed items rather than crashing the loop.
"""

from __future__ import annotations

from sidekick.analyst.classifier import AnalystResponse


class TestAnalystResponseFromJson:
    def test_parses_basic_item(self):
        text = (
            '{"items":[{"question":"What is the CU limit?","type":"research",'
            '"complexity":"simple","priority":"high"}]}'
        )
        r = AnalystResponse.from_json(text)
        assert len(r.items) == 1
        assert r.items[0].question == "What is the CU limit?"
        assert r.items[0].priority == "high"

    def test_strips_json_code_fence(self):
        text = '```json\n{"items":[{"question":"Q","type":"research"}]}\n```'
        r = AnalystResponse.from_json(text)
        assert len(r.items) == 1

    def test_filters_unknown_llm_keys(self):
        text = '{"items":[{"question":"Q","type":"research","hallucinated":42}]}'
        r = AnalystResponse.from_json(text)
        assert len(r.items) == 1  # unknown key dropped, no crash

    def test_skips_item_missing_required_fields(self):
        text = (
            '{"items":[{"type":"research"},'
            '{"question":"Valid","type":"research"}]}'
        )
        r = AnalystResponse.from_json(text)
        assert len(r.items) == 1
        assert r.items[0].question == "Valid"

    def test_applies_complexity_and_priority_defaults(self):
        r = AnalystResponse.from_json('{"items":[{"question":"Q","type":"research"}]}')
        assert r.items[0].complexity == "medium"
        assert r.items[0].priority == "medium"

    def test_parses_phase_and_threads_update(self):
        text = '{"items":[],"phase":"wrapup","threads_update":[{"topic":"Latency"}]}'
        r = AnalystResponse.from_json(text)
        assert r.phase == "wrapup"
        assert r.threads_update == [{"topic": "Latency"}]

    def test_phase_defaults_to_core(self):
        assert AnalystResponse.from_json('{"items":[]}').phase == "core"

    def test_missing_items_key_yields_empty_list(self):
        assert AnalystResponse.from_json("{}").items == []
