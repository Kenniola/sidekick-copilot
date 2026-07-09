"""Tests for the unified LLM JSON parser (Phase 2a).

Pins the behaviour that was previously duplicated across classifier.py,
priority_queue.py, and server.py (detect_domains + suggest_questions).
"""

from __future__ import annotations

import json

import pytest

from sidekick.llm import parse_llm_json


class TestPlainJson:
    def test_bare_object(self):
        assert parse_llm_json('{"a": 1}') == {"a": 1}

    def test_object_with_surrounding_whitespace(self):
        assert parse_llm_json('  \n {"a": 1}\n  ') == {"a": 1}

    def test_array_top_level(self):
        assert parse_llm_json('[1, 2, 3]') == [1, 2, 3]

    def test_nested(self):
        text = '{"items": [{"q": "x"}], "phase": "core"}'
        assert parse_llm_json(text) == {"items": [{"q": "x"}], "phase": "core"}


class TestFencedJson:
    def test_json_fence(self):
        text = '```json\n{"a": 1}\n```'
        assert parse_llm_json(text) == {"a": 1}

    def test_bare_fence(self):
        text = '```\n{"a": 1}\n```'
        assert parse_llm_json(text) == {"a": 1}

    def test_fence_with_leading_trailing_whitespace(self):
        text = '\n\n```json\n{"a": 1}\n```\n\n'
        assert parse_llm_json(text) == {"a": 1}

    def test_fence_no_newline_after_marker(self):
        # Defensive: "```{...}" with no newline — old code did cleaned[3:].
        text = '```{"a": 1}```'
        assert parse_llm_json(text) == {"a": 1}

    def test_multiline_body_inside_fence(self):
        text = '```json\n{\n  "a": 1,\n  "b": 2\n}\n```'
        assert parse_llm_json(text) == {"a": 1, "b": 2}


class TestJsonPrefixTag:
    def test_bare_json_prefix(self):
        # The old `.lstrip("json\n")` idiom handled a stray "json" tag.
        text = 'json\n{"match_index": 2}'
        assert parse_llm_json(text) == {"match_index": 2}

    def test_json_prefix_no_newline(self):
        text = 'json {"match_index": 0}'
        assert parse_llm_json(text) == {"match_index": 0}


class TestInvalid:
    def test_non_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            parse_llm_json("not json at all")

    def test_empty_raises(self):
        with pytest.raises(json.JSONDecodeError):
            parse_llm_json("")
