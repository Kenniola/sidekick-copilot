"""Tests for config deep-merge and dict→dataclass parsing (Phase 3)."""

from __future__ import annotations

from sidekick.config import _deep_merge, _parse_config


class TestDeepMerge:
    def test_override_replaces_scalar(self):
        assert _deep_merge({"a": 1}, {"a": 2}) == {"a": 2}

    def test_adds_new_key(self):
        assert _deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    def test_nested_dicts_merge_recursively(self):
        base = {"queue": {"fast_lane_max": 3, "deep_lane_max": 1}}
        over = {"queue": {"fast_lane_max": 5}}
        assert _deep_merge(base, over) == {
            "queue": {"fast_lane_max": 5, "deep_lane_max": 1}
        }

    def test_lists_replaced_wholesale_not_concatenated(self):
        assert _deep_merge({"domains": ["a", "b"]}, {"domains": ["c"]}) == {
            "domains": ["c"]
        }

    def test_does_not_mutate_base(self):
        base = {"a": {"x": 1}}
        _deep_merge(base, {"a": {"y": 2}})
        assert base == {"a": {"x": 1}}


class TestParseConfig:
    def test_empty_dict_yields_defaults(self):
        c = _parse_config({})
        assert c.customer == "General"
        assert c.queue.fast_lane_max == 3
        assert c.speech.backend == "whisper"

    def test_flat_consultant_string_becomes_list(self):
        assert _parse_config({"consultant": "Alice"}).consultant_names == ["Alice"]

    def test_nested_participants_used_when_no_flat_key(self):
        c = _parse_config({"participants": {"consultant": ["Bob"]}})
        assert c.consultant_names == ["Bob"]

    def test_flat_consultant_overrides_nested(self):
        c = _parse_config(
            {"consultant": "Alice", "participants": {"consultant": ["Bob"]}}
        )
        assert c.consultant_names == ["Alice"]

    def test_legacy_azure_backend_normalised_to_whisper(self):
        assert _parse_config({"speech": {"backend": "azure"}}).speech.backend == "whisper"

    def test_capture_microphone_defaults_false(self):
        assert _parse_config({}).speech.capture_microphone is False

    def test_capture_microphone_enabled_from_yaml(self):
        c = _parse_config({"speech": {"capture_microphone": True}})
        assert c.speech.capture_microphone is True

    def test_capture_microphone_truthy_string_coerced(self):
        c = _parse_config({"speech": {"capture_microphone": "yes"}})
        assert c.speech.capture_microphone is True

    def test_glossary_defaults_empty(self):
        assert _parse_config({}).glossary == []

    def test_glossary_parsed_and_trimmed(self):
        c = _parse_config({"glossary": ["Northwind", "  OneLake  ", "", "   "]})
        assert c.glossary == ["Northwind", "OneLake"]

    def test_stt_corrections_defaults_empty(self):
        assert _parse_config({}).stt_corrections == {}

    def test_stt_corrections_parsed_and_filtered(self):
        c = _parse_config(
            {"stt_corrections": {"on lake": "OneLake", "": "x", "y": ""}}
        )
        assert c.stt_corrections == {"on lake": "OneLake"}

    def test_models_section_parsed(self):
        c = _parse_config({"models": {"fast": ["copilot:x"]}})
        assert c.models.fast == ["copilot:x"]

    def test_models_default_when_section_absent(self):
        c = _parse_config({})
        assert c.models.fast  # falls back to code default chain

    def test_notifications_sound_lowercased(self):
        assert _parse_config({"notifications": {"sound": "CHIME"}}).notifications.sound == "chime"

    def test_queue_overrides_applied(self):
        c = _parse_config({"queue": {"fast_lane_max": 9, "deep_lane_max": 4}})
        assert c.queue.fast_lane_max == 9
        assert c.queue.deep_lane_max == 4


class TestAccuracyPipelineConfig:
    """Phase 1 sensitivity + objectives config."""

    def test_defaults(self):
        c = _parse_config({})
        s = c.sensitivity
        assert s.accuracy_mode is False
        assert s.adjudicator_interval_seconds == 40
        assert s.adjudicator_pause_flush is True
        assert s.max_surfaced_per_pass == 3
        assert s.surface_threshold == 0.7
        assert s.answer_tier == "auto"
        assert s.self_critique is False
        assert c.objectives == []

    def test_sensitivity_overrides_parsed(self):
        c = _parse_config(
            {
                "sensitivity": {
                    "accuracy_mode": True,
                    "adjudicator_interval_seconds": 25,
                    "adjudicator_pause_flush": False,
                    "max_surfaced_per_pass": 5,
                    "surface_threshold": 0.8,
                    "answer_tier": "DEEP",
                    "self_critique": True,
                }
            }
        )
        s = c.sensitivity
        assert s.accuracy_mode is True
        assert s.adjudicator_interval_seconds == 25
        assert s.adjudicator_pause_flush is False
        assert s.max_surfaced_per_pass == 5
        assert s.surface_threshold == 0.8
        assert s.answer_tier == "deep"  # lower-cased
        assert s.self_critique is True

    def test_objectives_trimmed_and_filtered(self):
        c = _parse_config({"objectives": [" land the S3 PoC ", "", "  ", "de-risk F64"]})
        assert c.objectives == ["land the S3 PoC", "de-risk F64"]

    def test_accuracy_mode_env_override(self, monkeypatch):
        monkeypatch.setenv("SIDEKICK_ACCURACY_MODE", "true")
        assert _parse_config({}).sensitivity.accuracy_mode is True


class TestSpeechDecodeConfig:
    """Phase 2 speech: chunking + VAD/decode thresholds + echo suppression."""

    def test_defaults(self):
        sp = _parse_config({}).speech
        assert sp.chunk_seconds == 5.0
        assert sp.vad_min_silence_ms == 500
        assert sp.no_speech_threshold == 0.6
        assert sp.log_prob_threshold == -1.0
        assert sp.compression_ratio_threshold == 2.4
        assert sp.echo_suppression is True
        assert sp.speaker_naming is True

    def test_overrides_parsed(self):
        sp = _parse_config(
            {
                "speech": {
                    "chunk_seconds": 8,
                    "vad_min_silence_ms": 300,
                    "no_speech_threshold": 0.5,
                    "log_prob_threshold": -0.8,
                    "compression_ratio_threshold": 2.0,
                    "echo_suppression": False,
                    "speaker_naming": False,
                }
            }
        ).speech
        assert sp.chunk_seconds == 8.0
        assert sp.vad_min_silence_ms == 300
        assert sp.no_speech_threshold == 0.5
        assert sp.log_prob_threshold == -0.8
        assert sp.compression_ratio_threshold == 2.0
        assert sp.echo_suppression is False
        assert sp.speaker_naming is False


