"""Tests for the extracted finding notifier (Phase 2b)."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

from sidekick.output import notifier


@dataclass
class _FakeResult:
    action_type: str = "research"
    question: str = "What is OneLake?"
    confidence: str = "high"
    priority: str = "high"
    answer: str = ""
    sources: tuple = ()
    rationale: str = ""


class TestPlaySound:
    def test_no_sound_device_does_not_raise(self, monkeypatch):
        # Force the win32 branch but make winsound import fail → swallowed.
        monkeypatch.setattr(sys, "platform", "win32")
        # No winsound on non-Windows test hosts; the broad except must absorb it.
        notifier.play_sound("chime")  # must not raise

    def test_silent_is_noop(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        notifier.play_sound("silent")  # must not raise

    def test_non_windows_is_noop(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        notifier.play_sound("chime")  # must not raise


class TestWriteAlert:
    def test_writes_jsonl_line(self, tmp_path):
        result = _FakeResult()
        notifier.write_alert(result, alerts_dir=tmp_path)

        alerts_file = tmp_path / "alerts.jsonl"
        assert alerts_file.exists()
        line = alerts_file.read_text(encoding="utf-8").strip()
        record = json.loads(line)
        assert record["type"] == "research"
        assert record["summary"] == "What is OneLake?"
        assert record["confidence"] == "high"
        assert record["priority"] == "high"
        assert "timestamp" in record

    def test_feed_fields_present(self, tmp_path):
        result = _FakeResult(
            action_type="sizing", question="F64 headroom?", rationale="ties to sizing goal"
        )
        notifier.write_alert(result, alerts_dir=tmp_path)
        record = json.loads(
            (tmp_path / "alerts.jsonl").read_text(encoding="utf-8").strip()
        )
        assert record["id"] == "sizing:F64 headroom?"
        assert record["rationale"] == "ties to sizing goal"
        assert record["thread_id"] == ""

    def test_full_answer_carried(self, tmp_path):
        result = _FakeResult(
            action_type="research",
            question="How to size?",
            answer="Start at F64 and monitor CU consumption over a rolling window.",
        )
        notifier.write_alert(result, alerts_dir=tmp_path)
        record = json.loads(
            (tmp_path / "alerts.jsonl").read_text(encoding="utf-8").strip()
        )
        assert record["answer_full"].startswith("Start at F64 and monitor CU")

    def test_appends_multiple(self, tmp_path):
        notifier.write_alert(_FakeResult(question="Q1"), alerts_dir=tmp_path)
        notifier.write_alert(_FakeResult(question="Q2"), alerts_dir=tmp_path)
        lines = (tmp_path / "alerts.jsonl").read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["summary"] == "Q1"
        assert json.loads(lines[1])["summary"] == "Q2"

    def test_missing_optional_fields_default(self, tmp_path):
        @dataclass
        class _Minimal:
            action_type: str = "prototype"
            question: str = "Q"

        notifier.write_alert(_Minimal(), alerts_dir=tmp_path)
        record = json.loads((tmp_path / "alerts.jsonl").read_text(encoding="utf-8").strip())
        assert record["confidence"] == "medium"
        assert record["priority"] == "medium"

    def test_creates_dir_if_missing(self, tmp_path):
        nested = tmp_path / "live"
        assert not nested.exists()
        notifier.write_alert(_FakeResult(), alerts_dir=nested)
        assert (nested / "alerts.jsonl").exists()

    def test_answer_and_source_carried_into_alert(self, tmp_path):
        result = _FakeResult(
            answer=(
                "OneLake is the single, unified, logical data lake for the "
                "whole tenant.\n\nSources [HIGH]:\n  \u2022 MS Learn \u2014 "
                "https://learn.microsoft.com/fabric/onelake/onelake-overview"
            ),
            sources=(
                "MS Learn \u2014 https://learn.microsoft.com/fabric/onelake/onelake-overview",
            ),
        )
        notifier.write_alert(result, alerts_dir=tmp_path)
        record = json.loads((tmp_path / "alerts.jsonl").read_text(encoding="utf-8").strip())
        assert record["answer"].startswith("OneLake is the single")
        assert "Sources" not in record["answer"]
        assert record["source"] == "https://learn.microsoft.com/fabric/onelake/onelake-overview"

    def test_answer_and_source_empty_when_absent(self, tmp_path):
        notifier.write_alert(_FakeResult(answer="", sources=()), alerts_dir=tmp_path)
        record = json.loads((tmp_path / "alerts.jsonl").read_text(encoding="utf-8").strip())
        assert record["answer"] == ""
        assert record["source"] == ""

    def test_source_falls_back_to_answer_url_when_sources_empty(self, tmp_path):
        # The direct research/prototype paths leave result.sources empty but
        # cite URLs inline in the answer — the toast's source must still resolve.
        result = _FakeResult(
            answer=(
                "Deploy the standard on-premises data gateway.\n\n"
                "Sources:\n  \u2022 MS Learn \u2014 "
                "https://learn.microsoft.com/power-bi/connect-data/service-gateway-onprem"
            ),
            sources=(),
        )
        notifier.write_alert(result, alerts_dir=tmp_path)
        record = json.loads((tmp_path / "alerts.jsonl").read_text(encoding="utf-8").strip())
        assert record["source"] == (
            "https://learn.microsoft.com/power-bi/connect-data/service-gateway-onprem"
        )

    def test_structured_sources_take_precedence_over_answer(self, tmp_path):
        result = _FakeResult(
            answer="Answer body. See http://inline.example/x for more.",
            sources=("Title \u2014 https://structured.example/canonical",),
        )
        notifier.write_alert(result, alerts_dir=tmp_path)
        record = json.loads((tmp_path / "alerts.jsonl").read_text(encoding="utf-8").strip())
        assert record["source"] == "https://structured.example/canonical"


class TestWriteDeliverablesAlert:
    def test_writes_deliverables_alert_with_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")  # skip sound
        notifier.write_deliverables_alert(
            "/home/u/.sidekick/outputs/acme/deliverables_20260616_101010.md",
            alerts_dir=tmp_path,
        )
        record = json.loads((tmp_path / "alerts.jsonl").read_text(encoding="utf-8").strip())
        assert record["type"] == "deliverables"
        assert record["summary"] == "Post-call deliverables ready"
        assert record["answer"] == "Saved to deliverables_20260616_101010.md"
        assert record["file"].endswith("deliverables_20260616_101010.md")
        assert record["priority"] == "high"

    def test_creates_dir_if_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        nested = tmp_path / "live"
        assert not nested.exists()
        notifier.write_deliverables_alert("/tmp/d.md", alerts_dir=nested)
        assert (nested / "alerts.jsonl").exists()


class TestOneLineAnswer:
    def test_strips_sources_section(self):
        r = _FakeResult(answer="Direct answer here.\nSources:\n  \u2022 x \u2014 http://a")
        assert notifier._one_line_answer(r) == "Direct answer here."

    def test_first_line_only(self):
        r = _FakeResult(answer="Lead line.\nSecond paragraph that is ignored.")
        assert notifier._one_line_answer(r) == "Lead line."

    def test_clips_long_answer_with_ellipsis(self):
        long = "word " * 60  # ~300 chars, single line
        r = _FakeResult(answer=long.strip())
        out = notifier._one_line_answer(r)
        assert len(out) <= notifier._ANSWER_MAX_CHARS + 1  # +1 for ellipsis
        assert out.endswith("\u2026")

    def test_empty_answer_returns_empty(self):
        assert notifier._one_line_answer(_FakeResult(answer="")) == ""


class TestFirstSourceUrl:
    def test_extracts_url_from_titled_source(self):
        r = _FakeResult(sources=("MS Learn \u2014 https://learn.microsoft.com/x",))
        assert notifier._first_source_url(r) == "https://learn.microsoft.com/x"

    def test_skips_sources_without_url(self):
        r = _FakeResult(sources=("Based on training knowledge", "Doc \u2014 http://b.com/p"))
        assert notifier._first_source_url(r) == "http://b.com/p"

    def test_no_url_returns_empty(self):
        assert notifier._first_source_url(_FakeResult(sources=("no url here",))) == ""

    def test_strips_trailing_punctuation(self):
        r = _FakeResult(sources=("see (https://learn.microsoft.com/x).",))
        assert notifier._first_source_url(r) == "https://learn.microsoft.com/x"


class TestNotify:
    def test_notify_writes_alert_and_does_not_raise(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")  # skip sound
        notifier.notify(_FakeResult(), sound="silent", alerts_dir=tmp_path)
        assert (tmp_path / "alerts.jsonl").exists()


class TestStableId:
    def test_plain_question(self):
        assert notifier._stable_id("research", "What is F64?") == "research:What is F64?"

    def test_strips_enriched_wrapper(self):
        q = "[ENRICHED] What is F64? (previous answer: it is X)"
        assert notifier._stable_id("research", q) == "research:What is F64?"

    def test_enriched_alert_matches_original_id(self, tmp_path):
        notifier.write_alert(
            _FakeResult(action_type="research", question="What is F64?"),
            alerts_dir=tmp_path,
        )
        notifier.write_alert(
            _FakeResult(
                action_type="research",
                question="[ENRICHED] What is F64? (previous answer: X)",
            ),
            alerts_dir=tmp_path,
        )
        lines = (
            (tmp_path / "alerts.jsonl").read_text(encoding="utf-8").strip().splitlines()
        )
        ids = [json.loads(ln)["id"] for ln in lines]
        assert ids[0] == ids[1] == "research:What is F64?"


class TestRotateAlerts:
    def test_archives_existing_file(self, tmp_path):
        (tmp_path / "alerts.jsonl").write_text('{"x":1}\n', encoding="utf-8")
        notifier.rotate_alerts(alerts_dir=tmp_path)
        assert not (tmp_path / "alerts.jsonl").exists()
        archived = list((tmp_path / "archive").glob("alerts_*.jsonl"))
        assert len(archived) == 1
        assert archived[0].read_text(encoding="utf-8") == '{"x":1}\n'

    def test_noop_when_no_file(self, tmp_path):
        notifier.rotate_alerts(alerts_dir=tmp_path)
        assert not (tmp_path / "archive").exists()

    def test_noop_when_empty_file(self, tmp_path):
        (tmp_path / "alerts.jsonl").write_text("", encoding="utf-8")
        notifier.rotate_alerts(alerts_dir=tmp_path)
        # Empty file is left as-is (nothing worth archiving).
        assert (tmp_path / "alerts.jsonl").exists()
        assert not (tmp_path / "archive").exists()
