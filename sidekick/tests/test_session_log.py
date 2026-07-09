"""Tests for SessionLog recording and summary generation (Phase 3).

Only the in-memory behaviour is exercised (record / generate_summary /
format_outputs); disk-writing helpers are out of scope here.
"""

from __future__ import annotations

from types import SimpleNamespace

from sidekick.output.session_log import SessionLog


def _cfg(customer="Acme"):
    return SimpleNamespace(customer=customer, output=SimpleNamespace(auto_save=False))


def _result(action_type="research", question="Q", answer="A", sources=None,
            confidence="high"):
    return SimpleNamespace(
        action_type=action_type, question=question, answer=answer,
        sources=sources or [], confidence=confidence,
    )


class TestRecord:
    def test_record_appends_output(self):
        log = SessionLog(_cfg())
        log.record(_result(question="Throughput?"))
        assert len(log.outputs) == 1
        assert log.outputs[0]["question"] == "Throughput?"

    def test_record_uses_defaults_for_missing_attrs(self):
        log = SessionLog(_cfg())
        log.record(object())  # bare object — no attrs
        o = log.outputs[0]
        assert o["action_type"] == "unknown"
        assert o["confidence"] == "medium"
        assert o["question"] == ""


class TestGenerateSummary:
    def test_includes_customer_and_output_count(self):
        log = SessionLog(_cfg("Contoso"))
        log.record(_result(question="Throughput limit?"))
        s = log.generate_summary(context=None)
        assert "Contoso" in s
        assert "Outputs generated: 1" in s
        assert "Throughput limit?" in s

    def test_groups_outputs_by_type(self):
        log = SessionLog(_cfg())
        log.record(_result(action_type="research", question="r"))
        log.record(_result(action_type="prototype", question="p"))
        s = log.generate_summary(context=None)
        assert "[RESEARCH] (1)" in s
        assert "[PROTOTYPE] (1)" in s

    def test_lists_open_threads_from_context(self):
        ctx = SimpleNamespace(
            threads={"t": SimpleNamespace(status="open", topic="Latency budget")}
        )
        s = SessionLog(_cfg()).generate_summary(ctx)
        assert "OPEN THREADS" in s
        assert "Latency budget" in s

    def test_no_open_threads_section_when_all_closed(self):
        ctx = SimpleNamespace(
            threads={"t": SimpleNamespace(status="closed", topic="Done")}
        )
        s = SessionLog(_cfg()).generate_summary(ctx)
        assert "OPEN THREADS" not in s


class TestFormatOutputs:
    def test_empty_message(self):
        assert "none yet" in SessionLog(_cfg()).format_outputs()

    def test_lists_only_last_five(self):
        log = SessionLog(_cfg())
        for i in range(7):
            log.record(_result(question=f"Q{i}"))
        out = log.format_outputs()
        assert "Q6" in out
        assert "Q1" not in out  # only the most recent five are shown
