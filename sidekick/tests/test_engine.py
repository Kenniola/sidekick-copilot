"""Characterization tests for the classify/dispatch engine (Phase 2e).

Exercise the orchestration with mocked SessionState components so the
extraction from server.py is verifiably behaviour-preserving.
"""

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from sidekick import engine
from sidekick.session_state import SessionState


@dataclass
class _FakeResult:
    action_type: str = "research"
    question: str = "What is the throughput limit?"


def _make_state(action_items=None, results=None) -> SessionState:
    """Build a SessionState with mocked async components."""
    s = SessionState()
    s.config = MagicMock()
    s.config.domains = ["Microsoft Fabric"]
    # Default mode: the two-stage accuracy pipeline is OFF (a MagicMock would
    # otherwise make accuracy_mode truthy and change dispatch behaviour).
    s.config.sensitivity.accuracy_mode = False
    # Proactive advisor OFF by default (same MagicMock-truthiness trap).
    s.config.sensitivity.auto_suggest = False
    s.context = MagicMock()
    s.analyst = MagicMock()
    s.analyst.analyse_chunk = AsyncMock(return_value=action_items or [])
    s.queue = MagicMock()
    s.queue.enqueue = AsyncMock()
    s.queue.process_ready = AsyncMock(return_value=results or [])
    s.research = MagicMock()
    s.prototype = MagicMock()
    s.session_log = MagicMock()
    return s


class TestClassifyAndDispatch:
    @pytest.mark.asyncio
    async def test_increments_batch_count(self):
        s = _make_state()
        notify = MagicMock()
        await engine.classify_and_dispatch(s, ["line"], 0, notify)
        assert s.classify_batch_count == 1

    @pytest.mark.asyncio
    async def test_enqueues_each_action_item(self):
        items = ["a", "b", "c"]
        s = _make_state(action_items=items)
        notify = MagicMock()
        await engine.classify_and_dispatch(s, ["line"], 0, notify)
        assert s.queue.enqueue.await_count == 3

    @pytest.mark.asyncio
    async def test_records_and_notifies_each_result(self):
        results = [_FakeResult(), _FakeResult(action_type="prototype")]
        s = _make_state(results=results)
        notify = MagicMock()
        await engine.classify_and_dispatch(s, ["line"], 0, notify)
        assert s.session_log.record.call_count == 2
        assert notify.call_count == 2
        notify.assert_any_call(results[0])

    @pytest.mark.asyncio
    async def test_process_ready_receives_pipelines_and_domains(self):
        s = _make_state()
        notify = MagicMock()
        await engine.classify_and_dispatch(s, ["line"], 0, notify)
        _, kwargs = s.queue.process_ready.call_args
        assert kwargs["research"] is s.research
        assert kwargs["prototype"] is s.prototype
        assert kwargs["context"] is s.context
        assert kwargs["domains"] == ["Microsoft Fabric"]

    @pytest.mark.asyncio
    async def test_detect_domains_runs_on_third_batch(self, monkeypatch):
        s = _make_state()
        called = {"n": 0}

        async def fake_detect(state):
            called["n"] += 1

        monkeypatch.setattr(engine, "detect_domains", fake_detect)
        notify = MagicMock()
        for _ in range(2):
            await engine.classify_and_dispatch(s, ["line"], 0, notify)
        assert called["n"] == 0  # not yet
        await engine.classify_and_dispatch(s, ["line"], 0, notify)
        assert called["n"] == 1  # third batch triggers detection


class TestAccuracyMode:
    """Phase 1: candidates accumulate and flush through the deep adjudicator."""

    def _accuracy_state(self, action_items, *, objectives=("goal",)):
        s = _make_state(action_items=list(action_items))
        s.config.sensitivity = SimpleNamespace(
            accuracy_mode=True,
            adjudicator_interval_seconds=40,
            adjudicator_pause_flush=True,
            surface_threshold=0.7,
            max_surfaced_per_pass=3,
        )
        s.config.objectives = list(objectives)
        s.context = SimpleNamespace(objectives=list(objectives))
        return s

    @pytest.mark.asyncio
    async def test_accumulates_without_immediate_enqueue(self, monkeypatch):
        monkeypatch.delenv("SIDEKICK_ADJUDICATE_INTERVAL", raising=False)
        items = [
            SimpleNamespace(priority_score=0.5),
            SimpleNamespace(priority_score=0.6),
        ]
        s = self._accuracy_state(items)
        await engine.classify_and_dispatch(s, ["line"], 0, MagicMock())
        # Interval not elapsed and no critical candidate → buffered, not enqueued.
        assert s.queue.enqueue.await_count == 0
        assert s.pending_candidates == items

    @pytest.mark.asyncio
    async def test_flushes_on_interval_elapsed(self, monkeypatch):
        import time as _t

        monkeypatch.delenv("SIDEKICK_ADJUDICATE_INTERVAL", raising=False)
        surfaced = [SimpleNamespace(priority_score=0.9)]
        adj_mock = AsyncMock(return_value=surfaced)
        monkeypatch.setattr("sidekick.analyst.adjudicator.adjudicate", adj_mock)

        items = [SimpleNamespace(priority_score=0.5)]
        s = self._accuracy_state(items)
        s.last_adjudicate_time = _t.monotonic() - 100  # 40s interval elapsed

        await engine.classify_and_dispatch(s, ["line"], 0, MagicMock())
        adj_mock.assert_awaited_once()
        assert s.queue.enqueue.await_count == 1
        assert s.pending_candidates == []

    @pytest.mark.asyncio
    async def test_early_flush_on_critical_candidate(self, monkeypatch):
        monkeypatch.delenv("SIDEKICK_ADJUDICATE_INTERVAL", raising=False)
        adj_mock = AsyncMock(return_value=[SimpleNamespace(priority_score=0.95)])
        monkeypatch.setattr("sidekick.analyst.adjudicator.adjudicate", adj_mock)

        # Interval NOT elapsed, but a critical (>=0.9) candidate forces a flush.
        items = [SimpleNamespace(priority_score=0.95)]
        s = self._accuracy_state(items)

        await engine.classify_and_dispatch(s, ["line"], 0, MagicMock())
        adj_mock.assert_awaited_once()
        assert s.queue.enqueue.await_count == 1


class TestDetectDomains:
    @pytest.mark.asyncio
    async def test_noop_when_already_detected(self):
        s = SessionState()
        s.domains_detected = True
        s.context = MagicMock()
        s.config = MagicMock()
        await engine.detect_domains(s)
        # nothing to assert beyond no exception; flag stays set
        assert s.domains_detected is True

    @pytest.mark.asyncio
    async def test_noop_when_too_few_lines(self):
        s = SessionState()
        s.context = MagicMock()
        s.context.full_transcript = ["x"] * 5  # < 10
        s.config = MagicMock()
        s.config.domains = []
        await engine.detect_domains(s)
        # too little context: should not mark detected
        assert s.domains_detected is False

    @pytest.mark.asyncio
    async def test_merges_new_domains(self, monkeypatch):
        s = SessionState()
        s.context = MagicMock()
        s.context.full_transcript = [
            type("L", (), {"speaker": "A", "text": f"line {i}"})() for i in range(12)
        ]
        s.config = MagicMock()
        s.config.domains = ["Power BI"]

        async def fake_call_llm(**kwargs):
            return '{"domains": ["Power BI", "Cosmos DB"]}'

        import sidekick.llm as llm_mod

        monkeypatch.setattr(llm_mod, "call_llm", fake_call_llm)
        await engine.detect_domains(s)
        # only the genuinely new domain is appended; existing one not duplicated
        assert "Cosmos DB" in s.config.domains
        assert s.config.domains.count("Power BI") == 1
        assert s.context.detected_domains == ["Cosmos DB"]
        assert s.grounding_cache is None
        assert s.domains_detected is True


class TestInitialiseCapturePreroll:
    """5c: audio capture must begin (pre-roll) before the slow model load so
    the opening of the meeting buffers instead of being dropped."""

    class _FakeCapture:
        def __init__(self, calls, *a, **k):
            self._calls = calls
            self.speaker_label = k.get("speaker_label", "(audio)")
            self.capture_mode = k.get("capture_mode", "loopback")

        def begin(self):
            self._calls.append("begin")

        def stop(self):
            self._calls.append("stop")

        def list_devices(self):
            return []

    @pytest.mark.asyncio
    async def test_capture_begins_before_model_load(self, monkeypatch):
        s = _make_state()
        s.config.speech.capture_microphone = False
        calls: list[str] = []

        monkeypatch.setattr(
            "sidekick.transcript.audio_capture.AudioCapture",
            lambda *a, **k: self._FakeCapture(calls, *a, **k),
        )

        def _fake_create_recogniser(_speech):
            calls.append("load")
            return MagicMock()

        monkeypatch.setattr(
            "sidekick.transcript.speech_recogniser.create_recogniser",
            _fake_create_recogniser,
        )

        ok = await engine._initialise_capture(s)

        assert ok is True
        assert "begin" in calls and "load" in calls
        assert calls.index("begin") < calls.index("load")

    @pytest.mark.asyncio
    async def test_model_load_failure_stops_preroll_capture(self, monkeypatch):
        s = _make_state()
        s.config.speech.capture_microphone = False
        calls: list[str] = []

        monkeypatch.setattr(
            "sidekick.transcript.audio_capture.AudioCapture",
            lambda *a, **k: self._FakeCapture(calls, *a, **k),
        )

        def _boom(_speech):
            calls.append("load")
            raise RuntimeError("model boom")

        monkeypatch.setattr(
            "sidekick.transcript.speech_recogniser.create_recogniser", _boom
        )

        ok = await engine._initialise_capture(s)

        assert ok is False
        # The pre-roll capture we started must be torn down on failure.
        assert "stop" in calls


class TestAutoSuggest:
    """Proactive advisor pass (Phase 9.3)."""

    def _suggest_state(self) -> SessionState:
        s = _make_state()
        s.config.sensitivity.auto_suggest = True
        s.config.sensitivity.auto_suggest_interval_seconds = 120
        s.context = SimpleNamespace(
            full_transcript=[f"line {i}" for i in range(10)],
            objectives=["Assess Fabric readiness"],
        )
        s.recent_suggestions = []
        s.last_suggest_time = 0.0
        return s

    @pytest.mark.asyncio
    async def test_disabled_by_default_no_call(self, monkeypatch):
        s = self._suggest_state()
        s.config.sensitivity.auto_suggest = False
        called = False

        async def _gen(_s):
            nonlocal called
            called = True
            return {"question": "q", "rationale": "r", "impact": "high"}

        monkeypatch.setattr(engine, "_generate_suggestion", _gen)
        notify = MagicMock()
        await engine.maybe_auto_suggest(s, notify)
        assert notify.call_count == 0
        assert called is False

    @pytest.mark.asyncio
    async def test_first_pass_arms_timer_without_notifying(self, monkeypatch):
        s = self._suggest_state()
        monkeypatch.setattr(
            engine,
            "_generate_suggestion",
            AsyncMock(return_value={"question": "q", "rationale": "r", "impact": "high"}),
        )
        notify = MagicMock()
        await engine.maybe_auto_suggest(s, notify)
        assert notify.call_count == 0
        assert s.last_suggest_time > 0.0

    @pytest.mark.asyncio
    async def test_emits_one_suggestion_after_interval(self, monkeypatch):
        s = self._suggest_state()
        s.last_suggest_time = 1.0  # armed well in the past
        s.config.sensitivity.auto_suggest_interval_seconds = 0  # interval elapsed
        monkeypatch.setattr(
            engine,
            "_generate_suggestion",
            AsyncMock(
                return_value={
                    "question": "What is your data volume?",
                    "rationale": "Sizing needs it",
                    "impact": "high",
                }
            ),
        )
        notify = MagicMock()
        await engine.maybe_auto_suggest(s, notify)
        assert notify.call_count == 1
        result = notify.call_args.args[0]
        assert result.action_type == "suggestion"
        assert result.question == "What is your data volume?"
        assert result.priority == "high"
        assert "What is your data volume?" in s.recent_suggestions

    @pytest.mark.asyncio
    async def test_dedupes_repeat_suggestion(self, monkeypatch):
        s = self._suggest_state()
        s.last_suggest_time = 1.0
        s.config.sensitivity.auto_suggest_interval_seconds = 0
        s.recent_suggestions = ["What is your data volume?"]
        monkeypatch.setattr(
            engine,
            "_generate_suggestion",
            AsyncMock(
                return_value={
                    "question": "What is your data volume?",
                    "rationale": "r",
                    "impact": "high",
                }
            ),
        )
        notify = MagicMock()
        await engine.maybe_auto_suggest(s, notify)
        assert notify.call_count == 0

    @pytest.mark.asyncio
    async def test_too_little_transcript_skips(self, monkeypatch):
        s = self._suggest_state()
        s.context.full_transcript = ["only", "three", "lines"]
        s.last_suggest_time = 1.0
        s.config.sensitivity.auto_suggest_interval_seconds = 0
        gen = AsyncMock(
            return_value={"question": "q", "rationale": "r", "impact": "high"}
        )
        monkeypatch.setattr(engine, "_generate_suggestion", gen)
        notify = MagicMock()
        await engine.maybe_auto_suggest(s, notify)
        assert notify.call_count == 0
        assert gen.await_count == 0
