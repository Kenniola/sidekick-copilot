"""Tests for the STT benchmark decision logic (Phase 0 / C1).

Only the pure logic is exercised — model load, audio decode, and loopback
recording are injected/omitted so the suite runs offline in milliseconds.
"""

from __future__ import annotations

import pytest

from sidekick.transcript import benchmark as bm


class TestRealTimeFactor:
    def test_basic_ratio(self):
        assert bm.real_time_factor(5.0, 10.0) == 0.5

    def test_faster_than_realtime(self):
        assert bm.real_time_factor(3.0, 30.0) == pytest.approx(0.1)

    def test_zero_audio_is_inf(self):
        assert bm.real_time_factor(1.0, 0.0) == float("inf")

    def test_negative_audio_is_inf(self):
        assert bm.real_time_factor(1.0, -5.0) == float("inf")


class TestAccuracyRank:
    def test_known_order(self):
        assert bm.accuracy_rank("small.en") < bm.accuracy_rank("medium.en")
        assert bm.accuracy_rank("medium.en") < bm.accuracy_rank("distil-large-v3")
        assert bm.accuracy_rank("distil-large-v3") < bm.accuracy_rank("large-v3")

    def test_unknown_is_negative(self):
        assert bm.accuracy_rank("mystery-model") == -1


class TestRunBenchmark:
    def test_builds_results_with_rtf(self):
        def fake_fn(model, audio):
            timings = {"small.en": (1.0, 5.0), "medium.en": (2.0, 15.0)}
            load_s, tx_s = timings[model]
            return load_s, tx_s, f"transcript from {model}"

        results = bm.run_benchmark(
            audio=object(),
            audio_seconds=30.0,
            candidates=["small.en", "medium.en"],
            transcribe_fn=fake_fn,
        )
        assert [r.model for r in results] == ["small.en", "medium.en"]
        assert results[0].rtf == pytest.approx(5.0 / 30.0)
        assert results[1].rtf == pytest.approx(15.0 / 30.0)
        assert results[0].sample_text == "transcript from small.en"
        assert all(r.ok for r in results)

    def test_one_model_failing_is_captured_not_fatal(self):
        def fake_fn(model, audio):
            if model == "medium.en":
                raise RuntimeError("out of memory")
            return 1.0, 5.0, "ok"

        results = bm.run_benchmark(
            audio=object(),
            audio_seconds=30.0,
            candidates=["small.en", "medium.en"],
            transcribe_fn=fake_fn,
        )
        assert results[0].ok is True
        assert results[1].ok is False
        assert "out of memory" in results[1].error
        assert results[1].rtf is None

    def test_sample_text_is_clipped(self):
        def fake_fn(model, audio):
            return 1.0, 5.0, "x" * 500

        results = bm.run_benchmark(object(), 30.0, ["small.en"], fake_fn)
        assert len(results[0].sample_text) == 200


class TestRecommendModel:
    def _result(self, model, rtf, ok=True):
        return bm.BenchmarkResult(
            model=model,
            audio_seconds=30.0,
            transcribe_seconds=(rtf * 30.0 if rtf is not None else None),
            rtf=rtf,
            ok=ok,
        )

    def test_picks_most_accurate_passing(self):
        results = [
            self._result("small.en", 0.10),
            self._result("medium.en", 0.40),
            self._result("distil-large-v3", 0.60),
        ]
        assert bm.recommend_model(results, threshold=0.7) == "distil-large-v3"

    def test_excludes_models_over_threshold(self):
        results = [
            self._result("small.en", 0.10),
            self._result("medium.en", 0.40),
            self._result("distil-large-v3", 0.95),  # too slow
        ]
        assert bm.recommend_model(results, threshold=0.7) == "medium.en"

    def test_none_when_nothing_passes(self):
        results = [
            self._result("small.en", 0.80),
            self._result("distil-large-v3", 1.50),
        ]
        assert bm.recommend_model(results, threshold=0.7) is None

    def test_failed_results_are_ignored(self):
        results = [
            self._result("small.en", 0.10),
            self._result("distil-large-v3", None, ok=False),
        ]
        assert bm.recommend_model(results, threshold=0.7) == "small.en"

    def test_passes_helper(self):
        assert self._result("small.en", 0.5).passes(0.7) is True
        assert self._result("small.en", 0.8).passes(0.7) is False
