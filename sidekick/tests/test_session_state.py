"""Tests for the SessionState container (Phase 2d)."""

from sidekick.session_state import SessionState


class TestDefaults:
    def test_components_default_to_none(self):
        s = SessionState()
        assert s.config is None
        assert s.context is None
        assert s.analyst is None
        assert s.queue is None
        assert s.session_log is None
        assert s.research is None
        assert s.prototype is None
        assert s.audio_capture is None
        assert s.recogniser is None
        assert s.listen_task is None

    def test_counters_default(self):
        s = SessionState()
        assert s.last_error is None
        assert s.last_surface_output_count == 0
        assert s.last_surface_thread_count == 0
        assert s.classify_batch_count == 0
        assert s.domains_detected is False
        assert s.grounding_cache is None
        assert s.grounding_cache_time == 0.0


class TestReset:
    def test_reset_clears_counters_and_cache(self):
        s = SessionState()
        s.last_error = "boom"
        s.last_surface_output_count = 7
        s.last_surface_thread_count = 3
        s.classify_batch_count = 9
        s.domains_detected = True
        s.grounding_cache = "cached"
        s.grounding_cache_time = 123.4

        s.reset()

        assert s.last_error is None
        assert s.last_surface_output_count == 0
        assert s.last_surface_thread_count == 0
        assert s.classify_batch_count == 0
        assert s.domains_detected is False
        assert s.grounding_cache is None
        assert s.grounding_cache_time == 0.0

    def test_reset_preserves_components(self):
        s = SessionState()
        sentinel = object()
        s.config = sentinel
        s.analyst = sentinel
        s.reset()
        # reset clears per-session counters, not the wired-up components
        assert s.config is sentinel
        assert s.analyst is sentinel
