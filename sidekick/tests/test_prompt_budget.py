"""Tests for the prompt token-budget helper (Phase 2f)."""

from sidekick.prompt_budget import clip


class TestWithinBudget:
    def test_short_text_unchanged(self):
        assert clip("hello", 100) == "hello"

    def test_exact_length_unchanged(self):
        text = "x" * 50
        assert clip(text, 50) == text

    def test_empty_and_none_passthrough(self):
        assert clip("", 10) == ""
        assert clip(None, 10) is None


class TestClipTail:
    def test_keeps_end_of_text(self):
        text = "\n".join(f"line{i}" for i in range(100))
        out = clip(text, 60, keep="tail")
        assert len(out) == 60
        assert out.startswith("… [truncated]")
        # the most recent content survives
        assert "line99" in out
        assert "line0\n" not in out

    def test_default_keep_is_tail(self):
        text = "abc" * 100
        out = clip(text, 30)
        assert out.startswith("… [truncated]")


class TestClipHead:
    def test_keeps_start_of_text(self):
        text = "\n".join(f"line{i}" for i in range(100))
        out = clip(text, 60, keep="head")
        assert len(out) == 60
        assert out.endswith("… [truncated]")
        assert "line0" in out
        assert "line99" not in out


class TestTinyBudget:
    def test_budget_smaller_than_marker_hard_cuts(self):
        text = "x" * 100
        out = clip(text, 5)
        assert out == "xxxxx"
