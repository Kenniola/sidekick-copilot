"""Tests for the `sidekick init --stt-model` persistence helpers (Phase 0 / C1)."""

from __future__ import annotations

from sidekick import cli


class TestArgValue:
    def test_space_separated(self, monkeypatch):
        monkeypatch.setattr(cli.sys, "argv", ["sidekick", "init", "--stt-model", "medium.en"])
        assert cli._arg_value("--stt-model") == "medium.en"

    def test_equals_form(self, monkeypatch):
        monkeypatch.setattr(cli.sys, "argv", ["sidekick", "init", "--stt-model=large-v3"])
        assert cli._arg_value("--stt-model") == "large-v3"

    def test_absent_returns_none(self, monkeypatch):
        monkeypatch.setattr(cli.sys, "argv", ["sidekick", "init"])
        assert cli._arg_value("--stt-model") is None

    def test_flag_without_value_returns_none(self, monkeypatch):
        monkeypatch.setattr(cli.sys, "argv", ["sidekick", "init", "--stt-model"])
        assert cli._arg_value("--stt-model") is None


class TestSetEnvSttModel:
    def test_appends_when_absent(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("GITHUB_TOKEN=abc\n", encoding="utf-8")
        cli._set_env_stt_model(env, "distil-large-v3")
        text = env.read_text(encoding="utf-8")
        assert "SIDEKICK_WHISPER_MODEL=distil-large-v3" in text
        assert "GITHUB_TOKEN=abc" in text

    def test_replaces_commented_default(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text(
            "# SIDEKICK_WHISPER_MODEL=small.en\n# SIDEKICK_WHISPER_COMPUTE=int8\n",
            encoding="utf-8",
        )
        cli._set_env_stt_model(env, "medium.en")
        lines = env.read_text(encoding="utf-8").splitlines()
        assert "SIDEKICK_WHISPER_MODEL=medium.en" in lines
        # The commented model line is gone; the compute comment is untouched.
        assert "# SIDEKICK_WHISPER_MODEL=small.en" not in lines
        assert "# SIDEKICK_WHISPER_COMPUTE=int8" in lines

    def test_is_idempotent_and_dedupes(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("SIDEKICK_WHISPER_MODEL=small.en\n", encoding="utf-8")
        cli._set_env_stt_model(env, "distil-large-v3")
        cli._set_env_stt_model(env, "distil-large-v3")
        occurrences = env.read_text(encoding="utf-8").count("SIDEKICK_WHISPER_MODEL=")
        assert occurrences == 1
        assert "SIDEKICK_WHISPER_MODEL=distil-large-v3" in env.read_text(encoding="utf-8")

    def test_creates_file_when_missing(self, tmp_path):
        env = tmp_path / ".env"
        cli._set_env_stt_model(env, "base.en")
        assert env.exists()
        assert env.read_text(encoding="utf-8").strip() == "SIDEKICK_WHISPER_MODEL=base.en"


class TestApplySttModel:
    def test_warns_on_unknown_model(self, tmp_path, capsys):
        env = tmp_path / ".env"
        cli._apply_stt_model(env, "totally-made-up")
        out = capsys.readouterr().out
        assert "not a recognised Whisper model" in out
        assert "SIDEKICK_WHISPER_MODEL=totally-made-up" in env.read_text(encoding="utf-8")

    def test_no_warning_for_known_model(self, tmp_path, capsys):
        env = tmp_path / ".env"
        cli._apply_stt_model(env, "distil-large-v3")
        out = capsys.readouterr().out
        assert "not a recognised" not in out
