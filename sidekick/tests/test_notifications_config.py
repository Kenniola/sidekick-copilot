"""Regression tests for the config-driven notification sound (v0.3.0)."""

from __future__ import annotations

from sidekick.config import NotificationsConfig, _parse_config


class TestNotificationsConfigDefaults:
    def test_default_sound_is_chime(self):
        assert NotificationsConfig().sound == "chime"

    def test_default_when_block_missing(self):
        cfg = _parse_config({})
        assert cfg.notifications.sound == "chime"


class TestNotificationsSoundParsing:
    def test_silent(self):
        cfg = _parse_config({"notifications": {"sound": "silent"}})
        assert cfg.notifications.sound == "silent"

    def test_chime(self):
        cfg = _parse_config({"notifications": {"sound": "chime"}})
        assert cfg.notifications.sound == "chime"

    def test_asterisk(self):
        cfg = _parse_config({"notifications": {"sound": "asterisk"}})
        assert cfg.notifications.sound == "asterisk"

    def test_exclamation(self):
        cfg = _parse_config({"notifications": {"sound": "exclamation"}})
        assert cfg.notifications.sound == "exclamation"

    def test_beep_legacy(self):
        cfg = _parse_config({"notifications": {"sound": "beep"}})
        assert cfg.notifications.sound == "beep"

    def test_uppercase_normalised_to_lower(self):
        cfg = _parse_config({"notifications": {"sound": "SILENT"}})
        assert cfg.notifications.sound == "silent"

    def test_mixed_case_normalised(self):
        cfg = _parse_config({"notifications": {"sound": "Chime"}})
        assert cfg.notifications.sound == "chime"

    def test_unknown_value_passed_through(self):
        # Runtime in server._notify falls back to MB_OK for unknown values;
        # the parser does not validate the vocabulary so future values can
        # be added without code changes.
        cfg = _parse_config({"notifications": {"sound": "future-sound"}})
        assert cfg.notifications.sound == "future-sound"


class TestNotificationsBackwardCompat:
    def test_old_config_without_notifications_block_still_works(self):
        """Configs predating v0.3.0 must not break."""
        raw = {
            "customer": "Legacy Customer",
            "speech": {"backend": "whisper"},
        }
        cfg = _parse_config(raw)
        assert cfg.notifications.sound == "chime"
        assert cfg.customer == "Legacy Customer"
