"""Tests for CLI install/uninstall behaviour.

Covers the two install/uninstall fixes:
  * the MCP registration now pins ``SIDEKICK_WORKSPACE_ROOT`` so grounding and
    research resolve to the open workspace rather than the server's cwd; and
  * the uninstaller detects when it is running from *inside* the uv tool
    environment (the Windows self-lock that previously left a corrupted tool
    dir behind and printed a misleading "already removed").
"""

from __future__ import annotations

import json

from sidekick import cli


class TestRegisterMcpServer:
    def test_writes_workspace_root_env(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cli, "_get_vscode_user_settings_path", lambda: tmp_path)
        cli._register_mcp_server()
        entry = json.loads((tmp_path / "mcp.json").read_text(encoding="utf-8"))
        server = entry["servers"]["sidekick"]
        assert server["env"]["SIDEKICK_WORKSPACE_ROOT"] == "${workspaceFolder}"


class TestForceUtf8Output:
    def test_reconfigures_streams_to_utf8(self, monkeypatch):
        calls = []

        class _FakeStream:
            encoding = "cp1252"

            def reconfigure(self, **kwargs):
                calls.append(kwargs)

        monkeypatch.setattr(cli.sys, "stdout", _FakeStream())
        monkeypatch.setattr(cli.sys, "stderr", _FakeStream())
        cli._force_utf8_output()
        assert calls == [
            {"encoding": "utf-8", "errors": "replace"},
            {"encoding": "utf-8", "errors": "replace"},
        ]

    def test_survives_non_reconfigurable_stream(self, monkeypatch):
        class _Plain:
            pass  # no reconfigure() — e.g. a redirected buffer

        monkeypatch.setattr(cli.sys, "stdout", _Plain())
        monkeypatch.setattr(cli.sys, "stderr", _Plain())
        cli._force_utf8_output()  # must not raise


class TestRunningInsideUvTool:
    def test_true_when_executable_under_uv_tools(self, monkeypatch):
        monkeypatch.setattr(
            cli.sys,
            "executable",
            r"C:\Users\me\AppData\Roaming\uv\tools\sidekick-copilot\Scripts\python.exe",
        )
        assert cli._running_inside_uv_tool() is True

    def test_false_for_system_python(self, monkeypatch):
        monkeypatch.setattr(
            cli.sys, "executable", r"C:\Program Files\Python311\python.exe"
        )
        assert cli._running_inside_uv_tool() is False
