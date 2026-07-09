"""Integration test for the ``stop`` tool's response bounding (regression).

A real post-call deliverables pack (LLM email + tables) runs to many KB. When
``stop`` inlined the whole thing, the chat host spilled the tool result to an
overflow file the agent couldn't read, so the deliverables never rendered.

These tests drive the *real* ``server.stop`` with a deliberately oversized
email and assert that:

  * the response is hard-bounded (never overflows);
  * the full email is NOT inlined, but the short actionable sections are;
  * a banner points at the saved file; and
  * the *full* pack is persisted to disk even when ``auto_save`` is off.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sidekick import server
from sidekick.output.deliverables import DeliverablesPack


def _wire_state(monkeypatch, tmp_path, *, big_email: str, auto_save: bool = False):
    """Set up ``server._state`` for a stop with a controlled deliverables pack."""
    pack = DeliverablesPack(
        customer="Contoso",
        email=big_email,
        actions=(
            "| # | Action | Owner | Due |\n"
            "|---|--------|-------|-----|\n"
            "| 1 | Send capacity sizing doc | Alex | Fri |"
        ),
        follow_up="- [ ] What is the egress cost?",
    )

    async def _fake_build(session_log, context, config, **kwargs):
        return pack

    # Replace the LLM-backed builder; save into a temp dir; silence the toast.
    monkeypatch.setattr(server, "build_deliverables", _fake_build)
    monkeypatch.setattr(
        "sidekick.output.deliverables.get_output_dir", lambda customer: tmp_path
    )
    monkeypatch.setattr(
        server.notifier, "write_deliverables_alert", lambda *a, **k: None
    )

    session_log = SimpleNamespace(
        outputs=[],
        generate_summary=lambda ctx: "Session summary line.\n" * 40,
        save_to_disk=lambda: None,
        save_transcript=lambda ctx: None,
        save_markdown_summary=lambda ctx: None,
    )
    context = SimpleNamespace(threads={})
    config = SimpleNamespace(
        customer="Contoso",
        output=SimpleNamespace(auto_save=auto_save),
        notifications=SimpleNamespace(sound="silent"),
    )

    server._state.session_log = session_log
    server._state.context = context
    server._state.config = config
    server._state.audio_capture = None
    server._state.listen_task = None
    server._state.recogniser = None
    server._state.last_surface_output_count = 0
    server._state.last_surface_thread_count = 0
    return pack


@pytest.mark.asyncio
async def test_stop_response_is_bounded_and_persists_full_pack(tmp_path, monkeypatch):
    # ~14 KB email — the exact size that used to overflow the tool-result buffer.
    big_email = ("Thanks for your time today. " * 500).strip()
    _wire_state(monkeypatch, tmp_path, big_email=big_email)

    resp = await server.stop(deliverables=True)

    # 1. Hard-bounded — always renders inline.
    assert len(resp) <= server._MAX_STOP_RESPONSE_CHARS
    # 2. The oversized email is NOT inlined in full...
    assert big_email not in resp
    # 3. ...but the short, actionable sections ARE.
    assert "Send capacity sizing doc" in resp
    assert "What is the egress cost?" in resp
    # 4. A banner points at the saved file.
    assert "Post-call deliverables saved" in resp
    # 5. The FULL pack is on disk (auto_save off ⇒ proves the force-save path),
    #    and it contains the complete email.
    saved = list(tmp_path.glob("deliverables_*.md"))
    assert len(saved) == 1
    disk = saved[0].read_text(encoding="utf-8")
    assert big_email in disk
    assert len(disk) > len(resp)


@pytest.mark.asyncio
async def test_stop_without_deliverables_writes_nothing(tmp_path, monkeypatch):
    _wire_state(monkeypatch, tmp_path, big_email="x")

    resp = await server.stop(deliverables=False)

    assert len(resp) <= server._MAX_STOP_RESPONSE_CHARS
    assert "Post-call deliverables saved" not in resp
    assert list(tmp_path.glob("deliverables_*.md")) == []


@pytest.mark.asyncio
async def test_stop_bounds_a_huge_session_summary(tmp_path, monkeypatch):
    # Even with a runaway summary, the response must stay bounded.
    _wire_state(monkeypatch, tmp_path, big_email="short")
    server._state.session_log.generate_summary = lambda ctx: "S" * 50_000

    resp = await server.stop(deliverables=True)

    assert len(resp) <= server._MAX_STOP_RESPONSE_CHARS
    # Banner sits at the front, so the saved-file pointer survives the clip.
    assert "Post-call deliverables saved" in resp
