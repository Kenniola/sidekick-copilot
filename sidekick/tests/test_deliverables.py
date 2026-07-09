"""Tests for the post-call deliverables generator (Phase 4b).

The email draft path is exercised with an injected fake ``llm_fn`` so no
network call is made; the action-item table and follow-up batch are pure
functions tested directly.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sidekick.output import deliverables


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #

def _cfg(customer="Acme", auto_save=False):
    return SimpleNamespace(customer=customer, output=SimpleNamespace(auto_save=auto_save))


def _thread(topic, status="open"):
    return SimpleNamespace(topic=topic, status=status)


def _context(action_items=None, open_questions=None, threads=None, key_facts=None):
    return SimpleNamespace(
        action_items=action_items or [],
        open_questions=open_questions or [],
        threads=threads or {},
        key_facts=key_facts or [],
    )


def _log(outputs=None):
    return SimpleNamespace(outputs=outputs or [])


async def _fake_llm(**kwargs):
    return "  Thanks for your time today.\n\nKind regards,\n[Your name]  "


async def _boom_llm(**kwargs):
    raise RuntimeError("model down")


# --------------------------------------------------------------------------- #
# Action-item table
# --------------------------------------------------------------------------- #

class TestActionItemTable:
    def test_empty_returns_placeholder(self):
        out = deliverables._action_item_table(_context())
        assert "No action items" in out

    def test_renders_rows_with_owner_and_due(self):
        ctx = _context(action_items=[
            {"description": "Send capacity sizing doc", "owner": "Alex", "due": "Fri"},
        ])
        out = deliverables._action_item_table(ctx)
        assert "| # | Action | Owner | Due |" in out
        assert "| 1 | Send capacity sizing doc | Alex | Fri |" in out

    def test_missing_owner_due_default_dash(self):
        ctx = _context(action_items=[{"description": "Follow up"}])
        out = deliverables._action_item_table(ctx)
        assert "| 1 | Follow up | \u2014 | \u2014 |" in out

    def test_non_dict_item_coerced(self):
        ctx = _context(action_items=["just a string"])
        out = deliverables._action_item_table(ctx)
        assert "| 1 | just a string | \u2014 | \u2014 |" in out

    def test_pipe_in_description_escaped_and_single_line(self):
        ctx = _context(action_items=[{"description": "a|b\nc"}])
        out = deliverables._action_item_table(ctx)
        assert "a\\|b c" in out
        # the rendered row must stay on one line
        row = [ln for ln in out.splitlines() if ln.startswith("| 1 ")][0]
        assert "\n" not in row


# --------------------------------------------------------------------------- #
# Follow-up research batch
# --------------------------------------------------------------------------- #

class TestUnansweredBatch:
    def test_nothing_outstanding(self):
        out = deliverables._unanswered_research_batch(_log(), _context())
        assert "nothing outstanding" in out.lower()

    def test_lists_unresearched_open_questions(self):
        ctx = _context(open_questions=[{"question": "What is the egress cost?"}])
        out = deliverables._unanswered_research_batch(_log(), ctx)
        assert "- [ ] What is the egress cost?" in out

    def test_excludes_questions_already_researched(self):
        ctx = _context(open_questions=[{"question": "What is OneLake?"}])
        log = _log(outputs=[{"question": "What is OneLake?", "action_type": "research"}])
        out = deliverables._unanswered_research_batch(log, ctx)
        assert "nothing outstanding" in out.lower()

    def test_lists_open_and_blocked_threads(self):
        ctx = _context(threads={
            "a": _thread("Capacity sizing", "open"),
            "b": _thread("Network egress", "blocked"),
            "c": _thread("Resolved topic", "closed"),
        })
        out = deliverables._unanswered_research_batch(_log(), ctx)
        assert "Capacity sizing" in out
        assert "Network egress" in out
        assert "Resolved topic" not in out


# --------------------------------------------------------------------------- #
# Email draft
# --------------------------------------------------------------------------- #

class TestDraftEmail:
    @pytest.mark.asyncio
    async def test_returns_stripped_llm_body(self):
        out = await deliverables._draft_email(_log(), _context(), _cfg(), _fake_llm)
        assert out.startswith("Thanks for your time")
        assert out.endswith("[Your name]")

    @pytest.mark.asyncio
    async def test_failure_degrades_gracefully(self):
        out = await deliverables._draft_email(_log(), _context(), _cfg(), _boom_llm)
        assert "unavailable" in out.lower()


class TestEmailContextBlock:
    def test_includes_customer_facts_and_research(self):
        ctx = _context(
            key_facts=["Tenant on F64", "Two regions"],
            threads={"a": _thread("Capacity")},
        )
        log = _log(outputs=[
            {"action_type": "research", "question": "F64 limit?", "answer": "It is X."},
        ])
        block = deliverables._email_context_block(log, ctx, _cfg("Contoso"))
        assert "Contoso" in block
        assert "Tenant on F64" in block
        assert "Capacity" in block
        assert "F64 limit?" in block


# --------------------------------------------------------------------------- #
# Full assembly
# --------------------------------------------------------------------------- #

class TestGenerateDeliverables:
    @pytest.mark.asyncio
    async def test_assembles_all_three_sections(self):
        ctx = _context(
            action_items=[{"description": "Send doc"}],
            open_questions=[{"question": "Egress cost?"}],
        )
        out = await deliverables.generate_deliverables(
            _log(), ctx, _cfg("Globex"), llm_fn=_fake_llm
        )
        assert "# Post-Call Deliverables \u2014 Globex" in out
        assert "## Draft Follow-up Email" in out
        assert "## Action Items" in out
        assert "## Follow-up Research Batch" in out
        assert "Send doc" in out
        assert "Egress cost?" in out

    @pytest.mark.asyncio
    async def test_returns_string_even_when_llm_fails(self):
        out = await deliverables.generate_deliverables(
            _log(), _context(), _cfg(), llm_fn=_boom_llm
        )
        assert isinstance(out, str)
        assert "unavailable" in out.lower()


class TestSaveDeliverables:
    def test_no_save_when_auto_save_off(self):
        assert deliverables.save_deliverables("x", _cfg(auto_save=False)) is None

    def test_writes_file_when_auto_save_on(self, tmp_path, monkeypatch):
        monkeypatch.setattr(deliverables, "get_output_dir", lambda customer: tmp_path)
        path = deliverables.save_deliverables("# hello", _cfg(auto_save=True))
        assert path is not None
        assert path.exists()
        assert path.read_text(encoding="utf-8") == "# hello"
        assert path.name.startswith("deliverables_")

    def test_force_writes_even_when_auto_save_off(self, tmp_path, monkeypatch):
        monkeypatch.setattr(deliverables, "get_output_dir", lambda customer: tmp_path)
        path = deliverables.save_deliverables(
            "# forced", _cfg(auto_save=False), force=True
        )
        assert path is not None
        assert path.exists()
        assert path.read_text(encoding="utf-8") == "# forced"


# --------------------------------------------------------------------------- #
# Pack + inline digest (keeps the stop response small enough to render)
# --------------------------------------------------------------------------- #

class TestBuildDeliverables:
    @pytest.mark.asyncio
    async def test_returns_pack_with_sections(self):
        ctx = _context(
            action_items=[{"description": "Send doc"}],
            open_questions=[{"question": "Egress cost?"}],
        )
        pack = await deliverables.build_deliverables(
            _log(), ctx, _cfg("Globex"), llm_fn=_fake_llm
        )
        assert pack.customer == "Globex"
        assert pack.email.startswith("Thanks for your time")
        assert "Send doc" in pack.actions
        assert "Egress cost?" in pack.follow_up

    @pytest.mark.asyncio
    async def test_full_markdown_matches_generate_deliverables(self):
        ctx = _context(action_items=[{"description": "Send doc"}])
        pack = await deliverables.build_deliverables(
            _log(), ctx, _cfg("Globex"), llm_fn=_fake_llm
        )
        legacy = await deliverables.generate_deliverables(
            _log(), ctx, _cfg("Globex"), llm_fn=_fake_llm
        )
        assert pack.full_markdown() == legacy


class TestInlineDigest:
    def _pack(self, email="Short email body."):
        return deliverables.DeliverablesPack(
            customer="Globex",
            email=email,
            actions="| 1 | Send doc | \u2014 | \u2014 |",
            follow_up="- [ ] Egress cost?",
        )

    def test_short_email_not_truncated(self):
        digest = self._pack().inline_digest("C:/out/deliverables_x.md")
        assert "## Draft Follow-up Email (preview)" in digest
        assert "Short email body." in digest
        assert "truncated" not in digest.lower()
        assert "Send doc" in digest
        assert "Egress cost?" in digest

    def test_long_email_truncated_with_pointer(self):
        long_email = "word " * 400  # ~2000 chars, over the preview budget
        digest = self._pack(email=long_email).inline_digest(
            "C:/out/deliverables_x.md"
        )
        assert "truncated" in digest.lower()
        assert "deliverables_x.md" in digest
        # Digest stays far smaller than the full email.
        assert len(digest) < len(long_email)

    def test_digest_is_smaller_than_full_markdown(self):
        long_email = "word " * 400
        pack = self._pack(email=long_email)
        assert len(pack.inline_digest("p.md")) < len(pack.full_markdown())


# --------------------------------------------------------------------------- #
# Follow-up hygiene (Phase 6 / 6.3)
# --------------------------------------------------------------------------- #

class TestCleanFollowups:
    def test_drops_trailing_conjunction_fragment(self):
        out = deliverables._clean_followups(
            ["We may not be as compartmental as the diagram suggested because."]
        )
        assert out == []

    def test_keeps_question_mark_item(self):
        q = "Is it an advantage to use GitHub instead of Azure DevOps?"
        assert deliverables._clean_followups([q]) == [q]

    def test_keeps_long_statement_question(self):
        q = "We need to know how the data is consumed and who the consumers are"
        assert deliverables._clean_followups([q]) == [q]

    def test_dedupes_case_insensitively(self):
        out = deliverables._clean_followups(["Is GitHub better?", "is github better?"])
        assert len(out) == 1

    def test_drops_short_non_question_fragment(self):
        assert deliverables._clean_followups(["Stage two of this."]) == []

    def test_caps_length(self):
        qs = [f"Is this valid question number {i} in the batch?" for i in range(20)]
        assert len(deliverables._clean_followups(qs, limit=5)) == 5


# --------------------------------------------------------------------------- #
# Action-item capture (Phase 6 / 6.2)
# --------------------------------------------------------------------------- #

class TestActionItemCapture:
    def _item(self, question, type_):
        from sidekick.analyst.classifier import ActionItem

        return ActionItem(
            question=question, type=type_, complexity="simple", priority="high"
        )

    def test_action_item_captured_into_context(self):
        from sidekick.analyst.context import MeetingContext

        ctx = MeetingContext()
        ctx.record_decisions(
            [
                self._item("Prepare two slides on strategic direction", "action_item"),
                self._item("What is F64 headroom?", "research"),
            ]
        )
        descs = [a["description"] for a in ctx.action_items]
        assert descs == ["Prepare two slides on strategic direction"]

    def test_action_items_deduped(self):
        from sidekick.analyst.context import MeetingContext

        ctx = MeetingContext()
        ctx.record_decisions([self._item("Schedule a workshop", "action_item")])
        ctx.record_decisions([self._item("Schedule a workshop", "action_item")])
        assert len(ctx.action_items) == 1

    def test_captured_items_render_in_deliverables_table(self):
        from sidekick.analyst.context import MeetingContext

        ctx = MeetingContext()
        ctx.record_decisions([self._item("Schedule a workshop", "action_item")])
        table = deliverables._action_item_table(ctx)
        assert "Schedule a workshop" in table
        assert "No action items" not in table

