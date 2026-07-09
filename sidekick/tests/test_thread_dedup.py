"""Tests for topic-based thread deduplication in the transcript analyst.

The analyst LLM frequently mints a fresh ``thread_id`` for a topic it already
opened in an earlier chunk. Keying threads by id alone therefore produced a
duplicate thread on every mention (e.g. "Integration of GitHub with Azure
DevOps" appearing three times). ``_find_thread_by_topic`` resolves an unknown
id to an existing thread with a near-duplicate topic so the thread is updated
in place instead of duplicated.
"""

from __future__ import annotations

from sidekick.analyst.classifier import TranscriptAnalyst
from sidekick.analyst.context import MeetingContext, TopicThread


def _analyst_with_threads(*topics: str) -> TranscriptAnalyst:
    ctx = MeetingContext()
    for i, topic in enumerate(topics):
        tid = f"t{i}"
        ctx.threads[tid] = TopicThread(
            thread_id=tid, topic=topic, started_at="", last_active_at="",
        )
    return TranscriptAnalyst(config=None, context=ctx)


class TestFindThreadByTopic:
    def test_identical_topic_matches(self):
        a = _analyst_with_threads("Integration of GitHub with Azure DevOps")
        assert (
            a._find_thread_by_topic("Integration of GitHub with Azure DevOps")
            == "t0"
        )

    def test_near_duplicate_topic_matches(self):
        a = _analyst_with_threads("Integration of GitHub with Azure DevOps")
        # Minor rewording the LLM might emit for the same theme.
        assert (
            a._find_thread_by_topic("Integration of GitHub and Azure DevOps")
            == "t0"
        )

    def test_distinct_topic_does_not_match(self):
        a = _analyst_with_threads("Integration of GitHub with Azure DevOps")
        assert a._find_thread_by_topic("Capacity sizing for F64 SKU") is None

    def test_empty_topic_returns_none(self):
        a = _analyst_with_threads("Some topic")
        assert a._find_thread_by_topic("") is None

    def test_no_threads_returns_none(self):
        a = _analyst_with_threads()
        assert a._find_thread_by_topic("Anything") is None

    def test_picks_best_among_several(self):
        a = _analyst_with_threads(
            "Capacity sizing for F64 SKU",
            "Integration of GitHub with Azure DevOps",
            "Power BI report performance",
        )
        assert (
            a._find_thread_by_topic("Integration of GitHub with Azure DevOps")
            == "t1"
        )


class TestThreadUpdateDedup:
    """End-to-end: applying the same topic under a new id must not duplicate."""

    def _apply(self, analyst: TranscriptAnalyst, thread_data: dict) -> None:
        # Mirror the resolution logic in analyse_chunk's thread-update loop.
        tid = thread_data.get("thread_id", "")
        if not tid:
            return
        if tid not in analyst.context.threads:
            match = analyst._find_thread_by_topic(thread_data.get("topic", tid))
            if match is not None:
                tid = match
        if tid in analyst.context.threads:
            t = analyst.context.threads[tid]
            t.status = thread_data.get("status", t.status)
            t.questions.extend(thread_data.get("questions", []))
        else:
            analyst.context.threads[tid] = TopicThread(
                thread_id=tid,
                topic=thread_data.get("topic", tid),
                started_at="", last_active_at="",
                status=thread_data.get("status", "open"),
                questions=thread_data.get("questions", []),
            )

    def test_same_topic_new_id_merges(self):
        a = _analyst_with_threads()
        topic = "Integration of GitHub with Azure DevOps"
        self._apply(a, {"thread_id": "gh_ado_1", "topic": topic})
        self._apply(a, {"thread_id": "gh_ado_2", "topic": topic})
        self._apply(a, {"thread_id": "github_devops", "topic": topic})
        assert len(a.context.threads) == 1

    def test_distinct_topics_create_separate_threads(self):
        a = _analyst_with_threads()
        self._apply(a, {"thread_id": "a1", "topic": "GitHub vs Azure DevOps"})
        self._apply(a, {"thread_id": "b1", "topic": "Capacity sizing F64"})
        assert len(a.context.threads) == 2
