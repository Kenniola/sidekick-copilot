"""Tests for research per-domain source routing and verified-URL filtering."""

import pytest

from sidekick.actions import research as research_mod
from sidekick.actions.research import (
    ResearchPipeline,
    _parse_confidence,
    _relevance,
    _WebHit,
)


def _h(url, title="t", snippet="s"):
    return _WebHit(title=title, url=url, snippet=snippet, source_label="Web")


def _hit(url: str, title: str = "t") -> _WebHit:
    return _WebHit(title=title, url=url, snippet="s", source_label="Web")


def test_drops_unverified_hosts():
    """Only hosts in the trust map survive ranking (verified-URL rule)."""
    pipeline = ResearchPipeline()
    hits = [
        _hit("https://learn.microsoft.com/en-us/fabric/onelake/overview"),
        _hit("https://random-blog.example.com/post/fabric-tips"),
        _hit("https://stackoverflow.com/questions/123/fabric"),
    ]
    ranked = pipeline._rank_hits(hits, domains=None)
    assert any("learn.microsoft.com" in h.url for h in ranked)
    assert not any("example.com" in h.url for h in ranked)
    assert not any("stackoverflow.com" in h.url for h in ranked)
    assert len(ranked) == 1


def test_microsoft_ranks_first_by_default():
    """With no domain routing, Microsoft outranks partner/OSS docs."""
    pipeline = ResearchPipeline()
    hits = [
        _hit("https://docs.aws.amazon.com/AmazonS3/latest/userguide/intro.html"),
        _hit("https://learn.microsoft.com/en-us/fabric/onelake/onelake-shortcuts"),
    ]
    ranked = pipeline._rank_hits(hits, domains=None)
    assert ranked[0].url.startswith("https://learn.microsoft.com")


def test_aws_domain_promotes_aws_docs():
    """An AWS-detected question lifts AWS docs above the Microsoft baseline."""
    pipeline = ResearchPipeline()
    hits = [
        _hit("https://learn.microsoft.com/en-us/fabric/onelake/onelake-shortcuts"),
        _hit("https://docs.aws.amazon.com/AmazonS3/latest/userguide/access-points.html"),
    ]
    ranked = pipeline._rank_hits(hits, domains=["AWS S3 Integration"])
    assert ranked[0].url.startswith("https://docs.aws.amazon.com")
    # Microsoft is not suppressed — it is still present, just second.
    assert any("learn.microsoft.com" in h.url for h in ranked)


def test_dedup_by_url():
    """Duplicate URLs are collapsed to a single ranked hit."""
    pipeline = ResearchPipeline()
    url = "https://learn.microsoft.com/en-us/fabric/onelake/onelake-overview"
    ranked = pipeline._rank_hits([_hit(url), _hit(url)], domains=None)
    assert len(ranked) == 1


def test_subdomain_match_is_anchored():
    """Trust matching is host-anchored — look-alike domains are rejected."""
    pipeline = ResearchPipeline()
    hits = [_hit("https://learn.microsoft.com.evil.example/phish")]
    ranked = pipeline._rank_hits(hits, domains=None)
    assert ranked == []


def test_extra_trusted_domains_extends_map():
    """grounding.extra_trusted_domains adds a host without editing code."""

    class _Grounding:
        repo_paths = [".github/instructions/"]
        extra_trusted_domains = {"docs.snowflake.com": 60.0}

    class _Config:
        grounding = _Grounding()

    pipeline = ResearchPipeline(config=_Config())
    hits = [_hit("https://docs.snowflake.com/en/user-guide/intro")]
    ranked = pipeline._rank_hits(hits, domains=None)
    assert len(ranked) == 1


class TestRelevanceRanking:
    """Phase 8 / 8.1: topical relevance demotes off-topic high-trust pages."""

    def test_relevance_scoring(self):
        assert _relevance(
            "terraform fabric provider",
            "Terraform Fabric provider",
            "manage fabric with terraform",
        ) > 0.5
        assert _relevance(
            "terraform fabric provider", "Cosmos DB overview", "azure cosmos database"
        ) == 0.0

    def test_on_topic_beats_off_topic_same_trust(self):
        p = ResearchPipeline()
        hits = [
            _h(
                "https://learn.microsoft.com/azure/cosmos-db/overview",
                title="Cosmos DB overview",
                snippet="azure cosmos database",
            ),
            _h(
                "https://learn.microsoft.com/fabric/cicd/git-integration",
                title="Fabric Git integration for CI/CD",
                snippet="connect fabric workspaces to git for ci cd",
            ),
        ]
        ranked = p._rank_hits(hits, domains=None, query="fabric git integration ci cd")
        assert ranked[0].url.endswith("git-integration")

    def test_relevant_nonms_beats_off_topic_ms(self):
        p = ResearchPipeline()
        hits = [
            _h(
                "https://learn.microsoft.com/azure/cosmos-db/overview",
                title="Cosmos DB",
                snippet="nosql database",
            ),
            _h(
                "https://registry.terraform.io/providers/microsoft/fabric",
                title="Fabric Terraform provider resources",
                snippet="terraform provider to manage fabric resources",
            ),
        ]
        ranked = p._rank_hits(
            hits, domains=["Terraform"], query="fabric terraform provider resources"
        )
        assert ranked[0].url.startswith("https://registry.terraform.io")

    def test_relevance_recorded_on_hit(self):
        p = ResearchPipeline()
        hit = _h(
            "https://learn.microsoft.com/fabric/onelake",
            title="OneLake overview",
            snippet="onelake storage",
        )
        ranked = p._rank_hits([hit], domains=None, query="onelake storage")
        assert ranked[0].relevance > 0.0


class TestParseConfidence:
    """Phase 8 / 8.4: read the model's stated confidence instead of hardcoding."""

    def test_reads_stated_high(self):
        assert _parse_confidence("...\nConfidence: HIGH\n...", True) == "high"

    def test_reads_bold_low(self):
        assert _parse_confidence("blah **Confidence: LOW**", True) == "low"

    def test_fallback_no_sources_is_low(self):
        assert _parse_confidence("no statement here", False) == "low"

    def test_fallback_with_sources_is_medium(self):
        assert _parse_confidence("no statement here", True) == "medium"


class TestProviderPrecedence:
    """Phase 9 / 9.1: keyed provider wins; else keyless DuckDuckGo default."""

    @pytest.mark.asyncio
    async def test_tavily_used_when_key_set(self, monkeypatch):
        p = ResearchPipeline()
        monkeypatch.setenv("TAVILY_API_KEY", "k")
        called = {}

        async def _fake_tav(q, key):
            called["tav"] = key
            return []

        async def _fake_ddg(q):
            called["ddg"] = True
            return []

        monkeypatch.setattr(p, "_search_tavily", _fake_tav)
        monkeypatch.setattr(p, "_search_duckduckgo", _fake_ddg)
        await p._search_provider("q")
        assert called.get("tav") == "k"
        assert "ddg" not in called

    @pytest.mark.asyncio
    async def test_duckduckgo_used_when_no_key(self, monkeypatch):
        p = ResearchPipeline()
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)
        called = {}

        async def _fake_ddg(q):
            called["ddg"] = True
            return []

        monkeypatch.setattr(p, "_search_duckduckgo", _fake_ddg)
        out = await p._search_provider("q")
        assert called.get("ddg") is True
        assert out == []

    @pytest.mark.asyncio
    async def test_brave_used_before_duckduckgo(self, monkeypatch):
        p = ResearchPipeline()
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        monkeypatch.setenv("BRAVE_API_KEY", "b")
        called = {}

        async def _fake_brave(q, key):
            called["brave"] = key
            return []

        async def _fake_ddg(q):
            called["ddg"] = True
            return []

        monkeypatch.setattr(p, "_search_brave", _fake_brave)
        monkeypatch.setattr(p, "_search_duckduckgo", _fake_ddg)
        await p._search_provider("q")
        assert called.get("brave") == "b"
        assert "ddg" not in called

    @pytest.mark.asyncio
    async def test_refines_the_draft(self, monkeypatch):
        async def _fake_llm(**kwargs):
            return "  REFINED answer\n\nSources:\n1. url  "

        monkeypatch.setattr(research_mod, "call_llm", _fake_llm)
        pipeline = ResearchPipeline()
        out = await pipeline._critique_and_refine(
            "sys", "prompt", "draft answer", "deep"
        )
        assert out == "REFINED answer\n\nSources:\n1. url"

    @pytest.mark.asyncio
    async def test_degrades_to_draft_on_failure(self, monkeypatch):
        async def _boom(**kwargs):
            raise RuntimeError("model down")

        monkeypatch.setattr(research_mod, "call_llm", _boom)
        pipeline = ResearchPipeline()
        out = await pipeline._critique_and_refine(
            "sys", "prompt", "the draft", "deep"
        )
        assert out == "the draft"

    @pytest.mark.asyncio
    async def test_empty_refine_falls_back_to_draft(self, monkeypatch):
        async def _empty(**kwargs):
            return "   "

        monkeypatch.setattr(research_mod, "call_llm", _empty)
        pipeline = ResearchPipeline()
        out = await pipeline._critique_and_refine(
            "sys", "prompt", "the draft", "deep"
        )
        assert out == "the draft"
