"""Research pipeline — multi-source search and synthesis."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import httpx

from sidekick.llm import call_llm, stream_llm

logger = logging.getLogger(__name__)

# Boundary between the lead answer (read aloud on the call) and the trailing
# "Sources:" block. Used to surface the lead early on the streaming path.
_SOURCES_RE = re.compile(r"\n\s*Sources?\b", re.IGNORECASE)


def _lead_answer(text: str) -> str:
    """Return the lead answer — text before any ``Sources:`` block, stripped."""
    return _SOURCES_RE.split(text, maxsplit=1)[0].strip()

SYNTHESIS_SYSTEM_PROMPT = """You are a {domain_scope} technical research assistant \
embedded in a live customer engagement. Your answers are read aloud by the \
consultant on the call, so they must be specific, actionable, and anchored to \
the customer's actual situation.

RULES:
- Lead with the direct answer in 1-2 sentences
- ALWAYS include a "Sources:" section at the end with numbered URLs
- State confidence: HIGH (verified docs), MEDIUM (partial match), LOW (inference)
- Flag GA vs Preview vs Planned features
- State uncertainty explicitly when docs are ambiguous
- Keep it brief — the consultant is on a live call
- Reference the customer by name when relevant
- Tie recommendations back to the customer's active threads and domains

VERIFIED SOURCES (priority order):
1. Microsoft Learn — learn.microsoft.com
2. Microsoft Fabric Blog — blog.fabric.microsoft.com
3. Microsoft Fabric Roadmap — roadmap.fabric.microsoft.com
4. Databricks / Delta Lake docs — docs.databricks.com, docs.delta.io
5. Apache Spark docs — spark.apache.org/docs
6. AWS docs — docs.aws.amazon.com (for cross-cloud topics)
7. Workspace files — engagement artifacts, instruction files

OUTPUT FORMAT:
<direct answer>

Sources:
1. <title> — <URL>
2. <title> — <URL>

Only cite sources you are confident are reputable. Never fabricate URLs. \
If web results provide URLs, use those. If no URLs are available, state \
\"Sources: Based on training knowledge (no live URLs retrieved).\""""


@dataclass
class ResearchResult:
    """Result from the research pipeline."""

    question: str
    answer: str = ""
    sources: list[str] = field(default_factory=list)
    confidence: str = "medium"

    def format(self) -> str:
        source_text = "\n".join(f"  \u2022 {s}" for s in self.sources) if self.sources else "  (none)"
        return f"""{self.answer}

Sources [{self.confidence.upper()}]:
{source_text}"""


@dataclass
class _WebHit:
    """A single ranked web search result from any provider."""

    title: str
    url: str
    snippet: str = ""
    source_label: str = "Web"
    score: float = 0.0
    relevance: float = 0.0


# ---------------------------------------------------------------------------
# Verified-source trust map and per-domain routing
# ---------------------------------------------------------------------------
# Single source of truth for "verified URLs". A result is only surfaced if its
# host matches a family below; everything else is dropped. Weights set the
# default ranking — Microsoft properties rank highest (the engagement
# verification rule), then partner/OSS docs. Kept small and in-code (not a long
# config list); customer profiles may *extend* it via
# ``grounding.extra_trusted_domains`` but never need to restate these.
_SOURCE_TRUST: dict[str, float] = {
    "learn.microsoft.com": 100.0,
    "blog.fabric.microsoft.com": 95.0,
    "roadmap.fabric.microsoft.com": 95.0,
    "techcommunity.microsoft.com": 80.0,
    "azure.microsoft.com": 75.0,
    "microsoft.com": 70.0,            # other Microsoft properties
    "docs.databricks.com": 65.0,
    "databricks.com": 55.0,
    "docs.delta.io": 65.0,
    "spark.apache.org": 60.0,
    "docs.aws.amazon.com": 60.0,
    "aws.amazon.com": 50.0,
    "postgresql.org": 60.0,
    # Reputable non-Microsoft technical sources (Phase 8 / 8.2). Microsoft still
    # outranks these by trust; they surface when on-topic and MS has no match.
    "registry.terraform.io": 70.0,
    "developer.hashicorp.com": 70.0,
    "hashicorp.com": 55.0,
    "kubernetes.io": 55.0,
    "python.org": 55.0,
}

# Boost added to a result when its host is a preferred source for one of the
# question's detected domains. This is what makes routing work: an AWS question
# lifts AWS docs above their baseline so they can rank alongside Microsoft docs,
# without ever suppressing Microsoft sources when they are relevant.
_ROUTING_BOOST = 45.0

# Weight applied to topical relevance in ranking (Phase 8 / 8.1) so an off-topic
# high-trust page (e.g. a stray learn.microsoft.com result) is demoted below an
# on-topic one, and a relevant reputable non-Microsoft source can outrank an
# off-topic Microsoft page.
_RELEVANCE_WEIGHT = 80.0
# A source must clear this relevance floor to be cited, so a weakly-matched URL
# is dropped rather than surfaced as a wrong citation (8.3).
_SOURCE_RELEVANCE_FLOOR = 0.2
_MAX_SOURCES = 5

_STOP_WORDS = frozenset({
    "the", "a", "an", "of", "to", "for", "in", "on", "at", "by", "is", "are",
    "be", "do", "does", "can", "could", "would", "should", "how", "what",
    "which", "who", "when", "we", "you", "it", "this", "that", "with", "your",
    "our", "and", "or", "as", "if", "so", "about", "into", "from", "use",
    "using", "need", "want",
})


def _content_words(text: str) -> set[str]:
    return {
        w
        for w in re.findall(r"[a-z0-9]{3,}", (text or "").lower())
        if w not in _STOP_WORDS
    }


def _relevance(query: str, title: str, snippet: str) -> float:
    """Fraction of the query's content words present in a hit's title+snippet."""
    q = _content_words(query)
    if not q:
        return 0.0
    text = f"{title} {snippet}".lower()
    return sum(1 for w in q if w in text) / len(q)


_CONFIDENCE_RE = re.compile(r"confidence[:*\s]+\**\s*(HIGH|MEDIUM|LOW)", re.IGNORECASE)


def _parse_confidence(answer: str, has_sources: bool) -> str:
    """Read the answer's stated confidence (HIGH/MEDIUM/LOW), else infer it.

    The synthesis prompt asks the model to state confidence; honouring it means
    the feed shows real confidence instead of a hardcoded 'medium' (8.4).
    """
    m = _CONFIDENCE_RE.search(answer or "")
    if m:
        return m.group(1).lower()
    if not has_sources:
        return "low"
    return "medium"

# Detected-domain keyword (substring, lowercased) -> preferred host families.
# Matched against the domain labels already detected/configured upstream
# (e.g. "AWS S3 Integration", "Microsoft Fabric", "PostgreSQL").
_DOMAIN_ROUTING: dict[str, list[str]] = {
    "aws": ["docs.aws.amazon.com", "aws.amazon.com"],
    "s3": ["docs.aws.amazon.com", "aws.amazon.com"],
    "databricks": ["docs.databricks.com", "databricks.com", "docs.delta.io", "spark.apache.org"],
    "delta": ["docs.delta.io", "docs.databricks.com"],
    "spark": ["spark.apache.org", "docs.databricks.com"],
    "postgres": ["postgresql.org", "learn.microsoft.com"],
    "fabric": ["learn.microsoft.com", "blog.fabric.microsoft.com", "roadmap.fabric.microsoft.com"],
    "power bi": ["learn.microsoft.com", "blog.fabric.microsoft.com"],
    "azure": ["learn.microsoft.com", "azure.microsoft.com"],
    "terraform": [
        "registry.terraform.io", "developer.hashicorp.com", "hashicorp.com",
        "learn.microsoft.com",
    ],
}


class ResearchPipeline:
    """Multi-source research pipeline.

    Searches:
    1. Live web — Microsoft Learn API (always) + a general web-search provider
       (Tavily or Brave, if a key is set), ranked by a verified-source trust map
       with per-domain routing (e.g. an AWS question lifts AWS docs).
    2. Workspace files (engagement artifacts from grounding.repo_paths)
    3. Instruction files (.github/instructions/)
    4. LLM synthesis with context from above
    """

    def __init__(self, config=None):
        self._repo_paths = (
            config.grounding.repo_paths if config else [".github/instructions/"]
        )
        # Resolve paths relative to workspace root (not CWD)
        self._workspace_root = Path(
            os.environ.get("SIDEKICK_WORKSPACE_ROOT", ".")
        )
        # Verified-source trust map: code defaults + optional per-customer
        # extensions from grounding.extra_trusted_domains ({host: weight}).
        self._trust: dict[str, float] = dict(_SOURCE_TRUST)
        extra = getattr(getattr(config, "grounding", None), "extra_trusted_domains", None)
        if isinstance(extra, dict):
            for host, weight in extra.items():
                try:
                    self._trust[str(host).lower().lstrip(".")] = float(weight)
                except (TypeError, ValueError):
                    continue

    async def execute_direct(
        self,
        question: str,
        depth: str = "medium",
        context=None,
        tier: str = "deep",
        domains: list[str] | None = None,
        on_lead: Callable[[str], None] | None = None,
        self_critique: bool = False,
    ) -> ResearchResult:
        """Execute a research query directly (not from queue).

        Args:
            tier: LLM tier for synthesis — 'deep' (claude-opus-4.7) by default.
            domains: Customer domains for scoping the search query.
            on_lead: Optional callback fired once with the lead answer as soon
                as it streams in (before the Sources block). When provided,
                synthesis streams; when ``None`` (default) it uses the standard
                non-streaming path — byte-identical to prior behaviour.
        """
        # Rewrite the raw transcript question into a focused search query
        search_query = await self._rewrite_search_query(question, domains)

        # Gather context from the workspace
        repo_context = self._search_repo(search_query)
        instruction_context = self._search_instructions(search_query)

        # Gather live web context from MS Learn and verified sources,
        # ranked with per-domain source routing.
        web_context, ranked_hits = await self._search_web(search_query, domains)

        # Build customer engagement context
        customer_block = ""
        if context:
            customer_name = getattr(context, "customer_name", "") or ""
            threads = getattr(context, "threads", {})
            key_facts = getattr(context, "key_facts", [])
            thread_details = []
            for t in threads.values():
                detail = f"  - [{t.status}] {t.topic}"
                for kf in t.key_facts[:2]:
                    detail += f"\n      fact: {kf}"
                for q in t.questions[:2]:
                    detail += f"\n      question: {q}"
                thread_details.append(detail)
            parts = []
            if customer_name:
                parts.append(f"Customer: {customer_name}")
            if thread_details:
                parts.append("Active threads:\n" + "\n".join(thread_details))
            if key_facts:
                parts.append("Key facts:\n" + "\n".join(f"  - {f}" for f in key_facts[-8:]))
            if parts:
                customer_block = "\n\nCUSTOMER ENGAGEMENT:\n" + "\n".join(parts)

        user_prompt = f"""QUESTION: {question}

DEPTH: {depth}{customer_block}

WEB RESULTS (from verified sources):
{web_context}

WORKSPACE CONTEXT:
{repo_context}

TEAM STANDARDS:
{instruction_context}

MEETING CONTEXT:
{self._format_meeting_context(context)}

Research this question and provide a concise, sourced answer. \
Anchor your response to the customer's specific context where possible. \
Cite the URLs from web results where they support your answer."""

        # Scope the system prompt to actual domains being discussed
        domain_scope = ", ".join(domains) if domains else "Microsoft Fabric"
        system_prompt = SYNTHESIS_SYSTEM_PROMPT.format(domain_scope=domain_scope)

        if self_critique:
            # Phase 4 / A3: draft (non-streaming) → self-critique → refine, then
            # surface the refined lead once so the feed shows the corrected
            # answer rather than an un-reviewed streamed draft.
            draft = await call_llm(
                system_prompt=system_prompt, user_prompt=user_prompt, tier=tier
            )
            answer = await self._critique_and_refine(
                system_prompt, user_prompt, draft, tier
            )
            if on_lead is not None and answer.strip():
                try:
                    on_lead(_lead_answer(answer))
                except Exception:  # noqa: BLE001 — callback must never break synthesis
                    logger.debug("on_lead callback raised", exc_info=True)
        elif on_lead is not None:
            answer = await self._synthesise_streaming(
                system_prompt, user_prompt, tier, on_lead
            )
        else:
            answer = await call_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                tier=tier,
            )

        # Only cite sources that clear the relevance floor, so a weakly-matched
        # (often off-topic) URL is dropped rather than surfaced (Phase 8 / 8.3).
        sources = [
            h.url for h in ranked_hits if h.relevance >= _SOURCE_RELEVANCE_FLOOR
        ][:_MAX_SOURCES]

        return ResearchResult(
            question=question,
            answer=answer,
            sources=sources,
            confidence=_parse_confidence(answer, bool(sources)),
        )

    async def _synthesise_streaming(
        self,
        system_prompt: str,
        user_prompt: str,
        tier: str,
        on_lead: Callable[[str], None],
    ) -> str:
        """Stream the synthesis, firing ``on_lead`` once the lead answer is ready.

        Falls back to a single non-streaming :func:`call_llm` if streaming
        fails, so the result is never lost. ``on_lead`` is fired at most once.
        """
        chunks: list[str] = []
        lead_fired = False

        def _maybe_fire_lead(force: bool = False) -> None:
            nonlocal lead_fired
            if lead_fired:
                return
            acc = "".join(chunks)
            lead = _lead_answer(acc)
            # Fire when the Sources block starts, or once we have a complete
            # lead sentence of reasonable length, or on a forced final flush.
            ready = (
                _SOURCES_RE.search(acc) is not None
                or (len(lead) >= 40 and lead.rstrip().endswith((".", "!", "?")))
            )
            if (ready or force) and lead:
                try:
                    on_lead(lead)
                except Exception:  # noqa: BLE001 — callback must never break synthesis
                    logger.debug("on_lead callback raised", exc_info=True)
                lead_fired = True

        try:
            async for delta in stream_llm(
                system_prompt=system_prompt, user_prompt=user_prompt, tier=tier
            ):
                chunks.append(delta)
                _maybe_fire_lead()
            _maybe_fire_lead(force=True)
            return "".join(chunks)
        except Exception as e:  # noqa: BLE001 — degrade to non-streaming
            logger.warning("Streaming synthesis failed (%s); using call_llm", e)
            answer = await call_llm(
                system_prompt=system_prompt, user_prompt=user_prompt, tier=tier
            )
            if not lead_fired and answer.strip():
                try:
                    on_lead(_lead_answer(answer))
                except Exception:  # noqa: BLE001
                    logger.debug("on_lead callback raised", exc_info=True)
            return answer

    async def _critique_and_refine(
        self, system_prompt: str, user_prompt: str, draft: str, tier: str
    ) -> str:
        """Second-pass self-critique (Phase 4 / A3).

        Reviews the draft against the sources and answer rules, then returns a
        corrected final answer. Degrades to the draft on any failure so a
        critique error never loses the answer.
        """
        try:
            critique_prompt = (
                "Review the DRAFT answer below against the WEB RESULTS and TEAM "
                "STANDARDS in the original prompt, and the answer rules. Check: "
                "is every claim supported by a cited source? Any GA-vs-Preview "
                "mistakes, hallucinated features, or unsupported specifics? "
                "Correct them, keep it brief, keep the Sources section, and "
                "output ONLY the improved final answer.\n\n"
                f"ORIGINAL PROMPT:\n{user_prompt}\n\nDRAFT ANSWER:\n{draft}"
            )
            refined = await call_llm(
                system_prompt=system_prompt,
                user_prompt=critique_prompt,
                tier=tier,
            )
            return refined.strip() or draft
        except Exception as e:  # noqa: BLE001 — never lose the answer
            logger.warning("Self-critique failed (%s); using draft", e)
            return draft

    async def _rewrite_search_query(
        self, question: str, domains: list[str] | None = None,
    ) -> str:
        """Rewrite a raw transcript question into a focused search query.

        Transcript questions are often incomplete, verbose, or contain
        filler. This distills them into 3-6 keyword phrases that work
        well with MS Learn search and file keyword matching.
        """
        domain_hint = ", ".join(domains) if domains else "Microsoft Fabric"
        try:
            rewritten = await call_llm(
                system_prompt=(
                    "You convert meeting transcript questions into concise "
                    "Microsoft Learn search queries. Output ONLY the search "
                    "query — no explanation, no quotes. Use 3-8 words "
                    "separated by spaces. Focus on the core technical "
                    "concept being asked about."
                ),
                user_prompt=(
                    f"Domain: {domain_hint}\n"
                    f"Transcript question: {question}\n"
                    f"Search query:"
                ),
                tier="fast",
                timeout=8,
            )
            rewritten = rewritten.strip().strip('"').strip("'")
            # Accept if it's a reasonable length (not empty, not a full paragraph)
            if 5 <= len(rewritten) <= 120:
                logger.debug("Search query rewritten: %r → %r", question, rewritten)
                return rewritten
        except Exception as e:
            logger.debug("Query rewrite failed, using original: %s", e)
        return question

    def _search_repo(self, question: str) -> str:
        """Search configured repo paths for relevant engagement artifacts."""
        keywords = [w for w in question.lower().split() if len(w) > 3]
        if not keywords:
            return "(no search terms extracted)"

        results: list[tuple[int, str]] = []

        for repo_path_str in self._repo_paths:
            repo_path = self._workspace_root / repo_path_str
            if not repo_path.exists():
                continue

            # Skip instruction files (covered by _search_instructions)
            if repo_path_str.rstrip("/").endswith("instructions"):
                continue

            # Search markdown, text, and SQL files
            for suffix in ("*.md", "*.txt", "*.sql"):
                for f in repo_path.rglob(suffix):
                    try:
                        content = f.read_text(encoding="utf-8")
                        name_lower = f.name.lower()
                        preview = content[:1000].lower()

                        score = sum(
                            1 for kw in keywords
                            if kw in name_lower or kw in preview
                        )
                        if score > 0:
                            snippet = content[:300].strip()
                            rel = f.relative_to(self._workspace_root)
                            results.append((score, f"--- {rel} ---\n{snippet}"))
                    except Exception:
                        continue

            # Index notebooks by name only (avoid parsing large JSON)
            for f in repo_path.rglob("*.ipynb"):
                name_lower = f.name.lower()
                score = sum(1 for kw in keywords if kw in name_lower)
                if score > 0:
                    rel = f.relative_to(self._workspace_root)
                    results.append((score, f"--- {rel} (notebook) ---"))

        if not results:
            return "(no matching repo files)"

        # Sort by relevance score descending, take top 5
        results.sort(key=lambda x: x[0], reverse=True)
        return "\n\n".join(text for _, text in results[:5])

    def _search_instructions(self, question: str) -> str:
        """Search .github/instructions/ for relevant team standards."""
        instructions_dir = self._workspace_root / ".github" / "instructions"
        if not instructions_dir.exists():
            return "(no instruction files found)"

        keywords = [w.lower() for w in question.split() if len(w) > 3]
        if not keywords:
            return "(no search terms)"

        scored: list[tuple[int, str]] = []
        for f in instructions_dir.glob("*.instructions.md"):
            try:
                content = f.read_text(encoding="utf-8")
                name_lower = f.stem.lower()
                preview = content[:1500].lower()

                # Score by keyword hits in both filename and content
                score = sum(
                    2 if kw in name_lower else (1 if kw in preview else 0)
                    for kw in keywords
                )
                if score > 0:
                    scored.append((score, f"--- {f.name} ---\n{content[:500]}"))
            except Exception:
                continue

        if not scored:
            return "(no matching instructions)"

        scored.sort(key=lambda x: x[0], reverse=True)
        return "\n\n".join(text for _, text in scored[:3])

    def _format_meeting_context(self, context) -> str:
        if not context:
            return "(no meeting context)"
        facts = getattr(context, "key_facts", [])
        return "\n".join(f"- {f}" for f in facts) if facts else "(no key facts yet)"

    async def _search_web(
        self, question: str, domains: list[str] | None = None,
    ) -> tuple[str, list[_WebHit]]:
        """Search verified sources and rank them with per-domain routing.

        Two live layers feed a single ranker:
          1. Microsoft Learn search API (free, no key) — authoritative MS docs.
          2. A general web-search provider (Tavily or Brave, if a key is set) —
             broadens coverage to AWS / Databricks / Spark / PostgreSQL etc.
             Replaces the retired Bing Web Search API (decommissioned 2025-08-11).

        Every hit is filtered to the verified-source trust map and ranked so
        Microsoft sources stay high while the question's detected domain lifts
        its preferred sources (e.g. an AWS question promotes docs.aws.amazon.com
        above its baseline). Only verified URLs are surfaced for citation.
        """
        hits: list[_WebHit] = []

        # 1. Microsoft Learn (always on, no key)
        try:
            hits.extend(await self._search_ms_learn(question))
        except Exception as e:
            logger.warning("MS Learn search failed: %s", e)

        # 2. General web-search provider (optional, selected by which key is set)
        try:
            hits.extend(await self._search_provider(question))
        except Exception as e:
            logger.warning("Web provider search failed: %s", e)

        ranked = self._rank_hits(hits, domains, query=question)
        if not ranked:
            return (
                "(no verified web results — LLM will use training knowledge)",
                [],
            )

        return "\n\n".join(self._format_hit(h) for h in ranked[:8]), ranked

    # ----- ranking & source verification -----

    def _trust_for_host(self, host: str) -> float:
        """Trust weight for a host, or 0.0 if it is not a verified source."""
        host = host.lower()
        best = 0.0
        for trusted, weight in self._trust.items():
            if host == trusted or host.endswith("." + trusted):
                best = max(best, weight)
        return best

    @staticmethod
    def _routing_hosts(domains: list[str] | None) -> set[str]:
        """Preferred source hosts for the question's detected domains."""
        if not domains:
            return set()
        preferred: set[str] = set()
        for d in domains:
            dl = d.lower()
            for key, hosts in _DOMAIN_ROUTING.items():
                if key in dl:
                    preferred.update(hosts)
        return preferred

    def _rank_hits(
        self, hits: list[_WebHit], domains: list[str] | None, query: str = "",
    ) -> list[_WebHit]:
        """Filter to verified sources, score with domain routing + topical
        relevance, dedupe, sort."""
        routing = self._routing_hosts(domains)
        scored: list[_WebHit] = []
        seen: set[str] = set()
        for h in hits:
            url = (h.url or "").strip()
            if not url or url in seen:
                continue
            host = urlparse(url).netloc.lower()
            base = self._trust_for_host(host)
            if base <= 0:
                continue  # not a verified source — drop (verified-URL rule)
            in_route = any(
                host == r or host.endswith("." + r) for r in routing
            )
            rel = _relevance(query, h.title, h.snippet) if query else 0.0
            h.relevance = rel
            h.score = (
                base + _RELEVANCE_WEIGHT * rel + (_ROUTING_BOOST if in_route else 0.0)
            )
            seen.add(url)
            scored.append(h)
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored

    @staticmethod
    def _format_hit(h: _WebHit) -> str:
        return f"[{h.source_label}] {h.title}\n  URL: {h.url}\n  {h.snippet}"

    @staticmethod
    def _extract_urls(web_context: str) -> list[str]:
        """Extract URLs from web context string for the sources list."""
        urls = []
        for line in web_context.split("\n"):
            line = line.strip()
            if line.startswith("URL:"):
                url = line[4:].strip()
                if url:
                    urls.append(url)
        return urls

    async def _search_ms_learn(self, question: str) -> list[_WebHit]:
        """Search Microsoft Learn documentation via the free search API."""
        url = "https://learn.microsoft.com/api/search"
        params = {
            "search": question,
            "locale": "en-us",
            "$top": "8",
        }
        hits: list[_WebHit] = []
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

            for item in data.get("results", [])[:8]:
                title = item.get("title", "")
                snippet = str(item.get("description", ""))[:200]
                link = item.get("url", "")
                if title and link and self._is_useful_url(link):
                    hits.append(_WebHit(
                        title=title, url=link, snippet=snippet, source_label="MS Learn",
                    ))

        return hits

    @staticmethod
    def _is_useful_url(url: str) -> bool:
        """Filter out broad landing pages and certification guides."""
        # Reject root-level landing pages with very short paths
        parsed = urlparse(url)
        path = parsed.path.rstrip("/")
        path_segments = [s for s in path.split("/") if s]
        # Reject if path has fewer than 3 segments (e.g. /en-us/fabric/)
        if len(path_segments) < 3:
            return False
        # Reject certification/training pages
        reject_patterns = [
            "/training/", "/certifications/", "/credentials/",
            "/learn/paths/", "/study-guide",
        ]
        path_lower = path.lower()
        return not any(p in path_lower for p in reject_patterns)

    async def _search_provider(self, question: str) -> list[_WebHit]:
        """Query a web-search provider — a keyed one if configured, else keyless.

        Precedence (Phase 9 / 9.1): a Tavily or Brave key (per-user or a shared
        org key) wins for reliability; otherwise DuckDuckGo runs **keyless** so
        non-Microsoft results work out of the box with no user configuration.
        Every hit still passes through the verified-source ranker, so only
        trusted hosts are ever surfaced.
        """
        tavily_key = os.environ.get("TAVILY_API_KEY", "")
        if tavily_key:
            return await self._search_tavily(question, tavily_key)
        brave_key = os.environ.get("BRAVE_API_KEY", "")
        if brave_key:
            return await self._search_brave(question, brave_key)
        # Keyless default — real non-Microsoft results with zero user setup.
        return await self._search_duckduckgo(question)

    async def _search_duckduckgo(self, question: str) -> list[_WebHit]:
        """Keyless web search via DuckDuckGo (no API key). Best-effort.

        Runs the blocking ``ddgs`` client off the event loop and degrades to an
        empty list on any failure (rate-limit, missing package, network), so MS
        Learn + model knowledge still answer.
        """

        def _blocking() -> list[_WebHit]:
            try:
                try:
                    from ddgs import DDGS
                except ImportError:
                    from duckduckgo_search import DDGS  # older package name
            except ImportError:
                logger.debug("ddgs not installed; skipping keyless web search")
                return []
            out: list[_WebHit] = []
            try:
                with DDGS() as ddgs:
                    for r in ddgs.text(question, max_results=8):
                        title = r.get("title", "")
                        url = r.get("href") or r.get("url") or ""
                        body = str(r.get("body", ""))[:200]
                        if title and url:
                            out.append(_WebHit(
                                title=title, url=url, snippet=body, source_label="Web",
                            ))
            except Exception as e:  # noqa: BLE001 — best-effort keyless search
                logger.debug("DuckDuckGo search failed: %s", e)
                return []
            return out

        return await asyncio.to_thread(_blocking)

    async def _search_tavily(self, question: str, api_key: str) -> list[_WebHit]:
        """Tavily Search API — https://api.tavily.com/search.

        Scoped to the verified-source allowlist via ``include_domains`` so the
        API mostly returns citable results; the ranker still enforces the
        allowlist as the authority.
        """
        payload = {
            "api_key": api_key,
            "query": question,
            "search_depth": "basic",
            "max_results": 8,
            "include_domains": list(self._trust.keys()),
        }
        hits: list[_WebHit] = []
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post("https://api.tavily.com/search", json=payload)
            resp.raise_for_status()
            data = resp.json()
            for item in data.get("results", [])[:8]:
                title = item.get("title", "")
                link = item.get("url", "")
                snippet = str(item.get("content", ""))[:200]
                if title and link:
                    hits.append(_WebHit(
                        title=title, url=link, snippet=snippet, source_label="Web",
                    ))
        return hits

    async def _search_brave(self, question: str, api_key: str) -> list[_WebHit]:
        """Brave Search API — https://api.search.brave.com/res/v1/web/search.

        Brave has no per-request domain allowlist, so results are filtered by the
        verified-source ranker after retrieval.
        """
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {"Accept": "application/json", "X-Subscription-Token": api_key}
        params = {"q": question, "count": "10", "country": "GB"}
        hits: list[_WebHit] = []
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()
            for item in data.get("web", {}).get("results", [])[:10]:
                title = item.get("title", "")
                link = item.get("url", "")
                snippet = str(item.get("description", ""))[:200]
                if title and link:
                    hits.append(_WebHit(
                        title=title, url=link, snippet=snippet, source_label="Web",
                    ))
        return hits

