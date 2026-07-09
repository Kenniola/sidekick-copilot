"""Derived, self-maintaining domain vocabulary for the Whisper prior (Phase 5b).

Whisper's recognition of proper nouns and domain jargon (the terms a consultant
reads back to a client) is its weakest area — "Northwind" came out as "the node",
"Gen 1" as "gentlemen", "Priya Okafor" as "Pria O'Farrell". faster-whisper accepts an
``initial_prompt`` that biases recognition toward expected vocabulary, but a
hand-curated glossary is just hardcoding by another name: it rots, must be
authored per customer, and only ever covers terms someone remembered.

Instead, :class:`Vocabulary` **derives** the prior from material Sidekick already
holds, and lets it **improve over the call**:

1. **Seed** from the existing grounding inputs — the customer name/description,
   configured domains, and the loaded ``.github/instructions`` text. Nothing is
   authored specifically for Whisper; the prior is whatever the engagement
   context already contains.
2. **Adapt** in-session from the streams the LLM has *already corrected from
   context* — classified thread ``key_facts`` and research results. The analyst
   writes "Northwind" correctly even when Whisper misheard it, so feeding those
   terms back makes every later chunk recognise the term. The longer the
   meeting runs, the better proper-noun recognition gets.

The class holds no I/O and no LLM calls — callers pass plain text in.
"""

from __future__ import annotations

import re

# Acronyms: 2+ uppercase letters, optionally trailing digits (MI, AWS, ADLS,
# SQL, RLS, S3, PG16). Case-sensitive on purpose.
_ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,}\b")

# Capitalised proper-noun-ish tokens: an initial capital followed by lower-case
# letters, length >= 3 (Northwind, Fabric, Contoso, Tailwind, Fabrikam).
# Allows an internal hyphen ("T-SQL" is caught by the acronym rule; "Power-BI"
# style is rare). Case-sensitive.
_PROPER_RE = re.compile(r"\b[A-Z][a-z]{2,}(?:-[A-Z][a-z]+)?\b")

# Common capitalised words that start sentences but are not domain terms. Kept
# lower-cased for comparison. Deliberately small — over-filtering loses real
# terms; the frequency ranking already suppresses one-off noise.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "this", "that", "these", "those", "there", "their", "then",
        "they", "them", "and", "but", "for", "with", "from", "into", "your",
        "you", "our", "are", "was", "were", "will", "would", "can", "could",
        "should", "what", "when", "where", "which", "while", "who", "whom",
        "have", "has", "had", "not", "all", "any", "one", "two", "its",
        "yeah", "okay", "like", "just", "kind", "sort", "some", "also",
        "because", "about", "after", "before", "between", "over", "under",
        "here", "how", "why", "yes", "now", "let", "got", "get", "see",
        "want", "need", "make", "made", "use", "used", "using", "more",
        "most", "much", "many", "such", "very", "well", "good", "great",
        "think", "know", "going", "thing", "things", "way", "ways", "team",
        "teams", "data", "work", "works", "point", "points", "area", "areas",
        "say", "said", "ask", "asked", "come", "came", "give", "given",
        "happy", "sorry", "sure", "right", "back", "down", "out", "off",
        "long", "next", "last", "first", "second", "third", "still", "yet",
        "every", "each", "other", "both", "only", "even", "though", "really",
        "actually", "basically", "essentially", "probably", "maybe", "kinda",
    }
)

# An acronym that is purely a stopword in caps (e.g. "THE") — rare, but guard.
_ACRONYM_BLOCKLIST: frozenset[str] = frozenset({"THE", "AND", "FOR", "BUT", "YOU"})

# Bound on how much text we scan per update — keeps the hot path cheap.
_MAX_SCAN_CHARS = 20_000


def extract_terms(text: str, *, limit: int = 60) -> list[str]:
    """Extract candidate domain terms (proper nouns + acronyms) from ``text``.

    Returns a de-duplicated list in first-seen order, capped at ``limit``.
    Acronyms are kept verbatim; capitalised proper nouns are kept verbatim if
    they are not common sentence-initial stopwords.
    """
    if not text:
        return []
    text = text[:_MAX_SCAN_CHARS]

    seen: dict[str, None] = {}

    for m in _ACRONYM_RE.finditer(text):
        tok = m.group(0)
        if tok in _ACRONYM_BLOCKLIST:
            continue
        seen.setdefault(tok, None)

    for m in _PROPER_RE.finditer(text):
        tok = m.group(0)
        if tok.lower() in _STOPWORDS:
            continue
        seen.setdefault(tok, None)
        if len(seen) >= limit:
            break

    return list(seen.keys())[:limit]


class Vocabulary:
    """A frequency-ranked, bounded set of domain terms for the Whisper prior.

    Terms accumulate a weight: seeding adds a base weight, and in-session
    updates add (and thereby promote) terms harvested from LLM-corrected text.
    :meth:`initial_prompt` renders the top terms as a short comma-separated hint
    suitable for ``faster_whisper.transcribe(initial_prompt=...)``.
    """

    def __init__(self, max_terms: int = 50, max_prompt_chars: int = 800):
        self.max_terms = max_terms
        self.max_prompt_chars = max_prompt_chars
        # term -> weight. Higher weight = stronger/more recent signal.
        self._weights: dict[str, float] = {}

    # -- ingestion ---------------------------------------------------------

    def seed(self, *texts: str, weight: float = 1.0) -> None:
        """Add terms from seed text (config summary / grounding) at low weight."""
        for text in texts:
            for term in extract_terms(text):
                self._add(term, weight)

    def seed_terms(self, terms, *, weight: float = 3.0) -> None:
        """Add explicit glossary terms verbatim (no extraction) at high weight.

        Unlike :meth:`seed`, which mines terms out of free text, this trusts the
        caller's list — used for the per-customer ``glossary:`` config so exact
        proper nouns and multi-word phrases land in the prior from the first
        chunk, outranking generic seed vocabulary.
        """
        if not terms:
            return
        for term in terms:
            cleaned = str(term).strip()
            if cleaned:
                self._add(cleaned, weight)

    def update(self, texts, *, weight: float = 2.0) -> None:
        """Promote terms from LLM-corrected in-session text (key_facts, research).

        ``texts`` may be a single string or an iterable of strings. In-session
        terms get a higher default weight than seed terms so a proper noun the
        analyst spelled correctly outranks generic seed vocabulary.
        """
        if texts is None:
            return
        if isinstance(texts, str):
            texts = [texts]
        for text in texts:
            if not text:
                continue
            for term in extract_terms(str(text)):
                self._add(term, weight)

    def _add(self, term: str, weight: float) -> None:
        self._weights[term] = self._weights.get(term, 0.0) + weight

    # -- rendering ---------------------------------------------------------

    def terms(self) -> list[str]:
        """Top terms by weight (descending), capped at ``max_terms``."""
        ranked = sorted(
            self._weights.items(), key=lambda kv: (-kv[1], kv[0])
        )
        return [t for t, _ in ranked[: self.max_terms]]

    def initial_prompt(self) -> str | None:
        """Render the prior as a Whisper ``initial_prompt`` string, or None.

        Returns ``None`` when empty so callers transcribe with Whisper's default
        behaviour (no behavioural change when there is nothing to ground on).
        """
        terms = self.terms()
        if not terms:
            return None
        prompt = "Glossary: " + ", ".join(terms) + "."
        if len(prompt) > self.max_prompt_chars:
            prompt = prompt[: self.max_prompt_chars].rsplit(",", 1)[0] + "."
        return prompt

    def __len__(self) -> int:
        return len(self._weights)


def config_seed_text(config) -> str:
    """Build a seed string from the loaded config (always available at listen).

    Pulls the customer name, description, and configured domains — the same
    engagement inputs that feed grounding — into one block for :meth:`seed`.
    """
    if config is None:
        return ""
    parts: list[str] = []
    customer = getattr(config, "customer", None)
    if customer:
        parts.append(str(customer))
    description = getattr(config, "description", None)
    if description:
        parts.append(str(description))
    domains = getattr(config, "domains", None)
    if domains:
        parts.append(" ".join(str(d) for d in domains))
    return "\n".join(parts)
