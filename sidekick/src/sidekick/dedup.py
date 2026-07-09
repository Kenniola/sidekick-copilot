"""Deterministic near-duplicate detection for queued questions (Phase 2g).

Replaces the per-item fast-tier LLM dedup round-trip with a local similarity
check. Combines token-set Jaccard (robust to word reordering) with a
character-sequence ratio (sensitive to ordering and phrasing), taking the
stronger of the two. Fully deterministic and unit-testable.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

_WORD = re.compile(r"[a-z0-9]+")

# >80% overlap mirrors the intent of the previous LLM-based check.
DEFAULT_THRESHOLD = 0.8


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def similarity(a: str, b: str) -> float:
    """Return a 0..1 similarity score between two question strings."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    sa, sb = set(ta), set(tb)
    jaccard = len(sa & sb) / len(sa | sb)
    seq = SequenceMatcher(None, " ".join(ta), " ".join(tb)).ratio()
    return max(jaccard, seq)


def find_duplicate(
    question: str, candidates: list[str], threshold: float = DEFAULT_THRESHOLD
) -> int | None:
    """Return the index of the most similar candidate at/above ``threshold``.

    Args:
        question: the new question to test.
        candidates: previously-seen questions to compare against.
        threshold: minimum similarity (0..1) to count as a duplicate.

    Returns:
        The index of the best-matching candidate, or ``None`` if none reach
        the threshold.
    """
    best_idx: int | None = None
    best_score = 0.0
    for i, candidate in enumerate(candidates):
        score = similarity(question, candidate)
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx if best_score >= threshold else None
