"""Prompt token-budgeting helpers (Phase 2f).

The deep-tier ``suggest_questions`` prompt concatenates the recent transcript,
open threads, prior research, and grounding context. Any of those can grow
without bound over a long meeting, inflating latency and risking the model's
context window. ``clip`` enforces a per-block character budget (a cheap proxy
for tokens), keeping either the most recent tail or the leading head.
"""

from __future__ import annotations

_DEFAULT_MARKER = "… [truncated]"


def clip(text: str, max_chars: int, *, keep: str = "tail", marker: str = _DEFAULT_MARKER) -> str:
    """Clip ``text`` to at most ``max_chars`` characters.

    Args:
        text: the text to clip (returned unchanged if already within budget).
        max_chars: the maximum number of characters to keep, including marker.
        keep: ``"tail"`` keeps the end (most recent), ``"head"`` keeps the start.
        marker: the truncation indicator inserted at the cut.

    Returns:
        The original text if it fits, else a clipped version of exactly
        ``max_chars`` characters (when the budget allows for the marker).
    """
    if not text or len(text) <= max_chars:
        return text

    # If the budget is too small to fit the marker, just hard-cut.
    budget = max_chars - len(marker) - 1
    if budget <= 0:
        return text[:max_chars]

    if keep == "head":
        return text[:budget] + "\n" + marker
    return marker + "\n" + text[-budget:]
