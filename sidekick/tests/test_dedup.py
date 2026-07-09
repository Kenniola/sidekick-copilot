"""Tests for the deterministic dedup helper (Phase 2g)."""

from sidekick.dedup import find_duplicate, similarity


class TestSimilarity:
    def test_identical_is_one(self):
        assert similarity("What is the CU limit?", "What is the CU limit?") == 1.0

    def test_word_reorder_high(self):
        # token-Jaccard is order-insensitive
        s = similarity("limit CU the what is", "what is the CU limit")
        assert s >= 0.8

    def test_unrelated_low(self):
        s = similarity(
            "What is the capacity throttling threshold?",
            "Who owns the Dataverse pipeline?",
        )
        assert s < 0.5

    def test_empty_is_zero(self):
        assert similarity("", "anything") == 0.0
        assert similarity("anything", "") == 0.0

    def test_punctuation_and_case_ignored(self):
        assert similarity("Fabric capacity?", "fabric   CAPACITY!!!") == 1.0


class TestFindDuplicate:
    def test_returns_best_match_index(self):
        candidates = [
            "Who owns the pipeline?",
            "What is the CU limit for F64?",
            "When does the contract renew?",
        ]
        idx = find_duplicate("What's the CU limit for F64?", candidates)
        assert idx == 1

    def test_returns_none_when_no_match(self):
        candidates = ["Who owns the pipeline?", "When does the contract renew?"]
        assert find_duplicate("What is the egress cost?", candidates) is None

    def test_empty_candidates_returns_none(self):
        assert find_duplicate("anything", []) is None

    def test_threshold_respected(self):
        candidates = ["What is the CU limit for F64?"]
        # A loose paraphrase below threshold should not match at 0.95
        idx = find_duplicate(
            "Tell me about capacity units", candidates, threshold=0.95
        )
        assert idx is None

    def test_picks_highest_when_multiple_above_threshold(self):
        candidates = [
            "What is the CU limit?",
            "What is the CU limit for the F64 SKU exactly?",
        ]
        idx = find_duplicate("What is the CU limit?", candidates)
        assert idx == 0  # exact match scores highest
