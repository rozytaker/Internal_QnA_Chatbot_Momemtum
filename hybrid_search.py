"""
Shared hybrid-search building blocks used by both retrieval pipelines —
schema tables (core/retriever.py) and document passages (core/doc_store.py).

Hybrid search here means: BM25 (keyword/lexical) and vector (semantic)
search both run on every query, and their rankings are fused with
Reciprocal Rank Fusion (RRF) — not a fallback chain where one only runs
if the other fails. If vector search genuinely is unavailable (e.g. Ollama
embeddings unreachable), fusion degrades gracefully to BM25-only, since
an empty ranked list simply contributes nothing to the fused score.
"""

import re

from rank_bm25 import BM25Okapi

# Minimal stopword list -- without this, BM25 gives non-trivial weight to
# words like "which"/"had"/"the" that appear incidentally across many
# table descriptions, and short documents get disproportionately boosted
# by BM25's document-length normalization when they happen to contain one.
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "which", "who", "whom", "what",
    "in", "on", "at", "by", "for", "of", "to", "and", "or", "as",
    "it", "its", "do", "does", "did", "has", "have", "having",
    "with", "from", "but", "not", "no", "had",
}


def tokenize(text: str) -> list:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


class BM25Index:
    """Thin wrapper around rank_bm25 keyed by item id, so callers don't
    have to juggle parallel lists themselves."""

    def __init__(self, items: dict):
        """items: {id: text}"""
        self.ids = list(items.keys())
        self._bm25 = BM25Okapi([tokenize(t) for t in items.values()]) if items else None

    def rank(self, query: str) -> list:
        """Returns every id, best-first by BM25 score (including zero
        scores at the bottom — harmless, RRF naturally deprioritises
        them). Empty list if the index itself is empty."""
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self.ids, scores), key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in ranked]


def reciprocal_rank_fusion(ranked_lists: list, k: int = 60) -> list:
    """
    ranked_lists: list of ranked-id lists, each already best-first.
    score(id) = sum over lists of 1 / (k + rank_in_that_list + 1).
    k=60 is the standard RRF damping constant from the original paper —
    large enough that rank position matters more than which list an item
    came from. Lists that are empty (e.g. vector search unavailable)
    simply contribute no score, so fusion degrades gracefully to
    whichever list(s) are actually present.
    """
    scores: dict = {}
    for ranked in ranked_lists:
        for rank, item_id in enumerate(ranked):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    return [item_id for item_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
