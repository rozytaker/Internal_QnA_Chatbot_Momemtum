"""
Cross-encoder reranking — the final precision step after hybrid (BM25 +
vector) retrieval, shared by both pipelines (schema tables and document
passages).

A cross-encoder scores (question, candidate) pairs jointly — unlike the
bi-encoder used for embeddings, which scores each side independently —
so it's slower per pair but meaningfully more precise at picking the
truly best matches out of a small candidate pool. We only ever rerank
the already-narrowed hybrid shortlist (~8-10 items), never the full
corpus, so the extra cost is negligible.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (~80MB), loaded once and
cached for the life of the process. It's pulled from the HuggingFace Hub
the first time it's used, then cached locally — if that first download
can't happen (no internet, offline environment), reranking is skipped
and callers fall back to the hybrid (pre-rerank) order instead of
crashing the app.
"""

import streamlit as st

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@st.cache_resource(show_spinner="Loading reranker model (first run only)…")
def get_cross_encoder():
    """Returns the cached CrossEncoder instance, or None if it can't be
    loaded — callers must handle None by skipping reranking gracefully."""
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder(CROSS_ENCODER_MODEL)
    except Exception:
        return None


def cross_encoder_ready() -> bool:
    return get_cross_encoder() is not None


def rerank(query: str, candidates: list, text_for, top_k: int) -> list:
    """
    candidates : list of opaque items (table names, chunk dicts, etc.)
    text_for   : function(item) -> str, the text to score against `query`
    Returns the subset of `candidates` re-sorted by cross-encoder
    relevance, truncated to top_k. Falls back to the candidates' existing
    (hybrid pre-rerank) order, truncated to top_k, if the model isn't
    available.
    """
    if not candidates:
        return []
    model = get_cross_encoder()
    if model is None:
        return candidates[:top_k]

    pairs = [(query, text_for(c)) for c in candidates]
    scores = model.predict(pairs)
    ranked = [c for _, c in sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)]
    return ranked[:top_k]
