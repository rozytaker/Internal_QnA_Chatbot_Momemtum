"""
Cross-encoder reranking step (optional).

Attempts to load the ms-marco-MiniLM cross-encoder from sentence-transformers.
If the model or library is unavailable the module degrades gracefully — callers
check cross_encoder_ready() before using rerank().
"""

from __future__ import annotations

_model = None
_ready = False


def _try_load():
    global _model, _ready
    try:
        from sentence_transformers import CrossEncoder  # type: ignore
        _model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        _ready = True
    except Exception:
        _ready = False


_try_load()


def cross_encoder_ready() -> bool:
    return _ready


def rerank(question: str, passages: list[str], top_k: int | None = None) -> list[str]:
    """Return passages reranked by relevance to question.

    Falls back to original order when the cross-encoder is unavailable.
    """
    if not _ready or _model is None:
        return passages[:top_k] if top_k else passages

    pairs = [(question, p) for p in passages]
    scores = _model.predict(pairs)
    ranked = sorted(zip(scores, passages), key=lambda x: x[0], reverse=True)
    result = [p for _, p in ranked]
    return result[:top_k] if top_k else result
