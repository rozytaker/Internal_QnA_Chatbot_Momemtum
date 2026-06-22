"""
Schema retrieval layer (the 'LLM reads metadata catalogue' step on the
UC05 architecture slide) — now hybrid search + cross-encoder reranked:

  1. BM25 keyword ranking over the 16 table descriptions (always runs,
     no external dependency)
  2. Vector (semantic) ranking over the same descriptions, via local
     Ollama embeddings (nomic-embed-text) + ChromaDB
  3. Reciprocal Rank Fusion (RRF) combines both rankings into one
     candidate pool — this is what makes it "hybrid" rather than one
     method being a fallback for the other
  4. A cross-encoder (ms-marco-MiniLM-L-6-v2) reranks that pool against
     the raw question for the final top_k — precision step

We embed/index only TABLE DESCRIPTIONS, never row data — the LLM never
sees the data itself at this stage, only metadata, per the deck's
security architecture.
"""

import json
from pathlib import Path

import chromadb
from chromadb.api.types import EmbeddingFunction

from core.llm_engine import ollama_embed
from core.hybrid_search import BM25Index, reciprocal_rank_fusion
from core.reranker import rerank

METADATA_PATH = Path(__file__).parent.parent / "metadata" / "schema_catalogue.json"
CHROMA_DIR = Path(__file__).parent.parent / ".chroma_store"
COLLECTION_NAME = "momentum_schema_catalogue"
EMBED_MODEL = "nomic-embed-text"

RRF_POOL_SIZE = 8   # candidates carried from hybrid fusion into the rerank stage
RRF_K = 60          # standard RRF damping constant


def load_catalogue() -> dict:
    with open(METADATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def _table_document(name: str, meta: dict) -> str:
    cols = "; ".join(f"{c} ({d})" for c, d in meta["columns"].items())
    return (
        f"Table: {name}\n"
        f"Domain: {meta['domain']}\n"
        f"Description: {meta['description']}\n"
        f"Columns: {cols}"
    )


class OllamaEmbeddingFunction(EmbeddingFunction):
    """Calls the local Ollama /api/embeddings endpoint — no external network."""

    def __call__(self, input):  # noqa: A002 - chromadb's required signature
        return [ollama_embed(text, model=EMBED_MODEL) for text in input]


class SchemaRetriever:
    def __init__(self):
        self.catalogue = load_catalogue()
        self.tables_meta = self.catalogue["tables"]
        self._docs = {name: _table_document(name, meta) for name, meta in self.tables_meta.items()}

        # Keyword side of hybrid search — always available, zero external deps
        self.bm25 = BM25Index(self._docs)

        # Semantic side of hybrid search — needs a reachable Ollama
        self.vector_ready = False
        self._client = None
        self._collection = None
        self._try_build_vector_index()

    # -- semantic side -----------------------------------------------------------
    def _try_build_vector_index(self):
        try:
            self._client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            ef = OllamaEmbeddingFunction()
            ef(["ping"])  # liveness check before committing to the vector path
            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME, embedding_function=ef
            )
            if self._collection.count() != len(self._docs):
                self._client.delete_collection(COLLECTION_NAME)
                self._collection = self._client.create_collection(
                    name=COLLECTION_NAME, embedding_function=ef
                )
                self._collection.add(ids=list(self._docs.keys()), documents=list(self._docs.values()))
            self.vector_ready = True
        except Exception:
            self.vector_ready = False

    def _vector_rank(self, question: str, n: int) -> list:
        if not self.vector_ready or self._collection is None:
            return []
        try:
            res = self._collection.query(query_texts=[question], n_results=min(n, len(self._docs)))
            return res["ids"][0]
        except Exception:
            self.vector_ready = False
            return []

    # -- public API ----------------------------------------------------------------
    def retrieve(self, question: str, top_k: int = 4) -> list:
        """Hybrid retrieve (BM25 + vector, RRF-fused) then cross-encoder
        rerank. Returns a list of (table_name, schema_text) tuples."""
        pool_n = max(RRF_POOL_SIZE, top_k)

        bm25_ranked = self.bm25.rank(question)[:pool_n]
        vector_ranked = self._vector_rank(question, pool_n)

        fused = reciprocal_rank_fusion([bm25_ranked, vector_ranked], k=RRF_K)[:pool_n]
        if not fused:
            return []

        reranked_names = rerank(question, fused, text_for=lambda n: self._docs[n], top_k=top_k)
        return [(name, self._docs[name]) for name in reranked_names]
