"""
Document side of the unified copilot. A user can upload PDFs (policy
wording, procedure manuals, circulars) and ask about them in the SAME
chat as the structured-data questions — the router in llm_engine decides
per-question which source(s) to use.

Retrieval here uses the same hybrid + rerank pattern as core/retriever.py:
  1. BM25 keyword ranking over the uploaded chunks
  2. Vector (semantic) ranking over the same chunks (Ollama + ChromaDB)
  3. Reciprocal Rank Fusion combines both into one candidate pool
  4. A cross-encoder (ms-marco-MiniLM-L-6-v2) reranks that pool for the
     final top_k — this matters more here than for the 16-table schema
     retriever, since a single PDF can have far more candidate chunks
     than there are tables, so precision at this step counts for more.

Everything here is in-memory and scoped to the current browser session:
nothing is written to disk, nothing persists after the session ends, and
nothing is shared between users.
"""

import io
import re

import chromadb

from core.llm_engine import ollama_embed
from core.hybrid_search import BM25Index, reciprocal_rank_fusion
from core.reranker import rerank

EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE = 1100
CHUNK_OVERLAP = 150
TOP_K = 5
RRF_POOL_SIZE = 10
RRF_K = 60


class _OllamaEmbeddingFunction:
    def __call__(self, input):  # noqa: A002 - chromadb's required signature
        return [ollama_embed(text, model=EMBED_MODEL) for text in input]


def _chunk_page_text(text: str, page_num: int, filename: str) -> list:
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({"text": chunk, "filename": filename, "page": page_num})
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def extract_chunks(file_bytes: bytes, filename: str) -> list:
    from pypdf import PdfReader  # imported lazily so the app still boots if pypdf is missing

    reader = PdfReader(io.BytesIO(file_bytes))
    all_chunks = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text.strip():
            all_chunks.extend(_chunk_page_text(text, i, filename))
    return all_chunks


class DocumentStore:
    """One instance per browser session (held in st.session_state, never
    st.cache_resource — that would leak documents across users)."""

    def __init__(self):
        self._client = chromadb.Client()  # in-memory, ephemeral
        self._collection = None
        self._chunks: dict = {}  # chunk_id -> {"text", "filename", "page"}
        self.bm25 = None
        self.filenames = []
        self.vector_ready = True

    def _ensure_collection(self):
        if self._collection is None:
            self._collection = self._client.create_collection(
                name="session_documents", embedding_function=_OllamaEmbeddingFunction()
            )

    def _rebuild_bm25(self):
        self.bm25 = BM25Index({cid: c["text"] for cid, c in self._chunks.items()})

    def add_pdf(self, file_bytes: bytes, filename: str) -> int:
        chunks = extract_chunks(file_bytes, filename)
        if not chunks:
            return 0
        self._ensure_collection()
        ids = [f"{filename}::{i}" for i in range(len(chunks))]
        docs = [c["text"] for c in chunks]
        metas = [{"filename": c["filename"], "page": c["page"]} for c in chunks]
        try:
            self._collection.add(ids=ids, documents=docs, metadatas=metas)
            self.vector_ready = True
        except Exception:
            self.vector_ready = False  # embeddings unreachable -> BM25-only hybrid

        for cid, c in zip(ids, chunks):
            self._chunks[cid] = c
        self._rebuild_bm25()

        if filename not in self.filenames:
            self.filenames.append(filename)
        return len(chunks)

    def _vector_rank(self, question: str, n: int) -> list:
        if not self.vector_ready or self._collection is None:
            return []
        try:
            res = self._collection.query(query_texts=[question], n_results=min(n, len(self._chunks)))
            return res["ids"][0]
        except Exception:
            self.vector_ready = False
            return []

    def retrieve(self, question: str, top_k: int = TOP_K) -> list:
        """Hybrid retrieve (BM25 + vector, RRF-fused) then cross-encoder
        rerank. Returns a list of {"text", "filename", "page"} dicts."""
        if not self._chunks:
            return []
        pool_n = max(RRF_POOL_SIZE, top_k)

        bm25_ranked = self.bm25.rank(question)[:pool_n] if self.bm25 else []
        vector_ranked = self._vector_rank(question, pool_n)

        fused = reciprocal_rank_fusion([bm25_ranked, vector_ranked], k=RRF_K)[:pool_n]
        if not fused:
            return []

        reranked_ids = rerank(question, fused, text_for=lambda cid: self._chunks[cid]["text"], top_k=top_k)
        return [self._chunks[cid] for cid in reranked_ids]
