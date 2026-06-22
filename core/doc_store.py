"""
Document side of the unified copilot. A user can upload PDFs (policy
wording, procedure manuals, circulars) and ask about them in the SAME
chat as the structured-data questions — the router in llm_engine decides
per-question which source(s) to use.

Everything here is in-memory and scoped to the current browser session:
nothing is written to disk, nothing persists after the session ends, and
nothing is shared between users. This keeps the "zero data leaves the
firewall" promise intact for uploaded documents too.
"""

import io
import re

import chromadb

from core.llm_engine import ollama_embed

EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE = 1100
CHUNK_OVERLAP = 150
TOP_K = 5


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
        self._chunks = []  # parallel plain list, used for keyword fallback
        self.filenames = []
        self.vector_ready = True

    def _ensure_collection(self):
        if self._collection is None:
            self._collection = self._client.create_collection(
                name="session_documents", embedding_function=_OllamaEmbeddingFunction()
            )

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
            self.vector_ready = False  # embeddings unreachable -> keyword fallback only
        self._chunks.extend(chunks)
        if filename not in self.filenames:
            self.filenames.append(filename)
        return len(chunks)

    def _keyword_fallback(self, question: str, top_k: int) -> list:
        q_tokens = set(re.findall(r"[a-z]+", question.lower()))
        scored = []
        for c in self._chunks:
            doc_tokens = set(re.findall(r"[a-z]+", c["text"].lower()))
            scored.append((len(q_tokens & doc_tokens), c))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for score, c in scored[:top_k] if score > 0]

    def retrieve(self, question: str, top_k: int = TOP_K) -> list:
        if not self.filenames:
            return []
        if self.vector_ready and self._collection is not None:
            try:
                res = self._collection.query(query_texts=[question], n_results=min(top_k, len(self._chunks)))
                return [
                    {"text": doc, "filename": meta["filename"], "page": meta["page"]}
                    for doc, meta in zip(res["documents"][0], res["metadatas"][0])
                ]
            except Exception:
                self.vector_ready = False
        return self._keyword_fallback(question, top_k)
