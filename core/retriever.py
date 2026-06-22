"""
Schema retrieval layer (the 'LLM reads metadata catalogue' step on the
UC05 architecture slide).

We embed one document per table (description + columns + sample questions)
into a local ChromaDB collection. At query time we retrieve the tables most
relevant to the user's question and hand only that *metadata* — never row
data — to the LLM, exactly as specified in the deck's security architecture.

Embeddings are produced by a local Ollama model (nomic-embed-text), so the
retrieval step never leaves the firewall either. If Chroma or Ollama is not
reachable (e.g. embedding model not pulled yet) we fall back to a simple
keyword-overlap ranking so the demo never hard-fails on stage.
"""

import json
import re
from pathlib import Path

import chromadb
from chromadb.api.types import EmbeddingFunction

from core.llm_engine import OLLAMA_HOST, ollama_embed

METADATA_PATH = Path(__file__).parent.parent / "metadata" / "schema_catalogue.json"
CHROMA_DIR = Path(__file__).parent.parent / ".chroma_store"
COLLECTION_NAME = "momentum_schema_catalogue"
EMBED_MODEL = "nomic-embed-text"


def load_catalogue() -> dict:
    with open(METADATA_PATH) as f:
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
        self.vector_ready = False
        self._client = None
        self._collection = None
        self._try_build_vector_index()

    # -- vector path (ChromaDB + Ollama embeddings) -------------------------
    def _try_build_vector_index(self):
        try:
            self._client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            ef = OllamaEmbeddingFunction()
            # Quick liveness check before committing to the vector path
            ef(["ping"])
            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME, embedding_function=ef
            )
            if self._collection.count() != len(self.tables_meta):
                self._client.delete_collection(COLLECTION_NAME)
                self._collection = self._client.create_collection(
                    name=COLLECTION_NAME, embedding_function=ef
                )
                ids, docs = [], []
                for name, meta in self.tables_meta.items():
                    ids.append(name)
                    docs.append(_table_document(name, meta))
                self._collection.add(ids=ids, documents=docs)
            self.vector_ready = True
        except Exception:
            self.vector_ready = False

    # -- keyword fallback -----------------------------------------------------
    def _keyword_rank(self, question: str, top_k: int) -> list:
        q_tokens = set(re.findall(r"[a-z]+", question.lower()))
        scores = []
        for name, meta in self.tables_meta.items():
            doc = _table_document(name, meta).lower()
            doc_tokens = set(re.findall(r"[a-z]+", doc))
            overlap = len(q_tokens & doc_tokens)
            # small boost if the table or domain name literally appears
            if name.replace("_", " ") in question.lower():
                overlap += 5
            scores.append((overlap, name))
        scores.sort(reverse=True)
        return [name for score, name in scores[:top_k] if score > 0] or list(self.tables_meta)[:top_k]

    # -- public API -------------------------------------------------------------
    def retrieve(self, question: str, top_k: int = 4) -> list:
        """Return a list of (table_name, schema_text) for the most relevant tables."""
        if self.vector_ready:
            try:
                res = self._collection.query(query_texts=[question], n_results=top_k)
                names = res["ids"][0]
                return [(n, _table_document(n, self.tables_meta[n])) for n in names]
            except Exception:
                self.vector_ready = False  # drop to fallback for the rest of the session

        names = self._keyword_rank(question, top_k)
        return [(n, _table_document(n, self.tables_meta[n])) for n in names]
