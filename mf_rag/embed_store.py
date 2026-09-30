"""STAGE 3 (embedding) and STAGE 4 (vector store).

Embeds chunks with sentence-transformers/all-MiniLM-L6-v2 and upserts them into
a persistent ChromaDB collection using cosine distance.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Sequence

import chromadb
from chromadb.config import Settings

from .chunking import Chunk
from .config import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    CHUNKS_PATH,
    DATA_DIR,
    EMBEDDING_MODEL,
)
from .ingest import documents_to_json, load_documents

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

# Process-local caches. Correct for Streamlit's single process and for the CLI, wrong for
# multi-worker serving: each worker would load its own model copy and its own Chroma handle
# against the same on-disk store. Do not move these to a process pool in this project.
_MODEL: SentenceTransformer | None = None
_CLIENT: chromadb.ClientAPI | None = None


@dataclass
class VectorHit:
    chunk_id: str
    score: float
    text: str
    metadata: dict


@dataclass
class IndexStats:
    chunks: int
    dimensions: int
    model: str
    collection: str

    def summary(self) -> str:
        return (
            f"{self.chunks} chunks | {self.dimensions}-dim | {self.model} | "
            f"collection={self.collection}"
        )


def get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer(EMBEDDING_MODEL)
    return _MODEL


def get_client() -> chromadb.ClientAPI:
    global _CLIENT
    if _CLIENT is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _CLIENT = chromadb.PersistentClient(
            path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False)
        )
    return _CLIENT


def get_collection():
    return get_client().get_or_create_collection(
        name=CHROMA_COLLECTION, metadata={"hnsw:space": "cosine", "embed_model": EMBEDDING_MODEL}
    )


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    model = get_model()
    vectors = model.encode(
        list(texts),
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return [vector.tolist() for vector in vectors]


def upsert_chunks(chunks: Sequence[Chunk], *, batch_size: int = 64) -> IndexStats:
    collection = get_collection()
    if collection.count():
        collection.delete(where={"chunk_id": {"$in": [c.chunk_id for c in chunks]}})

    for offset in range(0, len(chunks), batch_size):
        batch = chunks[offset : offset + batch_size]
        collection.upsert(
            ids=[c.chunk_id for c in batch],
            documents=[c.text for c in batch],
            metadatas=[{**c.metadata, "chunk_id": c.chunk_id, "cite_text": c.cite_text} for c in batch],
            embeddings=embed_texts([c.text for c in batch]),
        )
    dimensions = len(embed_texts(["dimension probe"])[0])
    return IndexStats(
        chunks=collection.count(),
        dimensions=dimensions,
        model=EMBEDDING_MODEL,
        collection=CHROMA_COLLECTION,
    )


def query_vector(
    query: str, top_k: int, *, where: dict | None = None
) -> list[VectorHit]:
    collection = get_collection()
    if not collection.count():
        return []
    result = collection.query(
        query_embeddings=embed_texts([query]),
        n_results=min(top_k, collection.count()),
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    hits: list[VectorHit] = []
    for doc_id, document, metadata, distance in zip(
        result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        hits.append(
            VectorHit(
                chunk_id=doc_id,
                score=1.0 - float(distance),
                text=document or "",
                metadata=metadata or {},
            )
        )
    return hits


def persist_chunks(chunks: Iterable[Chunk]) -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    with CHUNKS_PATH.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
            written += 1
    return written


def load_chunks() -> list[Chunk]:
    if not CHUNKS_PATH.exists():
        return []
    chunks: list[Chunk] = []
    for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            chunks.append(Chunk(**json.loads(line)))
    return chunks


def persist_documents(documents) -> int:
    path = DATA_DIR / "documents.json"
    path.write_text(json.dumps(documents_to_json(documents), ensure_ascii=False, indent=2), encoding="utf-8")
    return len(list(documents))


def index_exists() -> bool:
    """True only if the chunks and the collection are both present and readable.

    An unreadable store counts as missing on purpose. `requirements.txt` pins
    `chromadb>=1.5.9` with no upper bound, so a deploy can install a version whose on-disk
    schema does not match the committed `data/chroma/`. Raising here would take the app past
    the self-heal in `app.ensure_index()` and surface a raw traceback instead of a rebuild,
    which is the one failure mode that leaves a hosted deploy permanently dead.
    """
    if not CHUNKS_PATH.exists():
        return False
    try:
        return get_collection().count() > 0
    except Exception:
        return False


def reset_store() -> None:
    """Delete the Chroma store. The index is reproducible from source, so a store that
    cannot be read is never worth keeping - and it is the one thing blocking a rebuild."""
    global _CLIENT
    if _CLIENT is not None:
        try:
            _CLIENT.delete_collection(CHROMA_COLLECTION)
        except Exception:
            pass
        try:
            _CLIENT.reset()
        except Exception:
            pass
        _CLIENT = None
    shutil.rmtree(CHROMA_DIR, ignore_errors=True)


def build_index(*, refresh: bool = False) -> IndexStats:
    from .chunking import chunk_corpus, count_tokens_minilm

    documents = load_documents(refresh=refresh)
    chunks = chunk_corpus(documents, count_tokens_minilm)
    persist_chunks(chunks)
    persist_documents(documents)
    try:
        return upsert_chunks(chunks)
    except Exception:
        # A store left over from an incompatible chromadb cannot be written to. It is
        # derived data, so drop it and build once more rather than leaving the deploy
        # stuck on an error it cannot clear from a browser.
        reset_store()
        return upsert_chunks(chunks)
