"""STAGE 5 - Retrieval. Hybrid dense (Chroma) + sparse (BM25) with RRF fusion.

Why hybrid: this corpus is numeric fact lookups ("expense ratio of HDFC ELSS",
"minimum SIP"). Dense retrieval on all-MiniLM-L6-v2 is weak on exact tokens like
"1.21%", "NIFTY 500 TRI" and "80C", while BM25 is weak on paraphrases. Reciprocal
Rank Fusion combines the two orderings without needing score calibration.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from rank_bm25 import BM25Okapi

from .chunking import Chunk
from .config import (
    RETRIEVE_BM25_K,
    RETRIEVE_FINAL_K,
    RETRIEVE_VECTOR_K,
    RRF_K,
)
from .embed_store import get_model, query_vector
from .sources import SOURCES

TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?%?")
BOOST_FIELDS = ("expense_ratio", "exit_load", "min_sip", "min_lumpsum", "benchmark", "riskometer", "lock_in")

# Single source of truth for scheme-name resolution. The answer stage imports this
# rather than keeping its own copy: the two maps had already drifted, so an alias
# could boost retrieval without resolving in the answer (or vice versa).
SCHEME_ALIASES: dict[str, str] = {
    "large cap": "Large Cap",
    "bluechip": "Large Cap",
    "flexi cap": "Flexi Cap",
    "flexicap": "Flexi Cap",
    "hdfc equity fund": "Flexi Cap",
    "elss": "ELSS",
    "tax saver": "ELSS",
    "80c": "ELSS",
    "small cap": "Small Cap",
    "balanced advantage": "Balanced Advantage",
}


def detect_scheme(query: str) -> str | None:
    """Resolve a scheme short name from a query, or None if it names no scheme.

    Two passes: exact registered/display names first, then aliases. Order matters
    because a page title can itself contain an alias substring.
    """
    lowered = query.lower()
    for source in SOURCES:
        names = {
            source.scheme_short.lower(),
            source.scheme_name.lower(),
            source.display_name.lower(),
        }
        for candidate in names:
            if candidate and candidate in lowered:
                return source.scheme_short
    for alias, short in SCHEME_ALIASES.items():
        if alias in lowered:
            return short
    return None


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    cite_text: str
    scheme_name: str
    scheme_short: str
    category: str
    section: str
    heading: str
    kind: str
    field_name: str
    source_url: str
    rrf_score: float
    vector_rank: int | None = None
    bm25_rank: int | None = None
    vector_score: float | None = None
    bm25_score: float | None = None
    ordinal: int = 0
    reasons: list[str] = field(default_factory=list)


class HybridRetriever:
    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self.chunks = list(chunks)
        self.by_id = {c.chunk_id: c for c in self.chunks}
        corpus = [f"{c.heading} {c.cite_text} {c.text}" for c in self.chunks]
        self.bm25 = BM25Okapi([tokenize(doc) for doc in corpus]) if corpus else None

    def _bm25(
        self, query: str, top_k: int, allowed: set[str] | None = None
    ) -> list[tuple[str, float, int]]:
        if not self.bm25:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        order = sorted(
            (
                (index, float(score))
                for index, score in enumerate(scores)
                if score > 0 and (allowed is None or self.chunks[index].chunk_id in allowed)
            ),
            key=lambda pair: pair[1],
            reverse=True,
        )[:top_k]
        return [
            (self.chunks[index].chunk_id, score, rank)
            for rank, (index, score) in enumerate(order, start=1)
        ]

    def _detect_scheme(self, query: str) -> str | None:
        return detect_scheme(query)

    def _dense(
        self, query: str, top_k: int, allowed: set[str] | None = None
    ) -> list[VectorHit]:
        where = None
        if allowed is not None:
            ids = sorted(allowed)
            if not ids:
                return []
            where = {"chunk_id": {"$in": ids}}
        return query_vector(query, top_k, where=where)

    def search(
        self, query: str, top_k: int = RETRIEVE_FINAL_K, *, section: str | None = None,
        allowed_schemes: set[str] | None = None,
    ) -> list[RetrievedChunk]:
        if not self.chunks:
            return []

        allowed = (
            {c.chunk_id for c in self.chunks if c.section == section} if section else None
        )
        if allowed_schemes is not None:
            scoped = {c.chunk_id for c in self.chunks if c.scheme_short in allowed_schemes}
            # Intersect rather than replace: a section filter and a scheme filter can both
            # apply. An empty intersection must return nothing, not everything.
            allowed = scoped if allowed is None else (allowed & scoped)
        if allowed is not None and not allowed:
            return []

        dense = self._dense(query, RETRIEVE_VECTOR_K, allowed)
        sparse = self._bm25(query, RETRIEVE_BM25_K, allowed)
        target_scheme = self._detect_scheme(query)

        scores: dict[str, float] = {}
        vector_rank: dict[str, int] = {}
        bm25_rank: dict[str, int] = {}
        vector_score: dict[str, float] = {}
        bm25_score: dict[str, float] = {}

        for rank, hit in enumerate(dense, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (RRF_K + rank)
            vector_rank[hit.chunk_id] = rank
            vector_score[hit.chunk_id] = hit.score
        for chunk_id, score, rank in sparse:
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
            bm25_rank[chunk_id] = rank
            bm25_score[chunk_id] = score

        results: list[RetrievedChunk] = []
        for chunk_id, rrf in scores.items():
            chunk = self.by_id[chunk_id]
            reasons: list[str] = []
            if chunk_id in vector_rank:
                reasons.append(f"dense rank {vector_rank[chunk_id]}")
            if chunk_id in bm25_rank:
                reasons.append(f"bm25 rank {bm25_rank[chunk_id]}")
            if target_scheme and chunk.scheme_short == target_scheme:
                rrf *= 1.35
                reasons.append(f"scheme match: {target_scheme}")
            if chunk.field_name in BOOST_FIELDS and target_scheme:
                rrf *= 1.05
                reasons.append(f"field boost: {chunk.field_name}")
            results.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=chunk.text,
                    cite_text=chunk.cite_text,
                    scheme_name=chunk.scheme_name,
                    scheme_short=chunk.scheme_short,
                    category=chunk.category,
                    section=chunk.section,
                    heading=chunk.heading,
                    kind=chunk.kind,
                    field_name=chunk.field_name,
                    source_url=chunk.source_url,
                    rrf_score=rrf,
                    vector_rank=vector_rank.get(chunk_id),
                    bm25_rank=bm25_rank.get(chunk_id),
                    vector_score=vector_score.get(chunk_id),
                    bm25_score=bm25_score.get(chunk_id),
                    ordinal=chunk.ordinal,
                    reasons=reasons,
                )
            )

        results.sort(key=lambda r: r.rrf_score, reverse=True)
        return results[:top_k]


# Process-local cache, same caveat as `_MODEL` / `_CLIENT` in embed_store: fine for the
# single-process Streamlit app and the CLI, but a multi-worker server would build one
# retriever (and one BM25 index) per worker.
_retriever: HybridRetriever | None = None


def get_retriever(chunks: Sequence[Chunk] | None = None) -> HybridRetriever:
    global _retriever
    if _retriever is None or chunks is not None:
        from .embed_store import load_chunks

        _retriever = HybridRetriever(chunks if chunks is not None else load_chunks())
    return _retriever


def embed_query(query: str) -> list[float]:
    return get_model().encode([query], normalize_embeddings=True)[0].tolist()
