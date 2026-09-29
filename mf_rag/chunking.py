"""STAGE 2 - Chunking.

Strategy (chosen after measuring this corpus, see README "Chunking decision"):
  Type A facts  -> atomic, one labelled fact per chunk, never split.
  Type B prose  -> recursive character split on sentence boundaries, then packed.

The corpus is a labelled fact sheet, not flowing prose, so fixed-size windows
break the unit that matters (a label separated from its value). Fixed-size is
kept only as a measured baseline for comparison.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterable, Sequence

from .config import (
    CHUNK_TOKEN_HARD_MAX,
    CHUNK_TOKEN_OVERLAP,
    CHUNK_TOKEN_TARGET,
    MAX_EMBEDDING_TOKENS,
    TOKENIZER_MODEL,
)
from .ingest import Document, Fact, ProseBlock

SEPARATORS = ("\n\n", "\n", ". ", "; ", ", ", " ", "")
CountTokens = Callable[[str], int]


def word_estimate(text: str) -> int:
    return max(1, len(text.split()))


_TOKENISER = None


def count_tokens_minilm(text: str) -> int:
    """Exact wordpiece count for the embedding model, with a word fallback."""
    global _TOKENISER
    if _TOKENISER is None:
        try:
            from transformers import AutoTokenizer

            _TOKENISER = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
        except Exception:  # noqa: BLE001
            _TOKENISER = False
    if _TOKENISER is False:
        return word_estimate(text)
    return len(_TOKENISER.tokenize(text))


@dataclass
class Chunk:
    chunk_id: str
    text: str
    cite_text: str
    kind: str
    section: str
    heading: str
    scheme_name: str
    scheme_short: str
    category: str
    plan: str
    source_url: str
    field_name: str = ""
    as_of: str = ""
    retrieved_at: str = ""
    token_count: int = 0
    ordinal: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def metadata(self) -> dict:
        return {
            "scheme_name": self.scheme_name,
            "scheme_short": self.scheme_short,
            "category": self.category,
            "plan": self.plan,
            "section": self.section,
            "heading": self.heading,
            "kind": self.kind,
            "field": self.field_name,
            "as_of": self.as_of,
            "source_url": self.source_url,
            "retrieved_at": self.retrieved_at,
            "ordinal": self.ordinal,
        }


def make_id(*parts: str) -> str:
    digest = hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"{parts[0]}_{digest}"


def _split_recursive(text: str, count: Callable[[str], int], limit: int) -> list[str]:
    if count(text) <= limit:
        return [text]
    for index, sep in enumerate(SEPARATORS):
        if sep not in text:
            continue
        parts = text.split(sep)
        if len(parts) == 1:
            continue
        rebuilt = sep if sep.strip() else ""
        chunks: list[str] = []
        buffer = ""
        for part in parts:
            candidate = part if not buffer else buffer + rebuilt + part
            if count(candidate) <= limit:
                buffer = candidate
            else:
                if buffer:
                    chunks.append(buffer)
                if count(part) <= limit:
                    buffer = part
                else:
                    deeper = _split_recursive(part, count, limit) if index + 1 < len(SEPARATORS) else [part]
                    chunks.extend(deeper[:-1])
                    buffer = deeper[-1] if deeper else ""
        if buffer:
            chunks.append(buffer)
        return [c.strip() for c in chunks if c.strip()]
    words = text.split()
    return [" ".join(words[i : i + limit]) for i in range(0, len(words), limit)]


def _pack(
    parts: Sequence[str],
    count: Callable[[str], int],
    target: int,
    overlap: int,
    hard_max: int = CHUNK_TOKEN_HARD_MAX,
) -> list[str]:
    packed: list[str] = []
    buffer: list[str] = []
    for part in parts:
        candidate = " ".join(buffer + [part])
        if buffer and count(candidate) > target:
            packed.append(" ".join(buffer))
            if overlap > 0:
                tail = " ".join(buffer).split()
                buffer = [" ".join(tail[-overlap:])] if tail else []
            else:
                buffer = []
            # The overlap seed sits on top of a part that is already near `target`,
            # so the pair can breach the hard ceiling. Overlap is a nicety; the
            # ceiling is a contract, so drop the overlap instead of the ceiling.
            if buffer and count(" ".join(buffer + [part])) > hard_max:
                buffer = []
        buffer.append(part)
    if buffer:
        packed.append(" ".join(buffer))
    # Safety net: `parts` are produced at `target`, so a single part cannot breach
    # `hard_max` on its own, but never emit a chunk that violates the invariant.
    final: list[str] = []
    for piece in packed:
        if count(piece) > hard_max:
            final.extend(_split_recursive(piece, count, hard_max))
        else:
            final.append(piece)
    return final


def chunk_fact(document: Document, fact: Fact) -> Chunk:
    cite = f"{fact.label}: {fact.value}" + (f" (as on {fact.as_of})" if fact.as_of else "")
    text = (
        f"{document.scheme_name} ({document.source.category}). "
        f"{fact.section}. {fact.label}: {fact.value}"
    )
    return Chunk(
        chunk_id=make_id(document.source.key, "fact", fact.field_name),
        text=text,
        cite_text=cite,
        kind="fact",
        section=fact.section,
        heading=fact.label,
        scheme_name=document.scheme_name,
        scheme_short=document.source.scheme_short,
        category=document.source.category,
        plan=document.source.plan,
        source_url=document.url,
        field_name=fact.field_name,
        as_of=fact.as_of or "",
        retrieved_at=document.retrieved_at,
    )


def chunk_prose(document: Document, block: ProseBlock, count: CountTokens) -> list[Chunk]:
    # Every prose chunk carries a scheme/category/heading prefix. Reserve its tokens
    # up front, otherwise the body can be legal on its own and the final chunk_text
    # still breaches CHUNK_TOKEN_HARD_MAX once the prefix is prepended.
    prefix = f"{document.scheme_name} ({document.source.category}). {block.heading}. "
    budget = max(1, CHUNK_TOKEN_HARD_MAX - count(prefix))
    target = min(CHUNK_TOKEN_TARGET, budget)
    parts = _split_recursive(block.text, count, target)
    packed = _pack(parts, count, target, CHUNK_TOKEN_OVERLAP, budget)
    chunks: list[Chunk] = []
    for index, body in enumerate(packed):
        cite = body if index == 0 else f"{block.heading} (continued): {body}"
        text = f"{document.scheme_name} ({document.source.category}). {block.heading}. {body}"
        chunks.append(
            Chunk(
                chunk_id=make_id(document.source.key, "prose", block.heading, str(index)),
                text=text,
                cite_text=cite,
                kind="prose",
                section=block.section,
                heading=block.heading,
                scheme_name=document.scheme_name,
                scheme_short=document.source.scheme_short,
                category=document.source.category,
                plan=document.source.plan,
                source_url=document.url,
                retrieved_at=document.retrieved_at,
            )
        )
    return chunks


def chunk_document(document: Document, count: CountTokens) -> list[Chunk]:
    chunks = [chunk_fact(document, fact) for fact in document.facts]
    for block in document.prose:
        chunks.extend(chunk_prose(document, block, count))
    seen: set[str] = set()
    unique: list[Chunk] = []
    per_section: dict[str, int] = {}
    for chunk in chunks:
        fingerprint = chunk.text.strip().lower()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        chunk.token_count = count(chunk.text)
        chunk.ordinal = per_section.get(chunk.section, 0)
        per_section[chunk.section] = chunk.ordinal + 1
        unique.append(chunk)
    return unique


def chunk_corpus(documents: Iterable[Document], count: CountTokens) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, count))
    return chunks


def fixed_size_baseline(document: Document, size: int = 180, overlap: int = 40) -> list[str]:
    text = "\n".join(
        [f"{f.label}: {f.value}" for f in document.facts]
        + [f"{p.heading}. {p.text}" for p in document.prose]
    )
    words = text.split()
    step = max(1, size - overlap)
    return [" ".join(words[i : i + size]) for i in range(0, len(words), step)]


def fixed_size_chunks(document: Document, size: int = 180, overlap: int = 40) -> list[Chunk]:
    pieces = fixed_size_baseline(document, size=size, overlap=overlap)
    chunks: list[Chunk] = []
    for index, body in enumerate(pieces):
        chunks.append(
            Chunk(
                chunk_id=make_id(document.source.key, "fixed", str(index)),
                text=f"{document.scheme_name} ({document.source.category}). {body}",
                cite_text=body,
                kind="fixed",
                section="mixed",
                heading="mixed window",
                scheme_name=document.scheme_name,
                scheme_short=document.source.scheme_short,
                category=document.source.category,
                plan=document.source.plan,
                source_url=document.url,
                retrieved_at=document.retrieved_at,
            )
        )
    return chunks


def fact_integrity(pieces: Sequence[str], facts: Sequence[Fact]) -> dict:
    """How often a labelled fact survives as one retrievable unit.

    cooccur_ratio is the metric that matters: the label and its value must land
    in the SAME chunk, otherwise retrieval can return "1% if redeemed within 1
    year" with no indication of what it refers to.
    """
    total = len(facts)
    if not total:
        return {"facts": 0, "chunks": len(pieces), "cooccur_ratio": 1.0, "avg_words": 0.0}

    cooccur = 0
    orphaned: list[str] = []
    for fact in facts:
        if not fact.value:
            continue
        if any(fact.value in piece and fact.label in piece for piece in pieces):
            cooccur += 1
        else:
            orphaned.append(f"{fact.label}={fact.value}")

    return {
        "facts": total,
        "chunks": len(pieces),
        "cooccur": cooccur,
        "cooccur_ratio": round(cooccur / total, 4),
        "orphaned": orphaned[:6],
        "avg_words": round(sum(len(p.split()) for p in pieces) / len(pieces), 1) if pieces else 0.0,
    }


__all__ = [
    "Chunk",
    "chunk_corpus",
    "chunk_document",
    "count_tokens_minilm",
    "fact_integrity",
    "fixed_size_baseline",
    "word_estimate",
    "MAX_EMBEDDING_TOKENS",
    "CHUNK_TOKEN_HARD_MAX",
    "field",
    "re",
]
