"""Compare chunking strategies on this corpus to justify the choice in the README."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mf_rag.chunking import (
    CHUNK_TOKEN_HARD_MAX,
    chunk_document,
    count_tokens_minilm,
    fact_integrity,
    fixed_size_baseline,
)
from mf_rag.config import DATA_DIR
from mf_rag.ingest import load_documents

documents = load_documents()
report: dict = {
    "tokenizer": "sentence-transformers/all-MiniLM-L6-v2",
    "hard_max_tokens": CHUNK_TOKEN_HARD_MAX,
    "metric": "cooccur_ratio = share of labelled facts whose label AND value land in the same chunk",
    "metric_is_diagnostic_only": (
        "cooccur_ratio scores 1.0 for EVERY strategy measured here, including fixed-size "
        "windows, because every labelled fact on these five pages is short enough that a "
        "fixed window usually still swallows the whole line. It therefore does NOT "
        "discriminate between the strategies and must not be read as evidence for the "
        "structure-aware choice. The discriminating evidence is scripts/benchmark_retrieval.py, "
        "which measures whether the right fact is actually retrievable."
    ),
    "what_this_report_does_show": [
        "chunks_per_doc: structure-aware makes ~7x more chunks, because facts become their own "
        "retrievable units instead of being buried in prose windows.",
        "chunks_over_hard_max: 0 for the chosen strategy, i.e. every chunk fits the embedding "
        "model's 256-token window. Fixed windows above ~180 words cannot promise this.",
        "avg_tokens_per_chunk: the chosen chunks are small, which is what keeps retrieval precise.",
    ],
    "real_evidence": "scripts/benchmark_retrieval.py (dense-only top-1/3/5 hit rates)",
}

for size in (120, 180, 260, 400):
    ratios, chunks, words, orphaned = [], [], [], []
    for document in documents:
        stats = fact_integrity(fixed_size_baseline(document, size=size, overlap=40), document.facts)
        ratios.append(stats["cooccur_ratio"])
        chunks.append(stats["chunks"])
        words.append(stats["avg_words"])
        orphaned.extend(stats["orphaned"])
    report[f"fixed_size_{size}"] = {
        "cooccur_ratio": round(sum(ratios) / len(ratios), 4),
        "chunks_per_doc": round(sum(chunks) / len(chunks), 1),
        "avg_words_per_chunk": round(sum(words) / len(words), 1),
        "example_orphans": orphaned[:5],
    }

ratios, chunks, words, fact_chunks, prose_chunks = [], [], [], [], []
all_orphans: list[str] = []
for document in documents:
    built = chunk_document(document, count_tokens_minilm)
    stats = fact_integrity([c.cite_text for c in built], document.facts)
    ratios.append(stats["cooccur_ratio"])
    chunks.append(len(built))
    words.append(sum(c.token_count for c in built) / len(built))
    fact_chunks.append(sum(1 for c in built if c.kind == "fact"))
    prose_chunks.append(sum(1 for c in built if c.kind == "prose"))
    all_orphans.extend(stats["orphaned"])
    over = [c.token_count for c in built if c.token_count > CHUNK_TOKEN_HARD_MAX]
    assert not over, f"{document.source.key}: chunks over hard max: {over}"

report["chosen_structure_aware"] = {
    "cooccur_ratio": round(sum(ratios) / len(ratios), 4),
    "chunks_per_doc": round(sum(chunks) / len(chunks), 1),
    "fact_chunks_per_doc": round(sum(fact_chunks) / len(fact_chunks), 1),
    "prose_chunks_per_doc": round(sum(prose_chunks) / len(prose_chunks), 1),
    "avg_tokens_per_chunk": round(sum(words) / len(words), 1),
    "chunks_over_hard_max": 0,
    "example_orphans": all_orphans[:5],
}

print(json.dumps(report, indent=2, ensure_ascii=False))
(DATA_DIR / "chunking_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
