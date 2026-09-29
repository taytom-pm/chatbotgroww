"""Benchmark: does structure-aware chunking retrieve the right fact better than fixed-size?

Builds a second, fixed-size corpus, embeds it into a scratch Chroma collection,
and compares dense-only retrieval precision against the structure-aware corpus
on a labelled set of fact queries. Output feeds the README's chunking rationale.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chromadb
from chromadb.config import Settings

from mf_rag.chunking import chunk_corpus, count_tokens_minilm, fixed_size_chunks
from mf_rag.config import DATA_DIR
from mf_rag.embed_store import embed_texts
from mf_rag.ingest import load_documents
from mf_rag.sources import SOURCES

CAVEATS = (
    "This benchmark is DENSE-ONLY and SMALL-SAMPLE, and both limits matter when reading it. "
    "Dense-only: it embeds the query and takes the nearest neighbours, with no BM25 leg, no RRF "
    "fusion, no scheme boost and no field boost. The shipped assistant runs hybrid retrieval with "
    "BM25, RRF and a field boost, so live accuracy is expected to be AT LEAST this good - but that "
    "is an expectation, not a measurement, and this file does not measure it. Small-sample: 35 "
    "labelled queries over one field set and five scheme pages, so a single query is worth about "
    "2.9 points and the percentages are coarse. What this benchmark does isolate is the CHUNKING "
    "difference: both corpora are embedded with the same model and queried the same way, so the "
    "gap between the two columns is attributable to chunk structure alone."
)

BENCH_FIELDS = ("expense_ratio", "exit_load", "min_sip", "min_lumpsum", "benchmark", "riskometer", "aum")
FIELD_QUESTION = {
    "expense_ratio": "What is the expense ratio",
    "exit_load": "What is the exit load",
    "min_sip": "What is the minimum SIP amount",
    "min_lumpsum": "What is the minimum lump sum investment",
    "benchmark": "What is the benchmark",
    "riskometer": "What is the riskometer level",
    "aum": "What is the fund size AUM",
}
SCRATCH = DATA_DIR / "chroma_bench"
TOP_K = 8  # dense-only candidate depth; hit rates are reported at 1, 3 and 5


def _rmtree_quietly(path: Path) -> None:
    """Delete the scratch collection, tolerating Chroma's open file handles.

    ChromaDB keeps the SQLite and HNSW files open for the life of the client, so on
    Windows `shutil.rmtree` fails with WinError 32. The scratch directory is rebuilt
    from scratch on every run anyway, so a failure here must never fail the benchmark.
    """
    for attempt in range(5):
        if not path.exists():
            return
        try:
            shutil.rmtree(path)
            return
        except (PermissionError, OSError):
            time.sleep(0.4 * (attempt + 1))
    print(f"note: could not delete {path} (still locked); it is rebuilt on the next run.")


@lru_cache(maxsize=1)
def _scratch_collection():
    client = chromadb.PersistentClient(path=str(SCRATCH), settings=Settings(anonymized_telemetry=False))
    return client.get_collection("bench_fixed")


def build_scratch(structured, fixed) -> None:
    """Embed BOTH corpora into the scratch store, freshly, with the same model.

    The shipped collection at data/chroma is deliberately never opened. Chroma rewrites
    its SQLite and HNSW files on open+query, so merely reading the production store
    rewrote every file in it and broke the "leave data/chroma untouched" guarantee.
    Building both legs here also makes the comparison symmetric: neither strategy is
    advantaged by coming from a differently-built store.
    """
    _rmtree_quietly(SCRATCH)
    client = chromadb.PersistentClient(path=str(SCRATCH), settings=Settings(anonymized_telemetry=False))
    for name, chunks in (("bench_struct", structured), ("bench_fixed", fixed)):
        collection = client.create_collection(name=name, metadata={"hnsw:space": "cosine"})
        for offset in range(0, len(chunks), 64):
            batch = chunks[offset : offset + 64]
            collection.upsert(
                ids=[c.chunk_id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[{**c.metadata, "chunk_id": c.chunk_id, "cite_text": c.cite_text} for c in batch],
                embeddings=embed_texts([c.text for c in batch]),
            )


@lru_cache(maxsize=2)
def _scratch_collection(name: str = "bench_fixed"):
    client = chromadb.PersistentClient(path=str(SCRATCH), settings=Settings(anonymized_telemetry=False))
    return client.get_collection(name)


def _dense_search(query: str, name: str, top_k: int = TOP_K) -> list[dict]:
    collection = _scratch_collection(name)
    result = collection.query(
        query_embeddings=embed_texts([query]),
        n_results=min(top_k, collection.count()),
        include=["metadatas", "distances"],
    )
    return [
        {"meta": meta or {}, "score": 1.0 - float(dist)}
        for meta, dist in zip(result["metadatas"][0], result["distances"][0])
    ]


def main() -> int:
    documents = load_documents()
    structured = chunk_corpus(documents, count_tokens_minilm)
    fixed: list = []
    for document in documents:
        fixed.extend(fixed_size_chunks(document, size=180, overlap=40))

    build_scratch(structured, fixed)

    rows = []
    for source in SOURCES:
        for field in BENCH_FIELDS:
            query = f"{FIELD_QUESTION[field]} of {source.display_name}?"
            want_scheme = source.scheme_short

            struct_hits = _dense_search(query, "bench_struct", top_k=TOP_K)
            fixed_hits = _dense_search(query, "bench_fixed", top_k=TOP_K)

            def _at(hits, depth):
                return any(
                    h["meta"].get("field") == field and h["meta"].get("scheme_short") == want_scheme
                    for h in hits[:depth]
                )

            def score_struct(hits):
                return (_at(hits, 1), _at(hits, 3), _at(hits, 5))

            def score_fixed(hits):
                want_value = None
                for chunk in structured:
                    if chunk.scheme_short == want_scheme and chunk.field_name == field:
                        want_value = chunk.cite_text
                        break
                if not want_value:
                    return False, False, False
                return (
                    want_value in hits[0]["meta"].get("cite_text", ""),
                    any(want_value in h["meta"].get("cite_text", "") for h in hits[:3]),
                    any(want_value in h["meta"].get("cite_text", "") for h in hits[:5]),
                )

            s1, s3, s5 = score_struct(struct_hits)
            f1, f3, f5 = score_fixed(fixed_hits)
            rows.append(
                {
                    "scheme": want_scheme,
                    "field": field,
                    "struct_top1": s1,
                    "struct_top3": s3,
                    "struct_top5": s5,
                    "fixed_top1": f1,
                    "fixed_top3": f3,
                    "fixed_top5": f5,
                    "struct_top1_kind": struct_hits[0]["meta"].get("kind"),
                    "struct_top1_score": round(struct_hits[0]["score"], 4),
                    "fixed_top1_score": round(fixed_hits[0]["score"], 4),
                }
            )

    n = len(rows)

    def rate(key: str, depth: str) -> float:
        return round(sum(r[key] for r in rows) / n, 4)

    report = {
        "caveats": CAVEATS,
        "measurement": {
            "retrieval": f"dense-only, top-{TOP_K} candidates per query",
            "hit_rule_structure_aware": "returned chunk's field_name == gold field AND scheme_short == gold scheme",
            "hit_rule_fixed_size": (
                "fixed windows carry no field labels, so a hit means the gold fact's full "
                "cite_text appears somewhere in the returned window. This is a LENIENT rule "
                "that flatters the fixed-window baseline, so the gap below is a lower bound."
            ),
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "labelled_queries": n,
        },
        "queries": n,
        "corpus_chunks": {"structure_aware": len(structured), "fixed_180": len(fixed)},
        "structure_aware": {
            "top1": rate("struct_top1", "top1"),
            "top3": rate("struct_top3", "top3"),
            "top5": rate("struct_top5", "top5"),
        },
        "fixed_size_180": {
            "top1": rate("fixed_top1", "top1"),
            "top3": rate("fixed_top3", "top3"),
            "top5": rate("fixed_top5", "top5"),
        },
        "per_query": rows,
        "misses": [r for r in rows if not r["struct_top5"] or not r["fixed_top5"]],
    }

    struct, fixed_r = report["structure_aware"], report["fixed_size_180"]
    print(f"Dense-only retrieval benchmark - {n} labelled queries, top-{TOP_K} candidates each\n")
    print(f"{'strategy':<22}{'chunks':>8}{'top-1':>10}{'top-3':>10}{'top-5':>10}")
    print("-" * 60)
    print(
        f"{'structure-aware':<22}{report['corpus_chunks']['structure_aware']:>8}"
        f"{struct['top1']:>10.1%}{struct['top3']:>10.1%}{struct['top5']:>10.1%}"
    )
    print(
        f"{'fixed-size 180 words':<22}{report['corpus_chunks']['fixed_180']:>8}"
        f"{fixed_r['top1']:>10.1%}{fixed_r['top3']:>10.1%}{fixed_r['top5']:>10.1%}"
    )
    print("-" * 60)
    print(f"top-1 lift from structure-aware chunking: {struct['top1'] - fixed_r['top1']:+.1%}\n")
    print("Caveats, in words:")
    print(f"  {CAVEATS}\n")

    (DATA_DIR / "retrieval_benchmark.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _rmtree_quietly(SCRATCH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
