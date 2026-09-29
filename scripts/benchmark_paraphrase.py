"""Measure retrieval robustness to phrasing, using the SHIPPED hybrid retriever.

`scripts/benchmark_retrieval.py` is dense-only: no BM25, no RRF, no boosts. Its own
caveat text says live accuracy is "expected to be at least this good" but that this file
does not measure it. This script closes that gap for the wording axis.

It runs the real `HybridRetriever` - dense + BM25, RRF fusion, scheme and field boosts -
over the templated benchmark queries and over each LLM-generated phrasing style, and
reports top-1/3/5 for every style against the same gold labels. A style that scores below
the template is a phrasing the retriever fails on.

Requires no API key: paraphrases are read from the committed cache. Regenerate the cache
with `scripts/gen_paraphrases.py`.

Usage:
    python scripts/benchmark_paraphrase.py             # report
    python scripts/benchmark_paraphrase.py --misses    # list the failures
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mf_rag.config import DATA_DIR
from mf_rag.retrieve import get_retriever

CACHE = DATA_DIR / "paraphrase_queries.json"
REPORT = DATA_DIR / "paraphrase_benchmark.json"
TOP_K = 8

CAVEATS = (
    "This measures the SHIPPED hybrid retriever (dense + BM25, RRF k=60, scheme boost, field boost), "
    "unlike scripts/benchmark_retrieval.py which is dense-only. Gold labels are the same 35 "
    "(scheme, field) pairs, so differences between rows are attributable to WORDING alone. "
    "READ THE ROWS IN TWO GROUPS, because they measure different things. "
    "(1) template, terse, jargon, keywords: these all contain the field's own vocabulary, so they "
    "are a fair 'same question, different words' comparison - this is where phrasing robustness shows. "
    "(2) colloquial, verbose: these describe the concept WITHOUT naming the field ('how much do they "
    "charge in fees' for expense_ratio), so they additionally demand implicit intent mapping that no "
    "retrieval system can do from surface tokens alone. Their low scores are a deliberate worst case, "
    "not an estimate of real-world failure - a human reader also has to know that 'penalty for "
    "redeeming early' means exit load. Quote the colloquial/verbose numbers as a bound, never as an "
    "accuracy figure. Separately, the paraphrases are synthetic and were produced once and cached, so "
    "this is one sample of generator output, not a distribution, and it cannot show a failure mode the "
    "author did not think to produce."
)


def _hit(results, gold_scheme: str, gold_field: str, depth: int) -> bool:
    """A hit is the gold field chunk for the gold scheme appearing within `depth`.

    Same rule the dense-only benchmark uses, applied to the hybrid retriever's output.
    """
    for chunk in results[:depth]:
        if chunk.scheme_short == gold_scheme and chunk.field_name == gold_field:
            return True
    return False


def evaluate(retriever, query: str, gold_scheme: str, gold_field: str) -> dict:
    results = retriever.search(query, top_k=TOP_K)
    return {
        "query": query,
        "gold_scheme": gold_scheme,
        "gold_field": gold_field,
        "top1": _hit(results, gold_scheme, gold_field, 1),
        "top3": _hit(results, gold_scheme, gold_field, 3),
        "top5": _hit(results, gold_scheme, gold_field, 5),
        "top1_scheme": results[0].scheme_short if results else None,
        "top1_field": results[0].field_name if results else None,
    }


def main() -> int:
    if not CACHE.exists():
        print(f"missing {CACHE}\nrun: python scripts/gen_paraphrases.py")
        return 1

    data = json.loads(CACHE.read_text(encoding="utf-8"))
    queries = data["queries"]
    retriever = get_retriever()

    # The templated phrasing is the baseline every style is compared against.
    rows: dict[str, list[dict]] = {"template": []}
    for key, entry in queries.items():
        rows["template"].append(
            evaluate(retriever, entry["template_query"], entry["scheme"], entry["field"])
        )
    for style in data["styles"]:
        rows[style] = []
        for key, entry in queries.items():
            for text in entry["paraphrases"][style]:
                rows[style].append(evaluate(retriever, text, entry["scheme"], entry["field"]))

    def rate(row: list[dict], depth: str) -> float:
        return round(sum(r[depth] for r in row) / len(row), 4)

    summary = {
        style: {
            "n": len(row),
            "top1": rate(row, "top1"),
            "top3": rate(row, "top3"),
            "top5": rate(row, "top5"),
        }
        for style, row in rows.items()
    }
    baseline = summary["template"]["top1"]

    # Two separable failure modes, counted so the numbers above are interpretable rather
    # than just alarming: resolving the wrong scheme, versus picking the right scheme but
    # the wrong field within it.
    def _split(row: list[dict]) -> dict[str, int]:
        wrong_scheme = sum(1 for r in row if not r["top1"] and r["top1_scheme"] != r["gold_scheme"])
        wrong_field = sum(
            1 for r in row if not r["top1"] and r["top1_scheme"] == r["gold_scheme"] and r["top1_field"] != r["gold_field"]
        )
        prose = sum(1 for r in row if not r["top1"] and r["top1_field"] in (None, ""))
        return {
            "miss_wrong_scheme": wrong_scheme,
            "miss_wrong_field": wrong_field,
            "miss_prose_not_fact": prose,
        }

    report = {
        "caveats": CAVEATS,
        "measurement": {
            "retrieval": f"shipped HybridRetriever, top-{TOP_K} candidates per query",
            "hit_rule": "chunk's scheme_short == gold scheme AND field_name == gold field",
            "generated_by": data.get("generated_by", "unknown"),
            "how_to_read": (
                "template/terse/jargon/keywords carry the field's own vocabulary and measure phrasing "
                "robustness. colloquial/verbose do not name the field and measure implicit intent mapping "
                "as well, which is a deliberately unfair comparison for a lexical retriever."
            ),
        },
        "summary": summary,
        "miss_breakdown": {style: _split(row) for style, row in rows.items()},
        "per_style": rows,
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Retrieval robustness to phrasing - shipped hybrid retriever, top-{TOP_K} candidates\n")
    print(f"{'style':<12}{'n':>5}{'top-1':>9}{'top-3':>9}{'top-5':>9}{'vs template':>13}")
    print("-" * 57)
    for style, stats in summary.items():
        delta = "" if style == "template" else f"{stats['top1'] - baseline:+.4f}"
        print(f"{style:<12}{stats['n']:>5}{stats['top1']:>9.4f}{stats['top3']:>9.4f}{stats['top5']:>9.4f}{delta:>13}")
    print("\nwhere top-1 misses come from:")
    for style, counts in report["miss_breakdown"].items():
        if sum(counts.values()):
            print(
                f"  {style:<12} wrong scheme {counts['miss_wrong_scheme']:>2}"
                f"   wrong field {counts['miss_wrong_field']:>2}"
                f"   prose not fact {counts['miss_prose_not_fact']:>2}"
            )
    print(f"\nwrote {REPORT}")

    if "--misses" in sys.argv:
        print("\nfailures at top-1 (gold field chunk not ranked first):")
        for style, row in rows.items():
            misses = [r for r in row if not r["top1"]]
            if not misses:
                continue
            print(f"\n  {style} - {len(misses)}/{len(row)}")
            for miss in misses[:12]:
                print(f"    {miss['query'][:70]!r}")
                print(f"      got: {miss['top1_scheme']}/{miss['top1_field']}")
            if len(misses) > 12:
                print(f"    ... {len(misses) - 12} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
