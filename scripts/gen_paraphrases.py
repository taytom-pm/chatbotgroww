"""Generate rephrasings of the 35 benchmark queries to test retrieval robustness.

Why this exists: every query in `scripts/benchmark_retrieval.py` is built from one
template, `"{FIELD_QUESTION[field]} of {display_name}?"`. That is 35 queries sharing a
single phrasing style, so the reported 74.29% top-1 says nothing about what happens
when a user words the same question differently. This script produces the missing
wording axis: terse, verbose, colloquial, jargon-heavy and keyword-only variants, all
pointing at the same gold (scheme, field) pair.

The gold labels never move. Only the wording changes, which is what makes the resulting
comparison meaningful: any drop in hit rate is retrieval failing to be robust to
phrasing, not a relabelling artefact.

Output is cached to `data/paraphrase_queries.json` and committed, so
`scripts/benchmark_paraphrase.py` reproduces without a key or network access. Regenerating
requires a provider; reading the cache does not.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mf_rag.config import DATA_DIR
from mf_rag.llm import LlmUnavailable, complete, parse_json_list, resolve_config
from mf_rag.sources import SOURCES
from scripts.benchmark_retrieval import BENCH_FIELDS, FIELD_QUESTION

CACHE = DATA_DIR / "paraphrase_queries.json"

# Each style is a different failure mode we want to be able to detect. Named for the
# hypothesis being tested, not for a linguistic register, so the results stay readable.
STYLES = {
    "terse": "as short as possible, like a search box - drop articles and auxiliary verbs",
    "colloquial": "how a first-time retail investor would actually say it out loud, casually",
    "verbose": "a full rambling sentence with a clause of context before the question",
    "jargon": "using fund-industry jargon and abbreviations, the way an analyst would",
    "keywords": "just the keywords, no grammatical sentence at all, like a query string",
}


def build_pairs() -> list[dict[str, str]]:
    """The 35 gold (scheme, field) pairs, templated exactly as the main benchmark does."""
    return [
        {
            "scheme": source.scheme_short,
            "scheme_name": source.display_name,
            "field": field,
            "template_query": f"{FIELD_QUESTION[field]} of {source.display_name}?",
        }
        for source in SOURCES
        for field in BENCH_FIELDS
    ]


# --- deterministic seed -----------------------------------------------------
# Wordings below are hand-authored, not LLM output, so the harness is usable with no key
# and so there is a baseline the LLM variants can be compared against. They are recorded
# as `hand-authored seed` in the cache; running the LLM path overwrites that marker, which
# is how a reader tells which corpus they are looking at.
SEED_WORDS = {
    "expense_ratio": {
        "terse": "{name} expense ratio",
        "colloquial": "how much do they charge in fees for {name}?",
        "verbose": "I have been looking at {name} and I am trying to work out what the annual charges work out at.",
        "jargon": "{name} TER / expense ratio",
        "keywords": "{name} {short} annual fee %",
    },
    "exit_load": {
        "terse": "{name} exit load",
        "colloquial": "if I sell {name} early, do I lose anything?",
        "verbose": "I am wondering whether there is any penalty if I redeem {name} before it has been running for a while.",
        "jargon": "{name} exit load schedule, 1yr cut-off",
        "keywords": "{name} {short} exit load redemption charge",
    },
    "min_sip": {
        "terse": "{name} minimum SIP",
        "colloquial": "what's the smallest amount I can put in each month for {name}?",
        "verbose": "I would like to start investing a small amount every month in {name} but I need to know the smallest amount that is actually allowed.",
        "jargon": "{name} min SIP / monthly instalment",
        "keywords": "{name} {short} SIP minimum amount",
    },
    "min_lumpsum": {
        "terse": "{name} minimum lump sum",
        "colloquial": "if I put money in all at once instead, what's the least I can do for {name}?",
        "verbose": "Rather than investing monthly, I am considering putting a single amount in, so what is the smallest one-time investment allowed for {name}?",
        "jargon": "{name} min lumpsum threshold",
        "keywords": "{name} {short} one-time investment minimum",
    },
    "benchmark": {
        "terse": "{name} benchmark",
        "colloquial": "what is {name} actually measured against?",
        "verbose": "I keep seeing references to an index that {name} is judged against and I would like to know which one it actually is.",
        "jargon": "{name} benchmark index / TRI",
        "keywords": "{name} {short} benchmark index name",
    },
    "riskometer": {
        "terse": "{name} riskometer",
        "colloquial": "how risky does the company itself rate {name}?",
        "verbose": "The company publishes a risk category for every fund and I would like to know which category {name} currently sits in.",
        "jargon": "{name} riskometer category / SEBI scale",
        "keywords": "{name} {short} risk level category",
    },
    "aum": {
        "terse": "{name} fund size",
        "colloquial": "how big is {name} these days?",
        "verbose": "I am trying to get a sense of how much money is sitting in {name} overall, so that I can judge whether it is a large or small fund.",
        "jargon": "{name} AUM / corpus size",
        "keywords": "{name} {short} total AUM crore",
    },
}


def seed(pairs: list[dict[str, str]]) -> dict[str, dict[str, list[str]]]:
    """Hand-authored paraphrases. No network, no key, fully deterministic.

    Returns the same shape as `generate()` - {style: {key: [text]}} - so `_write` and
    `benchmark_paraphrase.py` cannot tell which path produced the cache.
    """
    by_style: dict[str, dict[str, list[str]]] = {style: {} for style in STYLES}
    for pair in pairs:
        words = SEED_WORDS[pair["field"]]
        key = f"{pair['scheme']}|{pair['field']}"
        for style in STYLES:
            by_style[style][key] = [
                words[style].format(name=pair["scheme_name"], short=pair["scheme"])
            ]
    return by_style


def generate(style: str, hint: str, pairs: list[dict[str, str]], *, per_call: int = 3) -> dict[str, list[str]]:
    """One style over all 35 pairs, batched so a style costs 12 calls, not 35."""
    config = resolve_config()
    out: dict[str, list[str]] = {}
    for start in range(0, len(pairs), per_call):
        batch = pairs[start : start + per_call]
        questions = "\n".join(f"{i + 1}. {p['template_query']}" for i, p in enumerate(batch))
        prompt = (
            f"You are generating test queries for a mutual-fund retrieval system.\n\n"
            f"Rewrite each of the {len(batch)} questions below in the requested style.\n\n"
            f"Rules:\n"
            f"- Keep the SAME fund and the SAME piece of information in every rewrite. "
            f"Change only the wording.\n"
            f"- Keep the fund name recognisable. Do not invent fees, dates or numbers.\n"
            f"- One line each, no quotes, no numbering, no preamble.\n"
            f"- Return exactly {len(batch)} rewrites as a JSON array of strings, in order.\n\n"
            f"Style: {hint}\n\nQuestions:\n{questions}\n"
        )
        rewrites = parse_json_list(complete(prompt, config=config))
        if len(rewrites) != len(batch):
            raise SystemExit(
                f"style {style!r} batch at {start}: expected {len(batch)} rewrites, "
                f"got {len(rewrites)}"
            )
        for pair, text in zip(batch, rewrites):
            out.setdefault(f"{pair['scheme']}|{pair['field']}", []).append(text)
    return out


def main() -> int:
    pairs = build_pairs()

    if "--seed" in sys.argv:
        _write(seed(pairs), pairs, "hand-authored seed (no LLM)")
        return 0

    try:
        config = resolve_config()
    except LlmUnavailable as exc:
        print(f"Cannot generate paraphrases: {exc}")
        print(f"\n{CACHE} is the committed cache; scripts/benchmark_paraphrase.py reads it")
        print("without a key. Use --seed to rebuild it with no key at all.")
        return 1

    print(f"provider={config.provider} model={config.model}")
    print(f"generating {len(STYLES)} styles x {len(pairs)} queries\n")

    by_style = {}
    for style, hint in STYLES.items():
        print(f"  {style} ...", flush=True)
        by_style[style] = generate(style, hint, pairs)

    _write(by_style, pairs, f"{config.provider}/{config.model}")
    return 0


def _write(by_style: dict, pairs: list[dict[str, str]], generated_by: str) -> None:
    report = {
        "generated_by": generated_by,
        "note": (
            "Cache of paraphrased benchmark queries. Gold labels are unchanged; only wording "
            "varies. Regenerate with scripts/gen_paraphrases.py (LLM) or --seed (no key). "
            "Used by scripts/benchmark_paraphrase.py to measure retrieval robustness to phrasing."
        ),
        "styles": {style: hint for style, hint in STYLES.items()},
        "queries": {
            f"{pair['scheme']}|{pair['field']}": {
                "scheme": pair["scheme"],
                "field": pair["field"],
                "template_query": pair["template_query"],
                "paraphrases": {
                    style: by_style[style][f"{pair['scheme']}|{pair['field']}"]
                    for style in STYLES
                },
            }
            for pair in pairs
        },
    }
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    total = sum(len(v) for q in report["queries"].values() for v in q["paraphrases"].values())
    label = "hand-authored seed" if "seed" in generated_by else generated_by
    print(f"wrote {CACHE} - {len(report['queries'])} queries x {len(STYLES)} styles = {total} paraphrases")
    print(f"provenance: {label}")


if __name__ == "__main__":
    raise SystemExit(main())
