# HDFC Mutual Fund - facts-only RAG FAQ assistant

A local, citation-first question-answering app over **five public HDFC Mutual Fund scheme pages**.
It answers published facts (expense ratio, exit load, lock-in, minimum SIP, NAV, AUM, benchmark,
riskometer, tax implication, fund manager, holdings, launch date) and refuses everything else.

> **Facts-only. No investment advice. This assistant reproduces published facts from public fund
> pages and does not recommend, rate, or compare funds for purchase.**

## What it is (and is not)

| | |
|---|---|
| **Corpus** | 5 public scheme pages, HDFC Mutual Fund, all Direct Growth |
| **Answers** | Extractive - sentences are copied verbatim from the retrieved chunk |
| **LLM** | **None.** No API key, no external model, no network call at answer time |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (384-dim), run locally |
| **Vector store** | ChromaDB, persistent on disk |
| **Retrieval** | Hybrid: dense (Chroma) + BM25, fused with Reciprocal Rank Fusion |
| **UI** | Streamlit |
| **Refuses** | Buy/sell/hold advice, return calculations, projections, fund comparison, ranking |
| **Never accepts** | PAN, Aadhaar, account/folio/Demat numbers, OTP, IFSC, card, email, phone |

## Quick start

```powershell
# 1. Create the environment (Python 3.12)
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. Build the index (fetches the 5 pages, chunks, embeds, writes data/chroma)
.\.venv\Scripts\python.exe scripts\build_index.py

# 3. Launch the app
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Then open **http://localhost:8501**.

First run downloads the MiniLM model (~90 MB) into the HuggingFace cache and takes a couple of
minutes. Later runs load from disk. Subsequent `build_index.py` runs reuse the cache unless you
pass `--refresh`.

Step 2 is optional on a fresh clone: the index is committed (see [Deploying](#deploying)), so the
app answers as soon as it starts. Rebuild it only to pick up newer page values.

### Refresh the data

```powershell
.\.venv\Scripts\python.exe scripts\build_index.py --refresh
```

`--refresh` re-fetches all five pages. Without it, previously saved clean text is reused, so you
can rebuild embeddings offline.

### Test suite

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -t .
```

133 tests: ingestion completeness, PII guard, intent routing, field-answer correctness, answer
grounding, sentence limits, one-citation-per-answer, identifier leakage, source-provenance and
index idempotency, evaluation-harness honesty, deliverable traceability, conversation memory,
the guard that keeps the LLM out of the answer path, the UI's clickable example questions, and
the two that stop a deploy shipping without an index.

### Command line

```powershell
.\.venv\Scripts\python.exe scripts\ask.py
```

## Deploying

The app is a single `streamlit run app.py` process with no secrets and no external services. Point
Streamlit Community Cloud (or any host) at the repo, set nothing, and deploy. There is no build
step to configure and no environment variable to set.

**One non-obvious requirement: the index has to be in the repo.** A deployed copy is built from
git and has no shell, so it cannot run `scripts/build_index.py` itself. When `data/chunks.jsonl`
and `data/chroma/` were gitignored, every deploy opened on:

> No index found. Build it first: python scripts/build_index.py

which is advice a hosted app cannot act on. Both paths are now committed - 133 KB of chunks plus
3.0 MB of ChromaDB store, 3.2 MB total for 168 chunks - so a cold start answers immediately. If
you rebuild locally, commit the result too:

```powershell
.\.venv\Scripts\python.exe scripts\build_index.py --refresh
git add data/chunks.jsonl data/chroma
git commit -m "Refresh index"
```

Two layers protect the deploy, and the second is not optional:

1. **Committed index** - the fast path. No build, no fetch, no embed on startup.
2. **Self-heal in `app.py`** - `ensure_index()` builds the index if it is missing, so a fresh
   clone, a wiped container or a store that cannot be read still ends up working. It needs
   outbound network to `groww.in` and takes 1-3 minutes, and it shows progress in the UI. It runs
   at most once per container, because the result is cached with the rest of the assistant.

`requirements.txt` pins `chromadb>=1.5.9` with no upper bound, so a deploy can end up with a
version that cannot read the committed `data/chroma/`. That case is handled too:
`index_exists()` treats an unreadable store as missing (rather than raising past the self-heal),
and `build_index()` retries once after `reset_store()`. The store is derived from the source
pages, so an unreadable one is never worth keeping.

To test either layer, run the app against a checkout with the index deleted - the self-heal path
is covered by `tests/test_faq.py::TestUiSource::test_app_handles_missing_index` and
`test_index_is_committed_so_deploys_are_not_broken`.

## RAG pipeline

```
5 public pages
   -> 1. LOADING      requests + BeautifulSoup/lxml
                      content window isolated; footer/nav boilerplate dropped
   -> 2. CHUNKING     17 typed facts per page as atomic chunks
                      + remaining prose split recursively (180 tok target, 40 overlap, 250 hard max)
   -> 3. EMBEDDING    all-MiniLM-L6-v2, 384-dim, chunk text with its field label
   -> 4. VECTOR DATA  ChromaDB, collection "hdfc_mutual_funds", persisted to data/chroma
   -> 5. RETRIEVAL    dense top-8 + BM25 top-8 -> RRF(k=60) -> top-4
                      + scheme boost 1.35, + section filters
   -> 6. GENERATION   extractive sentence selection, intent routing, guardrails, one citation
```

**Stage 1 - loading.** Each page is parsed twice and the two are compared. The visible rendered
text is treated as truth; the embedded `__NEXT_DATA__` JSON is only used as a conflict detector.
This matters: the payload was stale for the riskometer, the fund manager and the launch date on
more than one page. Conflicts are recorded in `data/ingest_log.json`.

**Stage 2 - chunking.** Chunking is driven by the page's own structure rather than a single fixed
window. Each labelled fact (`Expense ratio: 1.03%`) becomes its own chunk, so a field can be
retrieved without dragging in unrelated numbers. The remaining prose (objective, manager bios,
holdings tables) is split recursively with a 40-token overlap. Chunk sizes are measured with the
real MiniLM tokenizer, not a word count.

Result: **168 chunks** (111 fact + 57 prose), mean 52 tokens, **0 chunks over the 250 hard max**.

**Stage 5 - retrieval.** Hybrid because exact tokens matter (`RTA`, `IFSC`, `NIFTY 50 Hybrid
Composite Debt 50:50`) but paraphrases also matter. RRF merges the two ranked lists without needing
score normalisation. If a question names a scheme, that scheme's chunks get a 1.35 boost; a detected
intent can hard-filter to a section (e.g. "holdings" -> the Holdings section only).

**Stage 6 - generation.** There is no language model. The assistant classifies intent, re-queries
with a field-specific boost, and quotes the matched sentences verbatim (hard cap of 3). This makes
grounding structural rather than a hope: a test asserts every fact answer appears character-for-character
inside the corpus, so a wrong answer is a test failure, not a subtle hallucination.

## Why this chunking - measured, not assumed

Two strategies were built and compared with the same embedding model, the same 35 labelled queries
and the same dense retriever (`scripts/benchmark_retrieval.py`):

| Strategy | Chunks | Top-1 hit | Top-3 hit | Top-5 hit |
|---|---:|---:|---:|---:|
| **Structure-aware** (atomic facts + prose) | 168 | **74.29%** | **94.29%** | 94.29% |
| Fixed 180-word windows (40-word overlap) | 26 | 2.86% | 14.29% | 74.29% |

Worst case for the structure-aware approach: a question asked *without* naming the scheme. Since
the source is a factsheet page, chunks that keep a field's label together with its value are
decisively more retrievable than large windows holding six unrelated fields.

**Read the top-1 column, not the top-5 column.** A fixed window *does* eventually contain the right
fact - it just ranks it below many neighbours, so fixed top-5 recall (74.29%) lands exactly on
structure-aware top-1 accuracy (74.29%). The argument for structure-aware chunking is therefore
about **precision at rank 1**, which is what the assistant actually consumes, not about recall.

Two honest caveats:

- The first comparison metric I used, `cooccur_ratio` ("label and value land in the same chunk"),
  came out **1.0 for every strategy including fixed windows**, so it proved nothing and does not
  discriminate between them. `data/chunking_report.json` is kept only as a diagnostic, and labels
  itself that way. The retrieval benchmark above is the real evidence.
- The benchmark is **dense-only**: it embeds the query and takes nearest neighbours, with no BM25
  leg, no RRF fusion and no scheme or field boost. The shipped pipeline is hybrid, so this number is
  **not a measurement** of the live system. The hybrid retriever *is* scored separately, on phrasing
  robustness, in the next section. The sample is also small: 35 labelled queries, so one query is
  worth ~2.9 points.

The fixed-window hit rule is deliberately lenient (a hit means the gold fact's text appears
*somewhere* in the returned window, since fixed windows carry no field labels), which flatters the
baseline. The 71-point top-1 gap is therefore a lower bound.

`data/retrieval_benchmark.json` holds the per-query results, and
`tests/test_faq.py::TestDeliverables` fails if the table above stops matching it.

## Retrieval robustness to phrasing - and where the system is weak

All 35 benchmark queries are built from one template, `"{FIELD_QUESTION[field]} of {name}?"`. So the
74.29% above says nothing about a user who words the same question differently.
`scripts/benchmark_paraphrase.py` re-runs the same 35 gold labels through the **shipped hybrid
retriever** - dense + BM25, RRF, scheme and field boosts - using five phrasings per query.

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_paraphrase.py            # table
.\.venv\Scripts\python.exe scripts\benchmark_paraphrase.py --misses   # the failures
```

| style | top-1 | top-3 | top-5 | vs template |
|---|---|---|---|---|
| template | 0.9429 | 1.0000 | 1.0000 | |
| jargon | 0.8857 | 0.9429 | 0.9714 | -0.0572 |
| keywords | 0.8571 | 0.9714 | 0.9714 | -0.0858 |
| terse | 0.7429 | 0.8286 | 0.8857 | -0.2000 |
| verbose | 0.2286 | 0.5429 | 0.6286 | -0.7143 |
| colloquial | 0.1714 | 0.5143 | 0.6286 | -0.7715 |

**Read the rows in two groups.** `template`, `jargon`, `keywords` and `terse` all contain the
field's own vocabulary, so they are a fair "same question, different words" comparison. `colloquial`
and `verbose` deliberately *omit* the field words - "how much do they charge in fees" for
`expense_ratio` - so they additionally require implicit intent mapping that no lexical or dense
retriever can do from surface tokens. A human reader has to know that "penalty for redeeming early"
means exit load. Quote those two rows as a worst case, never as an accuracy figure.

The useful signal is the miss breakdown, and it is not the alarming headline number:

| style | wrong scheme | wrong field | prose ranked first |
|---|---|---|---|
| template | 0 | 2 | 1 |
| terse | 0 | 9 | 4 |
| jargon | 0 | 4 | 4 |
| keywords | 0 | 5 | 0 |
| verbose | 3 | 24 | 6 |
| colloquial | 0 | 29 | 3 |

**Scheme resolution is essentially solved** - 0 wrong schemes in five of six styles, and the 3 in
`verbose` are the known alias substring collision, where "a small amount every month in HDFC
Large Cap" resolves to *Small Cap*. The actual weakness is **field disambiguation**: choosing
`expense_ratio` over `exit_load` when both are redemption-adjacent concepts. Field labels are the
only thing distinguishing those chunks, so when the query stops using the field's words, both the
BM25 and the dense leg lose their handle. That is the honest limit of the current design, and it is
invisible to the templated benchmark.

## Guardrails

Checks run in this order, before retrieval:

1. **PII** - PAN, Aadhaar, account/folio/Demat, OTP, IFSC, card, email, phone, passport. Refused
   before any retrieval; nothing is stored. The response links the privacy policy and is the only
   kind with no timestamp, because no source page was consulted.
2. **Advice** - "should I buy", "which is best", "how much should I invest", "is now a good time",
   "help me choose", portfolio questions. Refused with a pointer to a SEBI-registered adviser.
3. **Performance** - returns, CAGR, alpha, Sharpe, "top performer", ranking. Refused, points to the
   page and its official factsheet.
4. **Projection** - "calculate", "project", "estimate", "what if". Refused, points to the scheme page.

After retrieval, a lexical-overlap gate decides whether the question is even about these five
schemes; a weather or joke question gets a scope message instead of a confident guess.

Every answer carries **exactly one source link** and, when a fund page was consulted, a
`Last updated from sources: <timestamp>` stamp. Answers are capped at 3 sentences.

The stamp is when that page was **actually fetched** — the mtime of its cached copy in
`data/raw/`, not the time the app started or the index was rebuilt. Re-running `build_index.py`
without `--refresh` reuses the cached HTML and therefore leaves the stamp unchanged, so the
timestamp stays an honest measure of source freshness instead of drifting forward every restart.

## Conversation memory

The assistant answers one question at a time, so "what about its lock-in?" names no fund and would
otherwise be refused. It now looks back over the **last 10 question/answer turns** and resolves the
fund being referred to:

```
> What is the expense ratio of HDFC ELSS Tax Saver Fund?
HDFC ELSS Tax Saver Fund Direct Plan Growth - Expense ratio: 1.21%.

> What about its lock-in?
HDFC ELSS Tax Saver Fund Direct Plan Growth - Lock-in period: 3Y (as stated on the scheme page).
```

Memory is used for exactly one thing: deciding **which scheme's chunks retrieval is allowed to
consider**. It never contributes text to an answer, never rewrites the question, and never reaches a
prompt - the answer is still a verbatim span from a retrieved chunk, one citation, 3 sentences max,
and no LLM is involved. Turn on the retrieval trace to see when it fired; it reports which turn the
scheme came from.

The limits are deliberate, and each one exists because the alternative is a confident wrong answer:

- **Only when the question names no fund itself.** "Exit load of the HDFC Flexi Cap Fund?" ignores
  memory entirely.
- **Only for a clear follow-up.** A pronoun ("its", "that one"), a short question ("lock-in?"), or
  "what about ...". A fresh, complete question is answered on its own terms.
- **Never for a comparison.** "Which of them has the lowest exit load?" ranges over all five funds, so
  memory refuses to narrow it to one. Without this, a comparative would collapse onto whichever fund
  was discussed last.
- **It cannot widen the scope gate.** "What is the weather in Mumbai?" is still refused after a long
  ELSS discussion. The proper-noun and lexical checks run on your actual words.
- **No identifiers are ever remembered.** Every remembered turn is passed through the same PII scrub
  as your live question before it is examined, so a PAN or folio number typed two turns ago cannot
  re-enter a retrieval query. A turn that was nothing but identifiers is skipped. PII in the *current*
  question is still refused outright.
- **It is a 10-turn window, not 10 documents.** It resolves one scheme; it does not retrieve extra
  chunks, and it does not make the assistant stateful beyond that.

The CLI interactive session and the Streamlit chat both keep this history. One-shot
`ask.py "question"` has no earlier turns, so it behaves exactly as before.

## Sources

See `deliverables/SOURCES.md` and `deliverables/SOURCES.csv`. All five are public pages needing no
login:

| Scheme | Category |
|---|---|
| HDFC Large Cap Fund Direct Growth | Equity - Large Cap |
| HDFC Equity Fund Direct Growth (Flexi Cap) | Equity - Flexi Cap |
| HDFC ELSS Tax Saver Fund Direct Plan Growth | Equity - ELSS |
| HDFC Small Cap Fund Direct Growth | Equity - Small Cap |
| HDFC Balanced Advantage Fund Direct Growth | Hybrid - Dynamic Asset Allocation |

Two naming/scoping quirks handled explicitly:

- The Flexi Cap page is titled *HDFC Flexi Cap Direct Plan Growth*, but the registered scheme name
  is *HDFC Equity Fund Direct Growth*. Both are recognised; answers use the registered name.
- The pages carry **two** AUM figures: scheme-level `Fund size (AUM)` and an AMC-wide
  `Total AUM: 9,86,236.84 Cr` that is identical on all five. The `aum` intent answers scheme size
  only, and a test asserts `Total AUM` never appears in an AUM answer.

Guidance-type answers (how to get a statement) cite the Groww help centre; risk questions can cite
SEBI. Fund facts never cite these.

## Project layout

```
app.py                      Streamlit UI
mf_rag/
  config.py                 paths, model, chunk and retrieval constants
  sources.py                the 5-source registry, disclaimer, educational links
  ingest.py                 fetching, visible-page parsing, typed fact extraction
  chunking.py               atomic fact + recursive prose chunking, tokenizer-based limits
  embed_store.py            MiniLM embeddings + persistent ChromaDB
  retrieve.py               BM25 + dense + RRF + scheme boost + section filters
  answer.py                 intents, extractive answers, citations, refusals
  pii.py                    identifier detection and scrubbing
  pipeline.py               cached assistant assembly
scripts/
  build_index.py            full ingest -> chunk -> embed -> store
  ask.py                    command line interface
  make_sample_qa.py         regenerates deliverables/SAMPLE_QA.{md,json}
  make_sources.py           regenerates deliverables/SOURCES.{md,csv}
  compare_chunking.py       chunk-size comparison
  benchmark_retrieval.py    35-query dense-only chunking benchmark
  benchmark_paraphrase.py   phrasing robustness, scored on the shipped hybrid retriever
  gen_paraphrases.py        regenerates the paraphrase cache (--seed needs no key)
  test_ui.py                Streamlit AppTest smoke test
tests/test_faq.py           133-test suite
data/
  raw/ clean/               fetched HTML and extracted text (not committed)
  chunks.jsonl              168 chunks (committed - a deploy has no shell to build it)
  chroma/                   persistent vector store (committed, same reason)
  documents.json            structured documents
  ingest_log.json           fetch + conflict log
  chunking_report.json      chunking diagnostics
  paraphrase_queries.json   cached query rephrasings (committed, so eval needs no key)
  paraphrase_benchmark.json phrasing-robustness results
  retrieval_benchmark.json  benchmark results
deliverables/
  SAMPLE_QA.md / .json      10 sample Q&A from the real assistant
  SOURCES.md / .csv         source list
  DISCLAIMER.md             disclaimer text used
```

## Known limitations

- **Source scope.** The corpus is the five user-specified Groww pages. No HDFC AMC, SEBI or AMFI
  document (KIM, SID, factsheet) is included, so a field only those documents carry - portfolio
  turnover, scheme performance sheet, distributor details - cannot be answered.
- **Extractive, not abstractive.** Answers read like a fact sheet, not like a chat reply. This is
  deliberate: it is the only way to guarantee nothing is invented without a language model.
- **Ranking is meaningless.** The performance refusal means a user cannot ask which fund did best,
  even though the pages publish returns. That is a policy decision, not a capability gap.
- **Snapshot only.** Values are as at the per-page timestamp. NAV and AUM move daily; expense ratio
  changes periodically. Re-run with `--refresh`.
- **Five schemes.** Questions about other HDFC funds correctly return a scope message.
- **English, keyword-tuned routing.** The intent patterns are hand-written and tested on English
  phrasing. Unusual paraphrases fall back to hybrid retrieval, which is weaker than a matched
  intent but still grounded.
- **Scoring.** The retrieval benchmark is dense-only and small (35 queries, written by the author of
  the system). It is evidence of a real difference, not a statistically strong result.
