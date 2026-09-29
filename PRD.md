# PRD — Mutual Fund Facts-Only FAQ Assistant (HDFC)

**Type:** Class milestone brief → working prototype
**Status:** Implemented (v1) — 29/29 tests passing
**Source of truth:** `Docs/problemstatement.txt` (milestone brief + key constraints + deliverables)
**One-line:** A local, citation-first RAG chatbot that answers published facts about 5 HDFC
Mutual Fund schemes and refuses everything else.

---

## 1. Problem

Retail users comparing HDFC Mutual Fund schemes have to hunt across scheme pages for basic
published facts — expense ratio, exit load, minimum SIP, ELSS lock-in, riskometer, benchmark,
NAV, fund size — and support/content teams answer the same questions repeatedly. Generic
chatbots fill the gap by *guessing*, and in finance a plausible-sounding wrong number is worse
than no answer.

The gap this project closes: **a question-answering assistant that cannot hallucinate**, because
it has no language model. Every sentence it returns is copied verbatim from a public source page
and shipped with the link to that page.

## 2. Who this helps

| User | Need |
|---|---|
| Retail investor comparing schemes | "What's the exit load on this ELSS fund?" — one link, one number, no opinion |
| Support / content team | Repetitive MF questions answered consistently, without re-reading five pages |

## 3. Goals

| # | Goal | How success is measured |
|---|---|---|
| G1 | Answer the 8 named fact types correctly, per scheme | 29-test suite green; per-scheme non-cross-contamination test |
| G2 | Every answer carries exactly one source link | `test_fact_answers_cite_exactly_one_source`, `test_every_answer_path_returns_a_citation` |
| G3 | Nothing is invented | `test_answers_are_grounded_in_retrieved_chunks` — every fact answer must appear character-for-character in the corpus |
| G4 | Out-of-scope and unsafe questions are refused, not guessed | advice / performance / projection / PII / out-of-scope tests |
| G5 | Answers are scannable | Hard cap of 3 sentences, plus a `Last updated from sources:` stamp |
| G6 | The RAG architecture is demonstrable end to end | 6-stage pipeline visible in the UI sidebar + optional retrieval trace |

## 4. Non-goals

- **No investment advice.** No buy/sell/hold verdicts, no fund ranking, no suitability calls.
- **No performance analysis.** Returns, CAGR, alpha, Sharpe are refused; the user is pointed at
  the official factsheet. *(A policy decision, not a capability gap — the pages do publish returns.)*
- **No LLM.** No API key, no external model, no network call at answer time. Extractive only.
- **No live data.** Values are a snapshot as at each page's timestamp; NAV/AUM move daily.
- **No multi-AMC scope.** Five HDFC schemes only. Questions about other funds return a scope message.

## 5. Scope — corpus

**AMC:** HDFC Mutual Fund. **5 schemes, all Direct Growth plans.** All sources are public
scheme pages: no login, no paywall, no third-party blogs, no brokerage or back-office screenshots.

| # | Scheme | Category | URL |
|---|---|---|---|
| 1 | HDFC Large Cap Fund Direct Growth | Equity — Large Cap | https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth |
| 2 | HDFC Equity Fund Direct Growth | Equity — Flexi Cap | https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth |
| 3 | HDFC ELSS Tax Saver Fund Direct Plan Growth | Equity — ELSS | https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth |
| 4 | HDFC Small Cap Fund Direct Growth | Equity — Small Cap | https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth |
| 5 | HDFC Balanced Advantage Fund Direct Growth | Hybrid — Balanced Advantage | https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth |

**Secondary / educational links** — cited *only* for guidance-type answers, never for a fund fact:
Groww help centre (statements, tax docs), Groww mutual funds index, SEBI (riskometer context).

### 5.1 Naming and scoping quirks handled explicitly

- The Flexi Cap page is titled *HDFC Flexi Cap Direct Plan Growth*; the registered scheme name is
  *HDFC Equity Fund Direct Growth*. Both are recognised as aliases; answers use the registered name.
- The pages carry **two** AUM figures: scheme-level `Fund size (AUM)` and an AMC-wide
  `Total AUM: 9,86,236.84 Cr` that is identical on all five. The `aum` intent answers scheme size
  only (`test_aum_reports_scheme_size_not_amc_total`).

## 6. Functional requirements

### FR-1 — Corpus ingestion
- FR-1.1 Fetch all 5 public pages and persist raw HTML to `data/raw/`.
- FR-1.2 Extract clean visible text to `data/clean/`, isolated to the page's **content window** —
  nav and footer boilerplate dropped (`test_content_window_excludes_nav_and_footer_boilerplate`).
- FR-1.3 Parse each page **twice** (rendered visible text + embedded `__NEXT_DATA__` JSON) and
  compare. Visible text wins; the payload is used only as a conflict detector. The payload was
  stale for riskometer, fund manager and launch date on more than one page. Conflicts are recorded
  in `data/ingest_log.json` (`test_conflicts_prefer_visible_page`).
- FR-1.4 Extract **22 typed labelled facts per page** (23 distinct field names across the corpus;
  111 fact chunks total): expense ratio, exit load, lock-in, min SIP, min lump sum, min additional,
  benchmark, riskometer, rating, NAV, scheme AUM, AMC-wide total AUM, rank, tax implication, stamp
  duty, objective, launch date, incorporation date, custodian, RTA, fund house, category,
  sub-category — plus fund-manager and holdings as labelled sections.
- FR-1.5 Re-running without `--refresh` reuses saved clean text, so embeddings can be rebuilt offline.

### FR-2 — Chunking (strategy decided from the data, not assumed)
- FR-2.1 Each labelled fact (`Expense ratio: 1.03%`) becomes its **own atomic chunk**, so a field
  can be retrieved without dragging in unrelated numbers.
- FR-2.2 Remaining prose (objective, manager bios, holdings tables) is split **recursively** with
  overlap.
- FR-2.3 Chunk sizes are measured with the **real MiniLM tokenizer**, not a word count.
- FR-2.4 Constants: 180-token target, 40-token overlap, 250-token hard max (`mf_rag/config.py`).
- FR-2.5 **Acceptance:** 0 chunks exceed the 250-token hard max.

### FR-3 — Embedding
- FR-3.1 Model: `sentence-transformers/all-MiniLM-L6-v2` from Hugging Face, 384-dim, run locally.
- FR-3.2 Chunk text is embedded together with its field label.
- FR-3.3 First run downloads ~90 MB into the HF cache; later runs load from disk. No API key.

### FR-4 — Vector store
- FR-4.1 ChromaDB, persistent on disk at `data/chroma`.
- FR-4.2 Collection name `hdfc_mutual_funds` (`CHROMA_COLLECTION`).

### FR-5 — Retrieval (hybrid)
- FR-5.1 Dense top-8 (Chroma) **and** BM25 top-8, fused with **Reciprocal Rank Fusion**, `k=60`.
- FR-5.2 Rationale: exact tokens matter (`RTA`, `IFSC`, `NIFTY 50 Hybrid Composite Debt 50:50`)
  *and* paraphrases matter. RRF merges two ranked lists without score normalisation.
- FR-5.3 If the question names a scheme, that scheme's chunks get a **1.35 boost**.
- FR-5.4 A detected intent can **hard-filter to a section** (e.g. `holdings` → Holdings only).
- FR-5.5 Final context: top-4 chunks, max 6 fed to the answer stage.

### FR-6 — Answer generation (extractive, no LLM)
- FR-6.1 Classify intent, re-query with a field-specific boost, quote matched sentences **verbatim**.
- FR-6.2 Hard cap of **3 sentences** per answer.
- FR-6.3 **Exactly one** source link per answer.
- FR-6.4 `Last updated from sources: <timestamp>` whenever a fund page was consulted.
- FR-6.5 Intents cover: expense ratio, exit load, lock-in, min SIP, min lump sum, min additional,
  benchmark, riskometer, rating, NAV, AUM, tax implication, stamp duty, objective, launch date,
  incorporation date, custodian, RTA, fund house, category, sub-category, fund manager, holdings,
  exit-load history, statement guidance, plus refusal intents.

### FR-7 — Guardrails (run in this order, before retrieval)
| Order | Guard | Behaviour | Educational link |
|---|---|---|---|
| 1 | **PII** | PAN, Aadhaar, account/folio/Demat, OTP, IFSC, card, email, phone, passport → refused before any retrieval; nothing stored. Only answer kind with no timestamp, because no page was consulted | Privacy policy |
| 2 | **Advice** | "should I buy", "which is best", "how much should I invest", "is now a good time", "help me choose", portfolio questions → refused | SEBI-registered adviser pointer + Groww MF index |
| 3 | **Performance** | returns, CAGR, alpha, Sharpe, "top performer", ranking → refused, points to page + official factsheet | SEBI / factsheet |
| 4 | **Projection** | "calculate", "project", "estimate", "what if" → refused, points to scheme page | Scheme page |

After retrieval, a **lexical-overlap gate** decides whether the question is even about these five
schemes — a weather or joke question gets a scope message instead of a confident guess
(`test_out_of_scope_query`). Answers must never leak identifiers back
(`test_answers_never_leak_identifiers`).

### FR-8 — UI (tiny, as briefed)
- FR-8.1 Welcome line naming the 5 schemes and the fact types covered.
- FR-8.2 Exactly **3 example questions**, clickable/suggested.
- FR-8.3 The note **"Facts-only. No investment advice."** visible at the top and in the sidebar.
- FR-8.4 Sidebar: scope (AMC + 5 schemes), the pipeline diagram, embedding model, top-k, guardrail list.
- FR-8.5 Per-answer rendering: body, one **Source** link button, `Last updated from sources:` caption.
- FR-8.6 Optional **retrieval trace** expander showing intent, refusal flag, and each chunk's
  RRF / dense / BM25 scores — for the demo.
- FR-8.7 Graceful errors if the index is missing or unbuilt.

## 7. Non-functional requirements

| Area | Requirement |
|---|---|
| **Privacy** | No PII accepted, processed or stored. Runs fully locally. No telemetry. |
| **Offline at answer time** | No network call once the index is built. |
| **Determinism** | Same question → same answer. No sampling, no temperature. |
| **Portability** | Python 3.12, `requirements.txt`, three commands to first answer. |
| **Traceability** | Every answer links to the exact page it came from. |
| **Clarity** | ≤3 sentences; jargon as published on the page. |

## 8. Architecture

```
5 public pages
   -> 1. LOADING     requests + BeautifulSoup/lxml
                     content window isolated; footer/nav dropped; dual-parse conflict check
   -> 2. CHUNKING    22 typed facts per page as atomic chunks
                     + prose split recursively (180 tok target, 40 overlap, 250 hard max)
   -> 3. EMBEDDING   all-MiniLM-L6-v2, 384-dim, chunk text + field label
   -> 4. VECTOR DATA ChromaDB, collection "hdfc_mutual_funds", persisted to data/chroma
   -> 5. RETRIEVAL   dense top-8 + BM25 top-8 -> RRF(k=60) -> top-4
                     + scheme boost 1.35, + section filters
   -> 6. GENERATION  extractive sentence selection, intent routing, guardrails, one citation
```

Stage 6 is deliberately *not* a language model. This makes grounding **structural rather than a
hope**: a wrong answer is a test failure, not a subtle hallucination.

### 8.1 Module map

| Module | Responsibility |
|---|---|
| `mf_rag/config.py` | Paths, model, chunk and retrieval constants |
| `mf_rag/sources.py` | 5-source registry, disclaimer, educational links |
| `mf_rag/ingest.py` | Fetching, visible-page parsing, typed fact extraction |
| `mf_rag/chunking.py` | Atomic fact + recursive prose chunking, tokenizer-based limits |
| `mf_rag/embed_store.py` | MiniLM embeddings + persistent ChromaDB |
| `mf_rag/retrieve.py` | BM25 + dense + RRF + scheme boost + section filters |
| `mf_rag/answer.py` | Intents, extractive answers, citations, refusals |
| `mf_rag/pii.py` | Identifier detection and scrubbing |
| `mf_rag/pipeline.py` | Cached assistant assembly, example questions |
| `app.py` | Streamlit UI |

## 9. Evidence — chunking strategy was measured, not assumed

Two strategies built and compared with the **same** embedding model, the **same** 35 labelled
queries and the **same** dense retriever (`scripts/benchmark_retrieval.py`):

| Strategy | Chunks | Top-1 hit | Top-3 hit |
|---|---:|---:|---:|
| **Structure-aware** (atomic facts + prose) | 168 | **74.29%** | **94.29%** |
| Fixed 180-token windows | 24 | 2.86% | 22.86% |

Worst case for structure-aware chunking: a question asked *without* naming the scheme. Since the
source is a factsheet page, chunks that keep a field's label together with its value are decisively
more retrievable than large windows holding six unrelated fields.

**Honest caveats carried into the README:**
- The first comparison metric used ("label and value land in the same chunk") was **1.0 for every
  strategy including fixed windows**, so it proved nothing. The retrieval benchmark is the real
  evidence; `data/chunking_report.json` is kept only as a diagnostic.
- The benchmark is **dense-only**. The shipped pipeline is hybrid, so live accuracy is expected to
  be at least this good, but the hybrid configuration itself has not been scored separately.

## 10. Acceptance criteria

| # | Criterion | Verified by |
|---|---|---|
| AC-1 | All 5 sources load, each with all core fields | `test_five_sources_loaded`, `test_every_source_has_all_core_fields` |
| AC-2 | ELSS lock-in and nil exit load are extractable | `test_elss_has_lock_in_and_nil_exit_load` |
| AC-3 | Boilerplate excluded; stale embedded JSON loses to visible text | `test_content_window_excludes_nav_and_footer_boilerplate`, `test_conflicts_prefer_visible_page` |
| AC-4 | 0 chunks over the 250-token hard max | `data/chunks.jsonl` — 168 chunks (111 fact + 57 prose), mean 52 tokens |
| AC-5 | Correct field answered, not a neighbour | `test_answers_report_the_asked_field_not_a_neighbour` |
| AC-6 | Scheme answers don't cross-contaminate | `test_per_scheme_answers_do_not_cross_contaminate` |
| AC-7 | Exactly one source link on every answer path | `test_fact_answers_cite_exactly_one_source`, `test_every_answer_path_returns_a_citation` |
| AC-8 | ≤3 sentences per answer | `test_answers_are_at_most_three_sentences` |
| AC-9 | Answers grounded character-for-character in the corpus | `test_answers_are_grounded_in_retrieved_chunks` |
| AC-10 | Timestamped whenever sourced; PII refusal is the sole exception | `test_every_answer_carries_timestamp_when_sourced`, `test_render_omits_timestamp_but_keeps_privacy_link_for_pii_refusal` |
| AC-11 | PII refused before retrieval; identifiers never leak back | `test_pii_is_refused_before_retrieval`, `test_answers_never_leak_identifiers` |
| AC-12 | Advice refused with an educational link | `test_advice_is_refused_with_educational_link` |
| AC-13 | Performance and projection questions refused | `test_performance_and_projection_are_refused` |
| AC-14 | Out-of-scope question gets a scope message, not a guess | `test_out_of_scope_query` |
| AC-15 | 3 example questions + facts-only note in the UI | `app.py:109-113`, `mf_rag/pipeline.py:11-15` |
| AC-16 | Streamlit smoke test passes | `scripts/test_ui.py` |
| AC-17 | 5–10 sample Q&A with answers + links | `deliverables/SAMPLE_QA.md` (10 rows) |

**Status: 29/29 tests green** (`python -m unittest discover -s tests -t .`).

## 11. Deliverables

| Deliverable (per brief) | Artefact | Status |
|---|---|---|
| Working prototype link or ≤3-min demo video | `app.py` → `streamlit run app.py` (local) | Done |
| Source list (CSV/MD) of the 5 URLs | `deliverables/SOURCES.md`, `deliverables/SOURCES.csv` | Done |
| README: setup, scope, known limits | `README.md` | Done |
| Sample Q&A (5–10 queries + answers + links) | `deliverables/SAMPLE_QA.md`, `.json` | Done (10) |
| Disclaimer snippet used in the UI | `deliverables/DISCLAIMER.md`, `mf_rag/sources.py` | Done |

> Disclaimer text used in the UI: *"Facts-only. No investment advice. This assistant reproduces
> published facts from public fund pages and does not recommend, rate, or compare funds for purchase."*

## 12. Known limitations

1. **Source scope.** Only the 5 user-specified Groww pages. No HDFC AMC, SEBI or AMFI document
   (KIM, SID, factsheet) is included, so fields only those carry — portfolio turnover, scheme
   performance sheet, distributor details — cannot be answered.
2. **Extractive, not abstractive.** Answers read like a fact sheet, not like a chat reply. This is
   deliberate: it is the only way to guarantee nothing is invented without a language model.
3. **Ranking is meaningless.** The performance refusal means a user cannot ask which fund did best,
   even though the pages publish returns. Policy decision, not a capability gap.
4. **Snapshot only.** Values are as at the per-page timestamp. NAV and AUM move daily; expense ratio
   changes periodically. Re-run with `--refresh`.
5. **Five schemes.** Questions about other HDFC funds correctly return a scope message.
6. **English, keyword-tuned routing.** Intent patterns are hand-written and tested on English phrasing.
   Unusual paraphrases fall back to hybrid retrieval — weaker than a matched intent, still grounded.
7. **Scoring.** The retrieval benchmark is dense-only and small (35 queries, written by the author
   of the system). Evidence of a real difference, not a statistically strong result.

## 13. Future scope (not in v1)

- Widen the corpus to KIM / SID / official factsheets to close the gaps in limitation 1.
- Score the **hybrid** configuration on its own benchmark, not just dense-only.
- Multilingual intent patterns; the guardrails are English-only today.
- Optional LLM *rewriter* for phrasing only — with the extractive sentence as the sole source of
  every number, so grounding stays structural.

## 14. Open questions for the milestone review

1. Should the performance refusal point to the HDFC factsheet PDF specifically, or is a link to the
   scheme page sufficient for the brief?
2. Do we need a hosted demo link, or is the local Streamlit app + a screen recording acceptable
   evidence of a "working prototype"?
3. Is a 5-scheme corpus enough, or does the rubric expect the 3-scheme minimum to be widened?
