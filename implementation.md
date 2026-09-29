# Implementation Plan — HDFC Mutual Fund Facts-Only RAG Assistant

**Inputs:** `PRD.md` (what) · `architecture.md` (how and why) · `Docs/problemstatement.txt` (brief)
**Purpose:** rebuild or extend the system **one phase at a time**, with a verification gate at
every boundary. Each phase contains a paste-ready Cursor prompt.

---

## 0. How to use this document

Work **strictly in phase order.** Each phase is independently verifiable, and later phases assume
earlier invariants hold.

```
for each phase:
  1. paste the phase's Cursor prompt
  2. let Cursor implement ONLY that phase
  3. run the phase's Verification commands
  4. if the Exit gate passes -> commit, move on
  5. if it fails -> fix inside this phase. Do NOT start the next phase.
```

**Rules of engagement**

- One phase per Cursor task. Do not batch phases — the gates are the point.
- Never skip a gate because "the output looks right". The numbers *are* the contract.
- If a phase forces a change to a global invariant (§1), stop and amend this document first.
- Commit at every gate boundary. A green gate is a rollback point.

### Regression command (run at *every* gate)

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -t .
```

Expected: `Ran N tests ... OK`, where N grows monotonically phase by phase.

---

## 1. Global invariants — never violate these

These hold in every phase. They are the reason the system is defensible, and several are enforced
by tests that must never be weakened or skipped.

| ID | Invariant | Enforced by |
|---|---|---|
| **I-1** | **No LLM anywhere.** No OpenAI/Anthropic/Groq client, no `transformers` pipeline for generation, no API key, no network call at answer time. | Review + `test_pii_is_refused_before_retrieval` adjacency |
| **I-2** | **Answers are verbatim.** Every sentence in a `fact` answer appears character-for-character in the corpus. No paraphrasing, no templated number formatting. | `test_answers_are_grounded_in_retrieved_chunks` |
| **I-3** | **Exactly one citation per answer**, and it is the page the text actually came from. | `test_fact_answers_cite_exactly_one_source`, `test_every_answer_path_returns_a_citation` |
| **I-4** | **≤3 sentences** per answer. | `test_answers_are_at_most_three_sentences` |
| **I-5** | **`Last updated from sources:`** on every answer *except* PII refusal. | `test_every_answer_carries_timestamp_when_sourced` |
| **I-6** | **No PII is stored.** PII is refused before retrieval; matched text is scrubbed to `[redacted]` before any processing. | `test_pii_is_refused_before_retrieval`, `test_answers_never_leak_identifiers` |
| **I-7** | **Guardrails run before retrieval**, in order: PII → advice → performance → projection. | `test_advice_is_refused_with_educational_link`, `test_performance_and_projection_are_refused` |
| **I-8** | **All tunables live in `mf_rag/config.py`.** No magic numbers in pipeline code, and **no unused constants** — a constant that is declared must be read. | Phase 11 gate |
| **I-9** | **Dependencies flow downward only.** `answer.py` may know about retrieval; nothing below it may know about `answer.py` or the UI. | Import review |
| **I-10** | **Layer imports are absolute within the package** (`from .config import …`). No cross-imports into `scripts/` or `app.py`. | Lint by eye |

---

## 2. Phase dependency graph

```
P0 Scaffold & config
      │
P1 Stage 1 · Ingestion ──────────────┐
      │                              │
P2 Stage 2 · Chunking ───────────────┤   (needs Documents to chunk)
      │                              │
P3 Stages 3–4 · Embed + store ───────┤   (needs Chunks)
      │                              │
P4 Stage 5 · Retrieval ──────────────┤   (needs stored index)
      │                              │
P5 PII + guardrail patterns ─────────┤   (independent of retrieval; do in parallel if desired)
      │                              │
P6 Stage 6 · Answer composition ◄────┘   (needs retrieval + guardrails)
      │
P7 Pipeline wiring + CLI
      │
P8 Streamlit UI
      │
P9 Evaluation harness  (compare_chunking, benchmark_retrieval)
      │
P10 Deliverables  (SOURCES, SAMPLE_QA, DISCLAIMER, README)
      │
P11 Hardening & full test suite
```

**Parallelisable:** P5 can be built alongside P1–P4. P9 needs P2+P3. P10 needs P7.

---

## P0 — Scaffold & configuration

**Goal:** a package skeleton whose every tunable is already named, so no later phase invents one.

**Why first:** I-8. The constants in this phase are referenced by all six stages; changing them
later means touching every module.

**Create / modify**
| File | Contents |
|---|---|
| `requirements.txt` | `chromadb`, `sentence-transformers`, `streamlit`, `rank-bm25`, `requests`, `beautifulsoup4`, `lxml`, `transformers` |
| `mf_rag/__init__.py` | empty (or one docstring) |
| `mf_rag/config.py` | all paths + constants from `architecture.md` §7 |
| `mf_rag/sources.py` | `Source` frozen dataclass, `AMC_NAME`, `DISCLAIMER`, `EDUCATIONAL_LINKS`, `SOURCES`, `SOURCES_BY_KEY` |
| `tests/__init__.py` | empty |
| `.gitignore` | `.venv/`, `__pycache__/`, `data/chroma*/`, `data/raw/`, `data/clean/` |

**`config.py` must define exactly** (this list is canonical — copy it, do not paraphrase):

```python
ROOT, DATA_DIR, RAW_DIR, CLEAN_DIR, CHROMA_DIR, CHUNKS_PATH, LOG_PATH, DELIVERABLES_DIR
CHROMA_COLLECTION = "hdfc_mutual_funds"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
TOKENIZER_MODEL = EMBEDDING_MODEL
MAX_EMBEDDING_TOKENS = 256
CHUNK_TOKEN_TARGET = 180
CHUNK_TOKEN_HARD_MAX = 250
CHUNK_TOKEN_OVERLAP = 40
RETRIEVE_VECTOR_K = 8
RETRIEVE_BM25_K = 8
RETRIEVE_FINAL_K = 4
RRF_K = 60
MAX_ANSWER_SENTENCES = 3
```

`FACTS_PATH` and `MIN_HYBRID_SCORE` are **not** created. They are dead weight — see Phase 11.

**`sources.py` must contain** the 5 HDFC schemes from `PRD.md` §5, all Direct Growth, with the
Flexi Cap quirk encoded: `scheme_name="HDFC Equity Fund Direct Growth"` and
`page_name="HDFC Flexi Cap Direct Plan Growth"`, so `display_name` returns the page title.
`EDUCATIONAL_LINKS` keys: `statements`, `elss`, `riskometer`, `terms`, `privacy`.

**Cursor prompt**

> Create the project scaffold for a local RAG chatbot. Read `PRD.md` and `architecture.md` first.
> 1. Write `requirements.txt` with exactly: chromadb>=1.5.9, sentence-transformers>=6.1.0,
>    streamlit>=1.64.0, rank-bm25>=0.2.2, requests>=2.34.2, beautifulsoup4>=4.15.0, lxml>=6.1.3,
>    transformers>=5.17.0.
> 2. Create `mf_rag/__init__.py` and `tests/__init__.py` (empty).
> 3. Create `mf_rag/config.py` with `ROOT` derived from `Path(__file__).resolve().parents[1]` and
>    the constant list I gave you above, verbatim names and values. No other constants.
> 4. Create `mf_rag/sources.py` with a frozen dataclass `Source(key, scheme_name, scheme_short,
>    category, plan, url, page_name, isin, tags)`, a `display_name` property returning
>    `page_name or scheme_name`, and the `SOURCES` tuple of the 5 HDFC schemes from PRD.md §5.
>    Include `AMC_NAME`, `AMC_SHORT`, `DISCLAIMER` and `EDUCATIONAL_LINKS` exactly as specified.
> 5. Create `.gitignore` ignoring `.venv/`, `__pycache__/`, `data/chroma/`, `data/chroma_bench/`.
> Do not create any other file. Do not add any other constant to config.py.

**Verification**
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "from mf_rag.sources import SOURCES; print(len(SOURCES), [s.display_name for s in SOURCES])"
.\.venv\Scripts\python.exe -c "from mf_rag import config; print(config.CHROMA_COLLECTION, config.RRF_K)"
```

**Exit gate**
- [ ] installs cleanly on Python 3.12
- [ ] `len(SOURCES) == 5`, Flexi Cap `display_name` prints *"HDFC Flexi Cap Direct Plan Growth"*
- [ ] `config.py` contains no constant outside the canonical list
- [ ] `requirements.txt` has exactly 8 lines

---

## P1 — Stage 1: Ingestion (`mf_rag/ingest.py`)

**Goal:** 5 public pages → 5 `Document`s, each with ~22 typed `Fact`s, ~10 `ProseBlock`s, and a
conflict log. No embedding, no retrieval yet.

**Why this shape:** the page has no semantic DOM landmarks and its label/value pairs are split
across sibling elements, so extraction must be schema-driven (locate a known label, pair it with
the following line) rather than generic.

**Create `mf_rag/ingest.py`** with, in order:

| Symbol | Contract |
|---|---|
| `BLOCK_TAGS`, `DROP_TAGS`, `USER_AGENT` | Block-tag allowlist; decompose `script/style/noscript/svg/template/iframe`; browser UA |
| `MAX_HOLDINGS = 25`, `TOP_LEVEL_CATEGORIES = {"Equity","Hybrid","Debt"}` | limits and category sentinels |
| `LOCK_IN_RE`, `RISK_RE`, `DATE_RE`, `NAV_DATE_RE`, `BENCHMARK_RE`, `CONTACT_LABEL_RE`, `TENURE_RE`, `MANAGER_INITIALS_RE`, `MONEY_RE`, `END_MARKERS` | the page's grammar. `END_MARKERS = ("Contact Us","Download the App","Show More","Others:","© 2016-")` |
| `LABEL_FIELDS` | 7 key-fact labels → field names: `Min. for SIP`→`min_sip`, `Fund size (AUM)`→`aum`, `Expense ratio`→`expense_ratio`, `Rating`→`rating`, `Min. for 1st investment`→`min_lumpsum`, `Min. for 2nd investment`→`min_additional`, `Min. for withdrawal`→`min_withdrawal` |
| `Fact`, `ProseBlock`, `Document` | dataclasses. `Document` has `.fact(name)`, `.value(name, default)`, `.scheme_name`, `.scheme_short` properties |
| `now_iso()` | UTC ISO-8601, `timespec="seconds"` |
| `fetch(url, retries=3, timeout=45)` | `requests` + UA + `Accept-Language: en-IN`, sleep `1.5*(attempt+1)` on failure, raise `RuntimeError` naming the URL |
| `_normalise`, `_has_text_bearing_block`, `block_leaf_lines` | flatten to an ordered line list; keep only blocks with no text-bearing child; drop lines >1500 chars |
| `content_window(lines, scheme_name)` | start at scheme name followed by a top-level category or lock-in marker; cut at first `END_MARKERS` hit |
| `extract_payload(html)`, `extract_links(soup)` | `__NEXT_DATA__` → `props.pageProps.mfServerSideData`; harvest `sid`, `factsheet`, `hdfc_site` links |
| `_next_value(lines, index, stop)` | skip blanks forward, return `(value, cursor)` |
| `_parse_key_facts`, `_parse_category`, `_parse_fees_and_tax`, `_parse_exit_load_history`, `_parse_glossary`, `_parse_about`, `_parse_managers`, `_parse_holdings` | eight sequential scanners over the line list |
| `detect_conflicts(document)` | 3 checks → list of human-readable notes |
| `parse_document`, `load_documents`, `documents_to_json` | orchestration + persistence |

**Field coverage required** (23 distinct names across the corpus): `expense_ratio`, `exit_load`,
`lock_in`, `min_sip`, `min_lumpsum`, `min_additional`, `nav`, `aum`, `total_aum`, `riskometer`,
`rating`, `category`, `sub_category`, `objective`, `benchmark`, `fund_house`, `rank`,
`incorporation_date`, `launch_date`, `custodian`, `rta`, `stamp_duty`, `tax_implication`.

**`detect_conflicts` must compare** payload `nfo_risk` vs visible `riskometer`, payload
`fund_manager` vs visible text, payload `launch_date` vs visible `launch_date` — and state
"(page wins)" in each note.

**`load_documents(refresh=False, sleep=1.2)` behaviour**
- reuse `data/raw/<key>.html` when it exists and `refresh` is False (`origin="cache"`)
- else fetch, write raw, `time.sleep(sleep)` (`origin="network"`)
- write `data/clean/<key>.txt` with a `SOURCE_URL / SCHEME / CATEGORY / RETRIEVED_AT` header, then
  `== FACTS ==` lines as `[section] field | label | value`, then `== PROSE ==` as `[section] heading`
  + body
- append `{key, url, origin, lines, facts, prose_blocks, conflicts, retrieved_at}` and write
  `data/ingest_log.json`

**Tests to add now** — `tests/test_faq.py::TestIngest`
| Test | Assertion |
|---|---|
| `test_five_sources_loaded` | 5 documents, one per `SOURCES` key |
| `test_every_source_has_all_core_fields` | each doc has `expense_ratio`, `exit_load`, `aum`, `benchmark`, `riskometer`, `nav`, `min_sip` |
| `test_elss_has_lock_in_and_nil_exit_load` | ELSS `lock_in` mentions 3Y; `exit_load` is Nil |
| `test_content_window_excludes_nav_and_footer_boilerplate` | no `"Download the App"`, `"Contact Us"`, `"Show More"` in any `doc.lines` |
| `test_conflicts_prefer_visible_page` | `doc.value("riskometer")` matches the visible page, and the payload value does not appear in the visible lines |

**Cursor prompt**

> Implement Stage 1 (ingestion) of the RAG pipeline in `mf_rag/ingest.py`. Read `PRD.md` FR-1 and
> `architecture.md` §4.1 first, and `mf_rag/config.py` + `mf_rag/sources.py` which already exist.
> Requirements:
> - Fetch each of the 5 `SOURCES` URLs with `requests`, browser User-Agent, 3 retries, linear
>   backoff `1.5*(attempt+1)`, and cache the raw HTML in `data/raw/<key>.html`.
> - Parse with BeautifulSoup + lxml. Decompose script/style/noscript/svg/template/iframe, then walk
>   a block-tag allowlist keeping only elements that have no text-bearing child. Normalise
>   whitespace, drop lines over 1500 chars. This produces a flat ordered list of lines.
> - Isolate the content window: start at the line where the scheme name is immediately followed by
>   `Equity`/`Hybrid`/`Debt` or a lock-in marker like `3Y Lock-in`; truncate at the first of
>   `Contact Us`, `Download the App`, `Show More`, `Others:`, `© 2016-`.
> - Extract typed facts by locating a KNOWN label and pairing it with the next non-blank line. Do
>   not write a generic scraper. Required field names: expense_ratio, exit_load, lock_in, min_sip,
>   min_lumpsum, min_additional, nav, aum, total_aum, riskometer, rating, category, sub_category,
>   objective, benchmark, fund_house, rank, incorporation_date, launch_date, custodian, rta,
>   stamp_duty, tax_implication.
> - Also emit `ProseBlock`s for: about-the-scheme, one per fund manager (heading
>   `Fund manager: <name>`, text `<name> (<tenure>). <bio>`), top holdings (max 25, text
>   `Largest holdings by weight as shown on the scheme page: <name> (<weight>); …`), exit-load
>   revision history, and glossary definitions.
> - Parse the embedded `__NEXT_DATA__` JSON but treat it ONLY as a conflict detector. Compare its
>   `nfo_risk`, `fund_manager` and `launch_date` against the visible page and record a note ending
>   in "(page wins)" for each mismatch. Never use a payload value as the answer.
> - Persist `data/raw/`, `data/clean/<key>.txt`, and `data/ingest_log.json` with per-page counts
>   and conflicts. Respect `refresh=` so cached HTML is reused offline.
> - Add the 5 `TestIngest` tests to `tests/test_faq.py`.
> Keep every constant in this module. Do not create any other module. Do not add an LLM.

**Verification**
```powershell
.\.venv\Scripts\python.exe -c "from mf_rag.ingest import load_documents; d=load_documents(); print(len(d), sum(len(x.facts) for x in d), sum(len(x.prose) for x in d))"
.\.venv\Scripts\python.exe -c "import json,pathlib; print(sum(len(x['conflicts']) for x in json.loads(pathlib.Path('data/ingest_log.json').read_text(encoding='utf-8'))))"
.\.venv\Scripts\python.exe -m unittest tests.test_faq.TestIngest -v
```

**Exit gate**
- [ ] `5 111 52` — 5 docs, **111 facts**, **52 prose blocks** (small drift OK if a source page changed; a large drop is a failure)
- [ ] **14** conflicts logged
- [ ] `TestIngest` 5/5 green
- [ ] Re-running with no `--refresh` performs **zero** network calls

---

## P2 — Stage 2: Chunking (`mf_rag/chunking.py`)

**Goal:** 168 chunks — 111 atomic facts + 57 prose — sized by the real MiniLM tokenizer, with a
fixed-window baseline retained purely for comparison.

**Create `mf_rag/chunking.py`**

| Symbol | Contract |
|---|---|
| `SEPARATORS = ("\n\n","\n",". ","; ",", "," ","")` | recursive split ladder |
| `word_estimate`, `count_tokens_minilm` | exact wordpiece count via `AutoTokenizer.from_pretrained(TOKENIZER_MODEL)`, memoised in a module global; **fall back to `word_estimate` on any exception** |
| `Chunk` | dataclass: `chunk_id, text, cite_text, kind, section, heading, scheme_name, scheme_short, category, plan, source_url, field_name, as_of, retrieved_at, token_count, ordinal` + `to_dict()` + `metadata` property (the dict persisted to Chroma) |
| `make_id(*parts)` | `f"{parts[0]}_{sha1('||'.join(parts))[:16]}"` — stable across rebuilds so upserts replace |
| `_split_recursive(text, count, limit)` | try each separator in order, greedily pack, recurse into an over-limit part with the next separator; word-level last resort |
| `_pack(parts, count, target, overlap)` | pack to `CHUNK_TOKEN_TARGET`, carry the last `CHUNK_TOKEN_OVERLAP` words into the next chunk |
| `chunk_fact(document, fact)` | **one chunk per fact, never split.** `text = f"{scheme} ({category}). {section}. {label}: {value}"`; `cite_text = f"{label}: {value}"` (+ `" (as on {as_of})"`) |
| `chunk_prose(document, block, count)` | split → pack. `cite_text` = body for index 0, `f"{heading} (continued): {body}"` after |
| `chunk_document`, `chunk_corpus` | assemble, dedupe on lowercased `text` fingerprint, set `token_count` and per-section `ordinal` |
| `fixed_size_baseline(document, size=180, overlap=40)`, `fixed_size_chunks(...)` | `kind="fixed"`, `section="mixed"`. **Comparison only — never used in production** |
| `fact_integrity(pieces, facts)` | returns `{facts, chunks, cooccur, cooccur_ratio, orphaned, avg_words}` |

**The one rule that matters most in this phase:** `text` and `cite_text` are different strings.
`text` is label-prefixed for retrieval recall; `cite_text` is bare page wording for quoting.
Collapsing them leaks assistant-authored phrasing into a "verbatim" answer and breaks I-2.

**Tests to add now** — new class `TestChunking`
| Test | Assertion |
|---|---|
| `test_fact_chunks_are_atomic` | every `kind=="fact"` chunk contains its `field_name`'s label **and** value |
| `test_no_chunk_exceeds_hard_max` | all `token_count <= CHUNK_TOKEN_HARD_MAX` |
| `test_cite_text_appears_in_source` | for fact chunks, `cite_text` is a substring of the document's clean text |
| `test_prose_chunks_carry_heading` | every prose chunk's `text` starts with `f"{scheme} ({category}). "` |
| `test_chunk_ids_are_stable` | `chunk_corpus` twice over the same docs yields identical ids |
| `test_fixed_baseline_is_marked` | baseline chunks have `kind == "fixed"` and `section == "mixed"` |

**Cursor prompt**

> Implement Stage 2 (chunking) in `mf_rag/chunking.py`. Read `PRD.md` FR-2 and `architecture.md`
> §4.2 first; `mf_rag/ingest.py` and `mf_rag/config.py` exist.
> - Two chunk types. `fact`: exactly one chunk per `Fact`, never split. `prose`: recursive split on
>   the separator ladder `("\n\n", "\n", ". ", "; ", ", ", " ", "")` then packed to
>   `CHUNK_TOKEN_TARGET` with `CHUNK_TOKEN_OVERLAP` carried forward.
> - Size everything with the REAL MiniLM wordpiece tokenizer
>   (`AutoTokenizer.from_pretrained(TOKENIZER_MODEL)`), memoised in a module global, falling back to
>   a word count if transformers is unavailable. Never estimate with `len(text.split())` when the
>   tokenizer is available.
> - `Chunk` needs both `text` and `cite_text`. `text` is
>   `f"{scheme_name} ({category}). {section}. {label}: {value}"` for facts. `cite_text` is the bare
>   page wording: `f"{label}: {value}"`, plus `" (as on {as_of})"` when an as-of date exists. The
>   answer stage will quote `cite_text` verbatim, so it must contain no assistant-authored phrasing.
> - `make_id` = `f"{parts[0]}_{sha1('||'.join(parts)).hexdigest()[:16]}"` so ids are stable and
>   Chroma upserts replace rather than duplicate.
> - Dedupe chunks on the lowercased `text` fingerprint; set `token_count` and a per-section
>   `ordinal` (position within the section) — the answer stage needs `ordinal` to restore reading
>   order for fund-manager answers.
> - Keep `fixed_size_baseline(document, size=180, overlap=40)` and `fixed_size_chunks(...)` with
>   `kind="fixed"`, `section="mixed"`, used ONLY by the Phase 9 comparison. Never call them from
>   `chunk_corpus`.
> - Add `fact_integrity(pieces, facts)` returning `cooccur_ratio` = share of facts whose label AND
>   value land in the same chunk, plus a few `orphaned` examples.
> - Add the 6 `TestChunking` tests.
> Verify `chunk_corpus` yields 168 chunks (111 fact + 57 prose), mean ~52 tokens, and 0 chunks over
> `CHUNK_TOKEN_HARD_MAX`. Report the numbers. Do not create any other module.

**Verification**
```powershell
.\.venv\Scripts\python.exe -c "from mf_rag.chunking import *; from mf_rag.ingest import load_documents; c=chunk_corpus(load_documents(), count_tokens_minilm); import collections; print(len(c), collections.Counter(x.kind for x in c), round(sum(x.token_count for x in c)/len(c),1), max(x.token_count for x in c))"
.\.venv\Scripts\python.exe -m unittest tests.test_faq.TestChunking -v
```

**Exit gate**
- [ ] `168`, `{'fact': 111, 'prose': 57}`, mean ≈52, **max ≤ 250**
- [ ] `TestChunking` 6/6 green
- [ ] `grep`-check: `chunk_corpus` does not reference `fixed_size_chunks`

---

## P3 — Stages 3 & 4: Embedding + vector store (`mf_rag/embed_store.py`)

**Goal:** 168 chunks embedded with MiniLM and upserted into a persistent cosine Chroma collection.

**Create `mf_rag/embed_store.py`**

| Symbol | Contract |
|---|---|
| `_MODEL`, `_CLIENT` | module-level singletons — a Streamlit session must load the model once |
| `VectorHit(chunk_id, score, text, metadata)` | `score = 1.0 - distance` |
| `IndexStats(chunks, dimensions, model, collection)` + `.summary()` | printed by `build_index.py` |
| `get_model()` | `SentenceTransformer(EMBEDDING_MODEL)` |
| `get_client()` | `chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False))` |
| `get_collection()` | `get_or_create_collection(CHROMA_COLLECTION, metadata={"hnsw:space":"cosine","embed_model":EMBEDDING_MODEL})` |
| `embed_texts(texts)` | `encode(..., batch_size=32, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)` |
| `upsert_chunks(chunks, batch_size=64)` | delete-then-upsert in batches of 64; metadata = `{**chunk.metadata, "chunk_id", "cite_text"}`; return `IndexStats` with a dimension probe |
| `query_vector(query, top_k, where=None)` | `collection.query(query_embeddings=..., n_results=min(top_k, count), where=where, include=["documents","metadatas","distances"])`; return `[]` if the collection is empty |
| `persist_chunks`, `load_chunks`, `persist_documents` | `data/chunks.jsonl` (JSONL) and `data/documents.json` |
| `index_exists()` | `CHUNKS_PATH.exists() and get_collection().count() > 0` |
| `build_index(refresh=False)` | `load_documents → chunk_corpus → persist_chunks → persist_documents → upsert_chunks` |

**Create `scripts/build_index.py`** — `--refresh` flag, then print `stats.summary()`, the
`kind` histogram, the distinct scheme count, and the sorted distinct section list.

**Why `chunks.jsonl` exists even though Chroma does too (AD-6):** BM25 needs the chunk objects in
RAM regardless, and stage 6 needs `cite_text` and `ordinal`, which retrieval must return intact.
Chroma alone is not a sufficient system of record.

**Tests to add now** — new class `TestStore`
| Test | Assertion |
|---|---|
| `test_index_has_all_chunks` | `get_collection().count() == len(load_chunks())` |
| `test_embedding_dim_is_384` | `len(embed_texts(["probe"])[0]) == EMBEDDING_DIM` |
| `test_vectors_are_normalised` | `‖v‖₂ == 1.0` within 1e-5 |
| `test_metadata_carries_citation_fields` | every stored metadata has non-empty `source_url` and `retrieved_at` |
| `test_upsert_is_idempotent` | run `upsert_chunks` twice; `count()` unchanged |
| `test_index_exists_true_after_build` | `index_exists()` is True |

**Cursor prompt**

> Implement Stages 3 and 4 in `mf_rag/embed_store.py`, plus the driver `scripts/build_index.py`.
> Read `PRD.md` FR-3/FR-4 and `architecture.md` §4.3-4.4 first. `mf_rag/chunking.py` and
> `mf_rag/ingest.py` already exist.
> - Load `SentenceTransformer(EMBEDDING_MODEL)` into a module-level `_MODEL` singleton, and the
>   Chroma `PersistentClient` into `_CLIENT`. Create `CHROMA_DIR` if needed and pass
>   `Settings(anonymized_telemetry=False)` — telemetry must stay off.
> - Collection `CHROMA_COLLECTION` with metadata `{"hnsw:space": "cosine", "embed_model":
>   EMBEDDING_MODEL}`. Vectors are L2-normalised, so cosine is the correct space.
> - `embed_texts` must use `batch_size=32, normalize_embeddings=True, convert_to_numpy=True,
>   show_progress_bar=False`.
> - `upsert_chunks` must delete the ids it is about to write, then upsert in batches of 64, storing
>   `documents=[c.text]`, the embedding, and metadata `{**c.metadata, "chunk_id": c.chunk_id,
>   "cite_text": c.cite_text}`. Return `IndexStats` including a real dimension probe. Ids are
>   content-hashed, so a rebuild must replace, never duplicate.
> - `query_vector(query, top_k, where=None)` must pass `where` straight through to Chroma (stage 5
>   uses `{"chunk_id": {"$in": [...]}}` for section filtering) and return `[]` when the collection
>   is empty rather than raising.
> - Also provide `persist_chunks`/`load_chunks` (JSONL at `CHUNKS_PATH`), `persist_documents`
>   (`data/documents.json`), `index_exists()`, and `build_index(refresh=False)` that runs
>   load → chunk → persist → upsert.
> - `scripts/build_index.py`: argparse `--refresh`, then print the stats summary, the chunk-kind
>   histogram, the distinct scheme count and the sorted distinct sections.
> - Add the 6 `TestStore` tests. Note these need a built index; make them skip with a clear message
>   if `CHUNKS_PATH` is missing rather than failing confusingly.
> Verify: 168 chunks in the collection, 384 dimensions, all vectors unit-norm, and a second
> `upsert_chunks` leaves the count unchanged. Do not create any other module.

**Verification**
```powershell
.\.venv\Scripts\python.exe scripts\build_index.py
.\.venv\Scripts\python.exe scripts\build_index.py
.\.venv\Scripts\python.exe -m unittest tests.test_faq.TestStore -v
```

**Exit gate**
- [ ] `168 chunks | 384-dim | all-MiniLM-L6-v2 | collection=hdfc_mutual_funds`
- [ ] Second run reports the same count (idempotent upsert)
- [ ] `TestStore` 6/6 green
- [ ] `data/chroma/` contains no telemetry config

---

## P4 — Stage 5: Hybrid retrieval (`mf_rag/retrieve.py`)

**Goal:** dense + BM25 fused with RRF, scheme boost, and section filtering.

**Create `mf_rag/retrieve.py`**

| Symbol | Contract |
|---|---|
| `TOKEN_RE` | `r"[a-z0-9]+(?:\.[0-9]+)?%?"` — keeps `1.03` and `1.21%` intact |
| `BOOST_FIELDS` | `("expense_ratio","exit_load","min_sip","min_lumpsum","benchmark","riskometer","lock_in")` |
| `tokenize(text)` | `TOKEN_RE.findall(text.lower())` |
| `RetrievedChunk` | carries `chunk_id, text, cite_text, scheme*, section, heading, kind, field_name, source_url, rrf_score, vector_rank, bm25_rank, vector_score, bm25_score, ordinal, reasons` |
| `HybridRetriever.__init__(chunks)` | index by id; build `BM25Okapi` over `f"{c.heading} {c.cite_text} {c.text}"` — heading first so a field label is the most salient token |
| `_bm25(query, top_k, allowed=None)` | score > 0 only, filtered to `allowed`, return `(chunk_id, score, rank)` |
| `_detect_scheme(query)` | full names (`display_name`, `scheme_name`, `scheme_short`) first, then aliases: `large cap`, `flexi cap`, `flexicap`, `hdfc equity fund`, `elss`, `tax saver`, `80c`, `small cap`, `balanced advantage` |
| `_dense(query, top_k, allowed=None)` | `query_vector(query, top_k, where={"chunk_id": {"$in": sorted(allowed)}} if allowed else None)` |
| `search(query, top_k=RETRIEVE_FINAL_K, *, section=None)` | the fusion below |
| `get_retriever(chunks=None)`, `embed_query(query)` | module singleton + normalised query encoder |

**`search` algorithm — implement exactly this order**
```
allowed  = {c.chunk_id for c in chunks if c.section == section}  if section else None
if section was given and allowed is empty -> return []
dense  = _dense(query, RETRIEVE_VECTOR_K, allowed)
sparse = _bm25(query, RETRIEVE_BM25_K, allowed)
target = _detect_scheme(query)

for rank, hit in enumerate(dense, 1):   score[hit] += 1/(RRF_K + rank)
for cid, s, rank in sparse:              score[cid] += 1/(RRF_K + rank)

for each candidate:
    rrf *= 1.35   if target and chunk.scheme_short == target      (record reason)
    rrf *= 1.05   if target and chunk.field_name in BOOST_FIELDS
return sorted by rrf desc [:top_k]
```

Populate `reasons` with `"dense rank N"` / `"bm25 rank N"` / `"scheme match: X"` — the UI
retrieval trace renders them.

**Do not** add a score threshold or a cosine-distance cut-off. RRF is rank-based; inventing a
minimum score here would silently drop the exact-token hits BM25 exists to find.

**Tests to add now** — new class `TestRetrieval`
| Test | Assertion |
|---|---|
| `test_dense_and_bm25_both_contribute` | at least one result has `vector_rank` set and one has `bm25_rank` set |
| `test_scheme_boost_prefers_named_scheme` | for a query naming ELSS, the top result's `scheme_short == "ELSS"` |
| `test_rrf_matches_manual_computation` | recompute `Σ 1/(RRF_K+rank)` for the top hit and compare to `rrf_score` (ignoring boosts) |
| `test_section_filter_restricts_results` | `search(q, section="Holdings")` returns only `section == "Holdings"` |
| `test_section_filter_on_empty_section` | `search(q, section="Nonexistent")` returns `[]` |
| `test_scheme_alias_resolution` | `_detect_scheme` resolves `80c`, `tax saver`, `bluechip`, `flexicap` |
| `test_top_k_is_respected` | `len(search(q, top_k=3)) <= 3` |

**Cursor prompt**

> Implement Stage 5 (hybrid retrieval) in `mf_rag/retrieve.py`. Read `PRD.md` FR-5 and
> `architecture.md` §4.5 first. The index is already built; `mf_rag/embed_store.py` exposes
> `query_vector(query, top_k, where=...)` and `load_chunks()`.
> - `HybridRetriever.__init__` must build a `rank_bm25.BM25Okapi` over
>   `f"{chunk.heading} {chunk.cite_text} {chunk.text}"` for each chunk — heading first so a field
>   label is the most salient token in its own document. Use
>   `TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?%?")` so `1.03` and `1.21%` survive tokenising.
> - `search(query, top_k=RETRIEVE_FINAL_K, *, section=None)`:
>   1. If `section` is given, build `allowed = {chunk_id where section matches}`; if it is empty,
>      return `[]` immediately.
>   2. Run dense `RETRIEVE_VECTOR_K` and BM25 `RETRIEVE_BM25_K` over the same `allowed` set. Pass
>      dense filtering to Chroma as `where={"chunk_id": {"$in": sorted(allowed)}}`.
>   3. Fuse with Reciprocal Rank Fusion: for each ranked list, `score[chunk] += 1/(RRF_K + rank)`
>      where `RRF_K` comes from config. RRF is rank-based on purpose — a cosine distance and a
>      BM25 score are not comparable, their ranks are.
>   4. Apply `* 1.35` when the chunk's `scheme_short` matches a scheme named in the query, and
>      `* 1.05` when a scheme was named and the chunk's `field_name` is in `BOOST_FIELDS`
>      `(expense_ratio, exit_load, min_sip, min_lumpsum, benchmark, riskometer, lock_in)`.
>   5. Sort by `rrf_score` desc and return `[:top_k]`.
> - Detect the named scheme by substring: try `display_name`, `scheme_name`, `scheme_short` for
>   every source first, then aliases `large cap`, `flexi cap`, `flexicap`, `hdfc equity fund`,
>   `elss`, `tax saver`, `80c`, `small cap`, `balanced advantage`.
> - Return `RetrievedChunk` objects carrying `vector_rank`, `bm25_rank`, `vector_score`,
>   `bm25_score`, `rrf_score`, `ordinal`, `cite_text` and a `reasons` list
>   (`"dense rank 3"`, `"bm25 rank 1"`, `"scheme match: ELSS"`) — the UI renders these.
> - Add `get_retriever(chunks=None)` behind a module-level singleton and `embed_query(query)`.
> - DO NOT add a minimum-score threshold. Do not re-rank inside `search`. Do not create another
>   module.
> - Add the 7 `TestRetrieval` tests, including one that recomputes the RRF score by hand to prove
>   the fusion formula is implemented as specified.
> Verify: for "expense ratio of HDFC ELSS" the top hit is an `expense_ratio` fact chunk for ELSS.
> Report the top 5 hits with their scores.

**Verification**
```powershell
.\.venv\Scripts\python.exe -c "from mf_rag.retrieve import get_retriever; r=get_retriever(); [print(round(c.rrf_score,5), c.scheme_short, c.field_name or c.heading, c.reasons) for c in r.search('expense ratio of HDFC ELSS Tax Saver Fund')]"
.\.venv\Scripts\python.exe -m unittest tests.test_faq.TestRetrieval -v
```

**Exit gate**
- [ ] Top hit for *"expense ratio of HDFC ELSS"* is the ELSS `expense_ratio` fact chunk
- [ ] `TestRetrieval` 7/7 green
- [ ] `bm25 rank` and `dense rank` both appear in `reasons` for at least one query

---

## P5 — PII guard and refusal patterns (`mf_rag/pii.py`)

**Goal:** the guardrail vocabulary, buildable in parallel with P1–P4.

**Create `mf_rag/pii.py`**

| Symbol | Contract |
|---|---|
| `PAN_RE` | `\b[A-Z]{5}\d{4}[A-Z]\b` |
| `AADHAAR_RE` | `\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b` |
| `EMAIL_RE`, `PHONE_RE`, `OTP_RE`, `CARD_RE`, `ACCOUNT_RE`, `IFSC_RE`, `PASSPORT_RE` | phone must use a lookbehind/lookahead so it does not match inside a longer digit run |
| `REDACTED = "[redacted]"` | the only substitution string |
| `REFUSAL` | the PII refusal text, ending in `"- see {link} for how personal data is handled."` |
| `RULES` | ordered 9-tuple of `(name, pattern)`: PAN, Aadhaar, email address, phone number, OTP, card number, account or folio number, IFSC, passport number |
| `PiiResult(is_pii, kinds)` + `.message` | `.message` formats `REFUSAL` with `EDUCATIONAL_LINKS["privacy"]` |
| `scan(text)` | names of every matching rule |
| `scrub(text)` | every match replaced with `[redacted]`, all rules applied |

**Add the refusal regexes to `mf_rag/answer.py`** (the file itself is created in P6 — if P6 has
not run yet, hold these constants for P6; they belong to the same module):
`ADVICE_PATTERNS` + `ADVICE_RE` (~20 alternatives: `should i`, `should i buy/sell/redeem/…`,
`is it a good time`, `which fund should`, `best/top fund for`, `recommend`, `how much should i`,
`market timing`, `my portfolio`, `help me choose`, `compare … better`), `PERFORMANCE_RE`
(`return(s)`, `performance`, `profit`, `growth`, `cagr`, `alpha`, `sharpe`, `outperform`,
`top performer`, `rank(ing)`), `COMPUTE_RE` (`calculate`, `compute`, `project`, `estimate`,
`forecast`, `what if`, `simulate`, `how much would/will/can i`).

**Tests to add now** — new class `TestPiiGuard`
| Test | Assertion |
|---|---|
| `test_detects_identifiers` | one positive string per rule → `is_pii` and the right `kinds` entry |
| `test_does_not_flag_ordinary_questions` | the 8 briefed question types, the 3 example questions and a greeting all return `is_pii == False` |
| `test_scrub_redacts_every_match` | after `scrub`, none of the original identifiers remain |
| `test_refusal_message_cites_privacy_policy` | `PiiResult(True).message` contains the privacy URL |

**Cursor prompt**

> Implement the PII guard in `mf_rag/pii.py`. Read `PRD.md` FR-7 and `architecture.md` §10 first.
> `mf_rag/sources.py` exists and provides `EDUCATIONAL_LINKS["privacy"]`.
> - Define 9 regexes as an ordered `RULES` tuple of `(human_name, compiled_pattern)`:
>   PAN `\b[A-Z]{5}\d{4}[A-Z]\b`; Aadhaar `\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b`; email;
>   Indian phone (use a negative lookbehind for a digit and a negative lookahead so it cannot match
>   inside a longer digit run, and allow an optional `+91` prefix); OTP (match the words
>   otp / verification code / one time password); card number (4 groups of 4 digits);
>   account/folio/demat number; IFSC `\b[A-Z]{4}0[A-Z0-9]{6}\b`; passport number.
> - `scan(text) -> PiiResult` returns `is_pii` plus the `kinds` that matched.
> - `scrub(text) -> str` replaces EVERY match with the literal `[redacted]`, applying all rules.
> - `REFUSAL` is the user-facing text: state that personal identifiers cannot be accepted, list
>   the categories, state that nothing from the message was stored because the assistant only
>   answers published fund facts, and end with `"- see {link} for how personal data is handled."`
> - `PiiResult.message` formats `REFUSAL` with `EDUCATIONAL_LINKS["privacy"]`.
> - Add the 4 `TestPiiGuard` tests. The negative test is the important one: assert that ordinary
>   questions like "What is the expense ratio of HDFC Large Cap Fund?", "Is there a lock-in period
>   on HDFC ELSS?", "hi", and "Should I buy HDFC Small Cap Fund?" are NOT flagged as PII.
> Do not create any other module. Do not wire this into a pipeline yet — Phase 6 does that.

**Verification**
```powershell
.\.venv\Scripts\python.exe -c "from mf_rag.pii import scan, scrub; print(scan('My PAN is ABCDE1234F and email a@b.com').kinds); print(scrub('call 9876543210 or mail a@b.com'))"
.\.venv\Scripts\python.exe -c "from mf_rag.pii import scan; print(scan('What is the expense ratio of HDFC Large Cap Fund?').is_pii)"
.\.venv\Scripts\python.exe -m unittest tests.test_faq.TestPiiGuard -v
```

**Exit gate**
- [ ] All 9 identifier types detected by name
- [ ] `is_pii == False` for every ordinary question, including the advice question
- [ ] `scrub` leaves no original identifier in the string
- [ ] `TestPiiGuard` 4/4 green

---

## P6 — Stage 6: answer composition (`mf_rag/answer.py`)

**Goal:** the whole system in one module — guardrail ladder, intent routing, extractive answers,
one citation, conditional timestamp.

**Create `mf_rag/answer.py`**

**Part A — data types**
```python
@dataclass
class Answer:
    text: str
    citations: list[str] = field(default_factory=list)
    retrieved_at: str = ""
    intent: str = "unknown"
    kind: str = "fact"
    chunks: list[RetrievedChunk] = field(default_factory=list)
    refused: bool = False
    pii_kinds: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def primary_citation(self) -> str: return self.citations[0] if self.citations else ""
    def render(self) -> str: ...   # text + "Source: <url>" + "Last updated from sources: <ts>"
```
`kind` is the UI's rendering contract. Allowed values: `fact`, `guidance`, `greeting`, `empty`,
`out_of_scope`, `advice_refusal`, `performance_refusal`, `compute_refusal`, `pii`.

**Part B — patterns and tables**
- `ADVICE_PATTERNS` / `ADVICE_RE`, `PERFORMANCE_RE`, `COMPUTE_RE` (from P5)
- `INTENTS`: an **ordered** tuple of `(intent_name, compiled_pattern, field_tuple)`, matched
  first-hit-wins. Order matters — `exit_load_history` before `exit_load`, `sub_category` before
  `category`. Required intents: `exit_load_history`, `statement`, `expense_ratio`, `exit_load`,
  `lock_in`, `min_sip`, `min_lumpsum`, `benchmark`, `riskometer`, `rating`, `nav`, `aum`,
  `stamp_duty`, `tax`, `objective`, `manager`, `holdings`, `launch_date`, `rta`, `custodian`,
  `sub_category`, `category`, `fund_house`.
  Each pattern must be generous about phrasing — `"how risky"`, `"80C"`, `"3 year"`,
  `"how old is"`, `"inception date"`, `"charges"`, `"TER"`.
- `FIELD_QUERY_HINT`: 20 entries mapping a field name to lexical retry hints, e.g.
  `"expense_ratio": "expense ratio"`, `"min_sip": "Min. for SIP minimum SIP amount"`.
- `SECTION_QUERY_HINT` + `SECTION_FOR_INTENT`: `manager`→`Fund management`,
  `holdings`→`Holdings`, `exit_load_history`→`Fees, exit load and tax`.
- `STOPWORDS`, `MIN_LEXICAL_OVERLAP = 0.5`
- Refusal/guidance templates: `ADVICE_REFUSAL` (point at a SEBI-registered adviser),
  `PERFORMANCE_REFUSAL` (point at the scheme page and its official factsheet),
  `COMPUTE_REFUSAL`, `OUT_OF_SCOPE`, `GREETING`

**Part C — helpers**
| Function | Contract |
|---|---|
| `content_tokens(text)` | lowercase alphanumeric tokens minus `STOPWORDS` |
| `_sentences(text, limit)` | split on `(?<=[.!?])\s+`, keep the first `limit`, re-append a final `.` if truncated |
| `_parse_manager(cite_text)` | `(name, tenure)` from `Name (Jul 2022 - Present)` |
| `detect_scheme(query)` | same two-pass alias logic as retrieval (keep both in sync) |
| `detect_intent(query)` | `(intent, fields)`, `"unknown"`/`()` on no match |
| `_pick_fact_chunk(chunks, fields, scheme)` | first field that has a chunk; prefer the chunk matching `scheme` |
| `_fact_sentence(chunk, *, scoped=False)` | `f"{scheme_name} - {cite_text}."`, or `f"On the {scheme_name} page: {cite_text}."` when unscoped. **Never reformat the value.** |
| `_link_for(scheme)`, `_educational(name)` | source URL for a scheme; `EDUCATIONAL_LINKS` lookup with a safe default |

**Part D — `FAQAssistant`**
```python
class FAQAssistant:
    def __init__(self, retriever: HybridRetriever, top_k: int = 6) -> None:
        # build self._stamps: {source_url -> retrieved_at} and self._vocabulary from
        # content_tokens(heading) + content_tokens(cite_text) across all chunks
        # self._corpus_stamp = max(self._stamps.values())
```
Methods: `_stamp_for(url)`, `_targeted(cleaned, scheme, hint, section=None)` (re-query as
`f"{cleaned} {display_name} {hint}"`), `_in_scope(query)` (lexical overlap ≥ 0.5),
`_manager_answer(scheme, traced)` (filter section `Fund management`, sort by `ordinal`, parse
name/tenure, name the primary then "The page also lists …").

**`ask(query)` — implement this ladder in this exact order**
```
 1. pii.scan(original) -> PII_REFUSAL            kind="pii", refused=True, cite privacy,
                                                  retrieved_at="" (NO stamp), nothing stored
 2. cleaned = pii.scrub(original); if not cleaned -> OUT_OF_SCOPE, kind="empty"
 3. if len(cleaned) < 3 -> GREETING              kind="greeting", no retrieval
 4. ADVICE_RE -> ADVICE_REFUSAL                 kind="advice_refusal", refused=True
  5. scheme = detect_scheme(cleaned)
     intent, fields = detect_intent(strip_scheme_names(cleaned, scheme))
                                                   # scheme name is metadata, not the
                                                   # question: see "The three rules" below
 6. intent == "statement" -> guidance text, cite EDUCATIONAL_LINKS["statements"]
 7. intent in SECTION_QUERY_HINT -> section-filtered _targeted(); scope to scheme;
       manager  -> _manager_answer
       holdings -> the chunk whose heading starts with "Top"
       else     -> the chunk whose heading mentions "exit load revision history"
       fall through if nothing found
 8. COMPUTE_RE and intent in ("unknown","performance") -> COMPUTE_REFUSAL
 9. chunks = retriever.search(cleaned, top_k=self.top_k)
10. if fields: chosen = _pick_fact_chunk(chunks, fields, scheme)
       if None: for each field, _targeted(cleaned, scheme, FIELD_QUERY_HINT[field]) and retry
       if chosen: return fact answer, cite chosen.source_url, stamp _stamp_for(that url)
11. PERFORMANCE_RE and not fields -> PERFORMANCE_REFUSAL
12. intent == "unknown" and not _in_scope(cleaned) -> OUT_OF_SCOPE
13. if not chunks -> OUT_OF_SCOPE
14. if scheme: re-filter chunks to that scheme
15. best = chunks[0]; if not best.field_name -> OUT_OF_SCOPE   # prose can't answer a fact
16. return fact answer from best
```

**The three rules that make this defensible**
- Step 1 runs **before** step 9. A PII message never reaches the vector store.
- Only paths that consult **no page** return without `retrieved_at` — the PII refusal and
  the greeting. Everything else that cites a URL carries the stamp.
- Steps 10 and 14 together are what stop cross-scheme contamination.

**Two corrections applied during P6 implementation** (both are in `answer.py` and tested):

1. **`strip_scheme_names(cleaned, scheme)` before `detect_intent`.** Scheme names are metadata,
   not the question. Without stripping, the words *inside* a name hijack routing: "What is the
   RTA of HDFC ELSS **Tax** Saver Fund?" was detected as a **tax** question (the name contains
   the literal word "Tax"), so it fell through to a scope message instead of returning the RTA.
   Longest match wins, so the alias `tax saver` is removed before the bare alias `elss`.

2. **`names_unresolved_scheme(cleaned, scheme)` after step 6.** Previously, "What is the expense
   ratio of HDFC Parag Parhat Fund?" resolved no scheme and then returned **HDFC Large Cap's**
   expense ratio — a confident, cited, wrong number, which is worse than admitting ignorance. The
   check is a proper noun that is neither sentence-initial nor part of any registered scheme name,
   and the vocabulary is derived from `SOURCES` + `SCHEME_ALIASES` so it cannot drift. Known
   trade-off: alias matching is substring-based, so "HDFC Ultra Large Cap Fund" still resolves to
   HDFC Large Cap. Tightening that needs token-boundary matching plus a qualifier blocklist.

**Tests to add now** — new class `TestRouting` and `TestAnswers`
| Test | Assertion |
|---|---|
| `test_detect_scheme_aliases` | `80c`, `tax saver`, `bluechip`, `flexicap` resolve |
| `test_detect_intent_prefers_specific_over_generic` | *"exit load history"* → `exit_load_history`, not `exit_load`; *"sub-category"* → `sub_category` |
| `test_intent_phrasings_route_to_the_right_field` | 10 paraphrases → expected intent |
| `test_answers_report_the_asked_field_not_a_neighbour` | exit-load question never returns the expense ratio |
| `test_aum_reports_scheme_size_not_amc_total` | `"Total AUM"` never appears in an AUM answer |
| `test_advice_patterns` | 10 advice phrasings all match `ADVICE_RE` |
| `test_fact_answers_cite_exactly_one_source` | `len(answer.citations) == 1` for all 5 schemes × 5 fields |
| `test_answers_are_at_most_three_sentences` | sentence count ≤ 3 everywhere |
| **`test_answers_are_grounded_in_retrieved_chunks`** | **every fact answer's sentences appear character-for-character in the corpus text** |
| `test_per_scheme_answers_do_not_cross_contaminate` | the ELSS answer contains no other scheme's name |
| `test_manager_answer_names_primary_and_others` | primary + "also lists" |
| `test_holdings_answer_is_grounded` | holding names and weights come from the page |
| `test_statement_question_returns_guidance` | `kind == "guidance"`, cites the help centre |
| `test_advice_is_refused_with_educational_link` | `refused`, cites an educational URL |
| `test_performance_and_projection_are_refused` | 6 queries → the two refusal kinds |
| `test_pii_is_refused_before_retrieval` | `chunks == []` on a PII answer |
| `test_out_of_scope_query` | a weather question → `kind == "out_of_scope"` |
| `test_answers_never_leak_identifiers` | no PAN/Aadhaar/IFSC/phone pattern in any answer text |
| `test_every_answer_carries_timestamp_when_sourced` | every kind except `pii` has a stamp |
| `test_render_omits_timestamp_but_keeps_privacy_link_for_pii_refusal` | exact I-5 behaviour |
| `test_every_answer_path_returns_a_citation` | no path returns an empty `citations` |
| `test_greeting` | `"hi"` → `kind == "greeting"` |

**Cursor prompt**

> Implement Stage 6 (answer composition) in `mf_rag/answer.py`. Read `PRD.md` FR-6/FR-7 and
> `architecture.md` §4.6 and §5 first. `mf_rag/retrieve.py` (`HybridRetriever.search`,
> `RetrievedChunk`), `mf_rag/pii.py` (`scan`, `scrub`, `REFUSAL`) and `mf_rag/sources.py` exist.
> There is NO language model in this system. Answers are verbatim excerpts from retrieved chunks.
>
> Build, in this order:
> 1. `Answer` dataclass: `text, citations, retrieved_at, intent, kind, chunks, refused, pii_kinds,
>    notes` with a `primary_citation` property and a `render()` that prints text, `Source: <url>` and
>    `Last updated from sources: <timestamp>`. `kind` must only ever be one of: fact, guidance,
>    greeting, empty, out_of_scope, advice_refusal, performance_refusal, compute_refusal, pii.
> 2. `ADVICE_RE` (~20 alternatives incl. "should i", "is now a good time", "which fund should",
>    "best fund for", "recommend", "how much should i", "market timing", "my portfolio",
>    "help me choose", "compare … better"), `PERFORMANCE_RE` (returns, performance, profit, cagr,
>    alpha, sharpe, outperform, top performer, ranking), `COMPUTE_RE` (calculate, project, estimate,
>    what if, simulate).
> 3. `INTENTS`: an ORDERED first-match-wins tuple of `(name, pattern, fields)`. Order is
>    load-bearing — `exit_load_history` must precede `exit_load`, `sub_category` must precede
>    `category`. Make patterns generous: "how risky", "80C", "3 year", "how old is", "inception
>    date", "TER", "charges". Include at least: exit_load_history, statement, expense_ratio,
>    exit_load, lock_in, min_sip, min_lumpsum, benchmark, riskometer, rating, nav, aum, stamp_duty,
>    tax, objective, manager, holdings, launch_date, rta, custodian, sub_category, category,
>    fund_house.
> 4. `FIELD_QUERY_HINT` (20 field name -> lexical retry hint) and `SECTION_QUERY_HINT` /
>    `SECTION_FOR_INTENT` (manager -> Fund management, holdings -> Holdings, exit_load_history ->
>    Fees, exit load and tax).
> 5. Refusal/guidance templates `ADVICE_REFUSAL`, `PERFORMANCE_REFUSAL`, `COMPUTE_REFUSAL`,
>    `OUT_OF_SCOPE`, `GREETING` — each with a `{link}` slot.
> 6. Helpers: `_sentences(text, limit)`, `_parse_manager`, `detect_scheme` (two-pass: full names
>    then aliases), `detect_intent`, `_pick_fact_chunk` (prefer the named scheme),
>    `_fact_sentence` (`"{scheme} - {cite_text}."` or `"On the {scheme} page: {cite_text}."` when
>    unscoped — never reformat the value), `_link_for`, `_educational`.
> 7. `FAQAssistant.__init__` must precompute `_stamps` ({source_url -> retrieved_at}) and
>    `_vocabulary` (content tokens of headings and cite_texts) plus `_corpus_stamp = max(stamps)`.
> 8. `ask(query)` implementing EXACTLY this ladder, in this order:
>    pii.scan -> refuse as kind="pii" with the privacy citation, `refused=True` and
>    `retrieved_at=""` (no stamp) and NO retrieval;
>    pii.scrub; empty -> out_of_scope/"empty"; len<3 -> greeting;
>    ADVICE_RE -> advice_refusal;
>    detect_scheme + detect_intent; "statement" -> guidance citing the help centre;
>    section intents -> section-filtered `_targeted(cleaned, scheme, hint, section)` then
>      manager via `_manager_answer` (sort by ordinal, parse "Name (tenure)", name primary then
>      "The page also lists ..."), holdings via the chunk whose heading starts with "Top";
>    COMPUTE_RE with no field -> compute_refusal;
>    `retriever.search(cleaned, top_k=6)`;
>    if the intent maps to fields: `_pick_fact_chunk`, and on a miss re-query per field with its
>      `FIELD_QUERY_HINT`;
>    PERFORMANCE_RE with no field -> performance_refusal;
>    unknown intent and lexical overlap < 0.5 -> out_of_scope;
>    no chunks -> out_of_scope;
>    if a scheme was named, re-filter chunks to that scheme;
>    if the best chunk has no `field_name`, return out_of_scope rather than quoting prose as a fact.
> 9. Add the 22 `TestRouting` and `TestAnswers` tests listed in the plan. The grounding test is the
>    most important: assert every sentence of every fact answer appears character-for-character in
>    the concatenated corpus text. Build the corpus text by joining all `cite_text` and `text`
>    values from the loaded chunks.
> Verify: the 8 briefed question types answer with a citation and a stamp; "Should I buy HDFC
> Small Cap Fund?" is refused; "My PAN is ABCDE1234F, check my returns" is refused as PII with no
> stamp; "What is the weather in Mumbai" returns out_of_scope. Report all 12.

**Verification**
```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_faq.TestRouting tests.test_faq.TestAnswers -v
```

**Exit gate**
- [ ] `TestRouting` 5/5 and `TestAnswers` 17/17 green
- [ ] PII answer has `chunks == []` **and** `retrieved_at == ""` **and** exactly 1 citation
- [ ] No fact answer contains a number that is not in the corpus
- [ ] Total suite: **49 tests**, all green

---

## P7 — Pipeline wiring and CLI (`mf_rag/pipeline.py`, `scripts/ask.py`)

**Goal:** one cached entry point, usable from Python and from a terminal.

**Create `mf_rag/pipeline.py`**
- `EXAMPLE_QUESTIONS` — **exactly 3**: expense ratio of HDFC Flexi Cap Fund; lock-in on HDFC ELSS
  Tax Saver Fund; minimum SIP for HDFC Small Cap Fund.
- `class PipelineMissing(Exception)`
- `@lru_cache(maxsize=1) def get_assistant() -> FAQAssistant` — raise `PipelineMissing("No index
  found. Run: python scripts/build_index.py")` when `index_exists()` is False, else
  `FAQAssistant(HybridRetriever(load_chunks()))`
- `def ask(query) -> Answer`

**Create `scripts/ask.py`** — positional `question` (nargs="*"), `--debug` flag, `sys.path` insert
of the repo root. With `--debug`, print `intent/kind/refused` then per chunk
`rrf / vec / bm25 / [kind/field] scheme :: cite_text` and its `reasons`. Interactive mode when no
question is given. Exit 1 on `PipelineMissing`.

**Tests to add now** — new class `TestPipeline`
| Test | Assertion |
|---|---|
| `test_get_assistant_is_cached` | two calls return the same object |
| `test_pipeline_missing_message_names_the_script` | message contains `build_index.py` |
| `test_three_example_questions` | `len(EXAMPLE_QUESTIONS) == 3`, each returns a cited answer |
| `test_ask_module_level_helper` | `ask("...")` returns an `Answer` |

**Cursor prompt**

> Create `mf_rag/pipeline.py` and `scripts/ask.py`. Read `architecture.md` §3 and §14.
> - `pipeline.py` must expose `EXAMPLE_QUESTIONS` (exactly 3 strings: the expense ratio of HDFC
>   Flexi Cap Fund, whether there is a lock-in on HDFC ELSS Tax Saver Fund, the minimum SIP for HDFC
>   Small Cap Fund), a `PipelineMissing` exception, `@lru_cache(maxsize=1) get_assistant()` that
>   raises `PipelineMissing("No index found. Run: python scripts/build_index.py")` when
>   `index_exists()` is False, and a module-level `ask(query) -> Answer`.
> - `scripts/ask.py`: argparse with a positional `question` (nargs="*") and a `--debug` flag; insert
>   the repo root on `sys.path`; on `PipelineMissing` print to stderr and return 1. With `--debug`,
>   after the rendered answer print `intent=... kind=... refused=...` and then for each retrieved
>   chunk its `rrf_score`, `vector_score`, `bm25_score`, `[kind/field_name or heading]`, the scheme
>   short name, a 70-char slice of `cite_text`, and its `reasons` list. With no question argument,
>   enter an interactive REPL.
> - Add the 4 `TestPipeline` tests.
> Verify `python scripts\ask.py "expense ratio of HDFC Large Cap Fund" --debug` prints a cited
> answer and a retrieval trace. Report the output.

**Exit gate**
- [x] `ask.py` answers from the terminal with and without `--debug`
- [x] `TestPipeline` green · suite is **72 tests** (the "53" originally written here assumed
      no extra regression tests were added in P2–P6; see the per-phase counts in the summary)

**Corrections applied during P7 implementation**

`pipeline.py` and `ask.py` already existed and matched this spec, so P7 was mostly
verification. Two real bugs were found in `ask.py` and fixed:

1. **The CLI crashed on every rupee-denominated answer.** `UnicodeEncodeError: 'charmap'
   codec can't encode character '\u20b9'`. The chunk store contains U+20B9 in every
   minimum-SIP / minimum-lump-sum / AUM fact, and a Windows console defaults to cp1252,
   which has no rupee sign. `print(answer.render())` raised and killed the process on exactly
   the questions the demo is built around. Fixed by `_use_utf8_console()`: set the console
   code page to 65001 via `ctypes` and `reconfigure(encoding="utf-8", errors="replace")`
   both streams, degrading to replacement characters rather than crashing if either step is
   refused. Pinned by `test_rupee_sign_renders_in_the_cli`.

2. **The disclaimer was unreachable.** `print(DISCLAIMER)` sat after the REPL loop, but every
   branch of that loop returned first, so interactive mode never showed it. Moved into the
   startup banner. Pinned by `test_interactive_banner_includes_disclaimer`.

Note for P8: the rupee bug was CLI-only. Streamlit renders UTF-8 in the browser, so `app.py`
needs no equivalent workaround — but if you ever print an answer to the *terminal* from inside
the app, call `_use_utf8_console()` first.

---

## P8 — Streamlit UI (`app.py`)

**Goal:** the tiny UI the brief asks for — welcome line, 3 example questions, facts-only note —
plus a retrieval trace for the demo.

**Create `app.py`**
- `st.set_page_config(page_title="HDFC MF Facts Assistant", page_icon="📄", layout="centered")`
- `WELCOME` — names the five schemes and the fact types covered
- `@st.cache_resource def assistant()` → `get_assistant()`, spinner *"Loading embedding model and
  vector index..."*
- `init_state()` — `history`, `show_trace` in `session_state`
- `sidebar()` — Scope (AMC + 5 schemes), the pipeline diagram in a `st.code` block, the embedding
  model caption, `Top-k` caption, the guardrail bullet list, and `DISCLAIMER`
- `render_answer(answer, show_trace)` — switch on `answer.kind`:
  `advice_refusal` → `st.info`; `pii` / `compute_refusal` / `performance_refusal` → `st.warning`;
  everything else → `st.markdown`. Then one `st.link_button("Source", url)` per citation, then
  `st.caption(f"Last updated from sources: {answer.retrieved_at}")` **only if** `retrieved_at` is
  truthy, then the optional `st.expander("Retrieval trace")` showing intent/kind/refused and one
  row per chunk with `rrf`, `dense`, `bm25`, `[kind/field]`, scheme and a 90-char slice.
- `main()` — if `not index_exists()`: `st.error` + the build command, and return. Catch
  `PipelineMissing` the same way. Render `WELCOME` + the 3 example questions when history is empty.
  `st.chat_input`, append to history, spinner *"Searching the corpus..."*. Settings expander with a
  `show_trace` checkbox and a "Clear conversation" button that reruns.
- `if __name__ == "__main__": main()`

**Create `scripts/test_ui.py`** — a `streamlit.testing.v1.AppTest` smoke test: run the app, assert
no exceptions, assert exactly 1 `chat_input`, set the expense-ratio question and run again, assert
no exceptions, assert `1.03%` appears in the markdown/captions, assert the source link button
exists, assert the `Last updated from sources` caption and the facts-only note.

**Tests to add now** — new class `TestUiSource` (source-level assertions, no Streamlit runtime,
so they stay fast in the main suite): `test_app_has_facts_only_note`, `test_app_lists_three_examples`,
`test_app_handles_missing_index`, `test_render_answer_switches_on_kind`.

**Cursor prompt**

> Create the Streamlit UI in `app.py` and a UI smoke test in `scripts/test_ui.py`. Read `PRD.md`
> FR-8 and `architecture.md` §3 (PRESENTATION layer) first. `mf_rag/pipeline.py` exposes
> `get_assistant()`, `EXAMPLE_QUESTIONS`, `PipelineMissing`; `mf_rag/sources.py` exposes
> `AMC_NAME`, `DISCLAIMER`, `SOURCES`; `mf_rag/embed_store.py` exposes `index_exists()`;
> `mf_rag/config.py` exposes `EMBEDDING_MODEL` and `RETRIEVE_FINAL_K`.
> - `app.py` must: set the page config; show the `DISCLAIMER` as a caption under the title; if
>   `index_exists()` is False show `st.error` with the `python scripts/build_index.py` command and
>   return; wrap `get_assistant()` in `@st.cache_resource` with a spinner; render a sidebar with
>   Scope, a `st.code` pipeline diagram, the embedding model, the top-k, the guardrail bullets and
>   the disclaimer; render the welcome line and the 3 `EXAMPLE_QUESTIONS` when the conversation is
>   empty; use `st.chat_input`; keep history in `session_state`.
> - `render_answer` must branch on `answer.kind`: `advice_refusal` -> `st.info`;
>   `pii`, `compute_refusal`, `performance_refusal` -> `st.warning`; else `st.markdown`. Then one
>   `st.link_button("Source", url)` per citation. Then the caption
>   `f"Last updated from sources: {answer.retrieved_at}"` ONLY when `answer.retrieved_at` is
>   truthy — the PII refusal must show no timestamp. Then, when the trace checkbox is on, an
>   `st.expander("Retrieval trace")` listing intent/kind/refused and per chunk the `rrf_score`,
>   `vector_score`, `bm25_score`, `[kind/field_name or heading]`, `scheme_short` and a 90-char
>   slice of `cite_text`.
> - Add a Settings expander with a "Show retrieval trace" checkbox and a "Clear conversation" button
>   that clears history and calls `st.rerun()`.
> - `scripts/test_ui.py` uses `streamlit.testing.v1.AppTest`: run the app, print exceptions, assert
>   one `chat_input` exists, set it to "What is the expense ratio of HDFC Large Cap Fund?", run
>   again, and report whether `1.03%` appears, whether a Source button exists, and whether the
>   `Last updated from sources` caption and the facts-only note are present.
> - Also add 4 fast source-level tests in `tests/test_faq.py` that read `app.py` as text and assert
>   the facts-only note, the 3 examples, the missing-index branch and the `kind` switch exist — so
>   the main suite does not need to boot Streamlit.
> Do not change any module under `mf_rag/` to make the UI work; adapt the UI instead.

**Verification**
```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
.\.venv\Scripts\python.exe scripts\test_ui.py
.\.venv\Scripts\python.exe -m unittest tests.test_faq.TestUiSource -v
```

**Exit gate**
- [x] App loads with no exceptions; expense-ratio question returns `1.03%` with one Source button
- [x] Advice question shows the info box; PII question shows the warning box **with no timestamp**
- [x] Trace checkbox shows per-chunk `rrf` / `dense` / `bm25`
- [x] `TestUiSource` green (7 tests) · suite is **79 tests**
- [x] `streamlit run app.py` serves HTTP 200 (the served HTML is only a shell; Streamlit renders
      client-side over a websocket, so assert on `AppTest`, not on the raw HTML)

**Corrections applied during P8 implementation**

`app.py` already existed and matched this spec, so P8 was mostly verification. Two real problems
were found, both in the *test* layer rather than the UI:

1. **`scripts/test_ui.py` asserted nothing.** It printed seven booleans and exited 0 no matter
   what, so it could never fail and could never catch anything. Rewritten as seven real
   `assert` checks (cold boot, the 3 examples, the fact answer + timestamp, the Source link
   button and its target URL, the opt-in trace, the PII warning with no timestamp, the advice
   info box, and Clear conversation).

2. **The retrieval trace could never appear.** `app.py` rendered the `Settings` expander
   *after* the transcript loop. Streamlit reruns a script top-to-bottom, so on the run where the
   user ticked "Show retrieval trace" the checkbox still held the *previous* value while the
   transcript above it was being drawn — the trace only showed up on the run after the *next*
   question. On a demo this reads as a dead checkbox. Fixed by moving the Settings block above
   the transcript. Pinned by `test_settings_precede_the_transcript`, and the check exists because
   the ordering is invisible to anyone reading the file top-to-bottom.

**FR-8.2 gap found and closed (post-P11 audit).** Auditing `app.py` against PRD FR-8.1–8.7 showed
every item implemented except one: FR-8.2 requires the 3 example questions to be
"**clickable**/suggested", and `app.py` drew them as `st.markdown` bullets — inert text. The word
"clickable" appeared in `PRD.md:142` and nowhere in the code or tests.

`test_app_lists_three_examples` could not catch this: it only asserted that `EXAMPLE_QUESTIONS` in
`pipeline.py` has 3 answerable members, never how `app.py` renders them. A test can pass while the
requirement it appears to cover is unmet, so the fix is two-sided.

Implementation: each example is now a `st.button`, and a click sets `session_state.pending_question`,
which is consumed as if typed. A click **cannot** prefill `st.chat_input` — that widget accepts a
`key` but no `value`, and assigning to its session-state key does not submit it (verified on 1.64:
the value never reaches the return), so the `pending_question` handoff is used instead.
`pending_question` is cleared *before* asking, otherwise a rerun — ticking the trace checkbox, say —
would re-answer the same example.

Two ordering facts worth knowing, both inherent to Streamlit's top-to-bottom rerun:
- The welcome block is drawn *before* the pending question is consumed, so on the click's run the
  example buttons and the new chat message are both on screen. The transcript must contain the
  question exactly once, which is the actual duplicate check.
- "Clear conversation" only appears on the run *after* an answer, because Settings is evaluated
  while history is still empty. Pre-existing, and the reason the smoke test needs an extra `.run()`.

Tests: `test_example_questions_are_clickable_not_plain_text` (source-level, also asserts the old
markdown bullet is gone) and `test_clicked_example_is_answered_and_not_repeated` (runtime, via
`AppTest`). `scripts/test_ui.py` gained an 8th check for the click, on its own `AppTest` instance so
the extra answer does not perturb the "exactly 1 Source link" assertions, and `_text()` now includes
button labels because the examples no longer appear in any markdown or caption. Suite is
**110 tests**; `TestUiSource` is 9.


- `st.link_button` does **not** appear in `at.button`. `at.button` holds only real buttons
  ("Clear conversation"). Use `at.get("link_button")` to find the Source button, and read
  `.label` / `.href` from it. The Source URL appears in no markdown or caption.
- A "Settings" expander is always present, so assert on the expander *label*
  (`"Retrieval trace" in e.label`) rather than on `at.get("expander")` being empty. Likewise
  `st.info(WELCOME)` means an advice refusal makes **two** info boxes, not one — assert the
  refusal text is present rather than counting boxes.

---

## P9 — Evaluation harness

**Goal:** produce the evidence that justifies the chunking and retrieval choices, reproducibly and
offline.

**Create `scripts/compare_chunking.py`**
- For sizes 120 / 180 / 260 / 400 compute `fact_integrity(fixed_size_baseline(doc, size, 40), doc.facts)`
  per document, and the same for the production `chunk_document` path.
- Write `data/chunking_report.json` with the tokenizer name, the hard max, the metric definition,
  and per-strategy `cooccur_ratio` / `chunks_per_doc` / `avg_words_per_chunk` / `example_orphans`.
- **Label the metric honestly in the file:** co-occurrence is a *diagnostic*; it comes out 1.0 for
  every strategy including fixed windows, so it does not discriminate. The retrieval benchmark in
  the next script is the real evidence.

**Create `scripts/benchmark_retrieval.py`**
- 35 labelled queries: 7 `BENCH_FIELDS` × 5 schemes, each phrased without the scheme name plus
  variations, so a query's gold chunk is unambiguous.
- Build a **second, isolated** corpus with `fixed_size_chunks` into a scratch collection at
  `data/chroma_bench` (`shutil.rmtree` first). Leave `data/chroma` untouched.
- Embed both corpora with the same `get_model()`, run **dense-only** top-8 retrieval for each query,
  and score a hit when the gold `field_name` appears in the returned chunk metadata.
- Report top-1, top-3 and top-5 hit rates per strategy, plus per-query rows.
- Write `data/retrieval_benchmark.json`.
- Print a comparison table and state in the output that the benchmark is dense-only and
  small-sample, so live hybrid accuracy is expected to be at least this good but is not measured.

**Exit gate**
- [x] Structure-aware top-1 **far** above fixed-window top-1 — measured **74.3% vs 2.9%**
      (+71.4 points), against the ≈74% vs ≈3% the gate predicted
- [x] `data/retrieval_benchmark.json` has 35 rows per strategy (5 schemes × 7 fields),
      each with top-1/top-3/top-5 for both strategies
- [x] `data/chroma/` is not modified by the benchmark — verified by SHA-256 over the whole
      directory, and pinned by `test_benchmark_never_opens_the_production_store`

> **How to re-verify that gate.** Hash `data/chroma`, run *only* `scripts/benchmark_retrieval.py`,
> hash again. Do not run the unit test suite in between: `TestStore`, `TestRetrieval` and every
> answer test read the production collection, and that alone rewrites the same six files. The
> benchmark is clean; the tests are the thing that still opens the store.
- [x] The report states the dense-only and small-sample caveats in a 138-word `caveats` field,
      printed to stdout as well as stored in the JSON

**Corrections applied during P9 implementation**

Both scripts already existed, so P9 was verification. Four real problems were found:

1. **The benchmark was modifying the shipped index.** It queried the production collection at
   `data/chroma` for the structure-aware leg. ChromaDB **rewrites its SQLite and HNSW files on
   open+query**, so a "read-only" query rewrote all six files in `data/chroma` — caught by
   hashing the directory before and after (the hash moved). Fixed by embedding **both** corpora
   fresh into `data/chroma_bench` as two collections (`bench_struct`, `bench_fixed`), which is
   also what this spec asks for ("Embed both corpora with the same `get_model()`") and makes the
   comparison symmetric. Headline numbers were unchanged (74.3% / 2.9%), confirming the earlier
   result was not an artefact of the asymmetry.
2. **No top-5 hit rate.** Retrieval depth was hard-coded to 3, so the required top-1/3/5 report
   was impossible. Now `TOP_K = 8` candidates per query, reported at depths 1, 3 and 5.
3. **The report asserted the caveats it was supposed to make.** Neither the JSON nor stdout
   contained the dense-only / small-sample caveat — only the module docstring did, and the spec
   says "in words, not just in a comment". Now a 138-word `caveats` field, printed as well.
4. **Cleanup crashed the script.** `shutil.rmtree(data/chroma_bench)` raised
   `PermissionError: [WinError 32]` because Chroma holds the files open, so the benchmark exited
   non-zero *after* printing correct results, and left the scratch directory behind. Now
   `_rmtree_quietly()` retries and degrades to a printed note; the directory is gitignored and
   rebuilt on every run. Chroma clients are also cached now instead of re-opened 35 times.

**On the chunking report's headline metric.** `cooccur_ratio` is **1.0 for every strategy**,
including all four fixed-window sizes, because every labelled fact on these five pages is short
enough that a window usually still swallows the whole line. A reader could easily mistake that
1.0 for a perfect score and cite it as proof the chunking is great. The file now says so in
`metric_is_diagnostic_only`, and points at the retrieval benchmark as the real evidence.
`test_chunking_report_confirms_every_strategy_scores_one` keeps both facts pinned together.

**Interesting result, reported honestly:** fixed-size *top-5* recall (74.3%) exactly equals
structure-aware *top-1* accuracy (74.3%). A fixed window does eventually contain the right fact —
it simply ranks it below many neighbours. The argument for structure-aware chunking is therefore
about **precision at rank 1**, which is what the assistant actually consumes, not about recall.
Do not quote the top-5 column as if it supported the design.

**Cursor prompt**

> Create the two evaluation scripts. Read `architecture.md` §4.2 and §9 first.
> - `scripts/compare_chunking.py`: for window sizes 120, 180, 260, 400 compute
>   `fact_integrity(fixed_size_baseline(doc, size, 40), doc.facts)` for every document, and the same
>   stats for the production `chunk_document(doc, count_tokens_minilm)` path. Write
>   `data/chunking_report.json` containing the tokenizer name, `CHUNK_TOKEN_HARD_MAX`, an explicit
>   `metric` string, and per-strategy `cooccur_ratio`, `chunks_per_doc`, `avg_words_per_chunk` and
>   up to 5 `example_orphans`. Label the metric in the JSON as a diagnostic only, because
>   co-occurrence comes out 1.0 for every strategy including fixed windows and therefore does not
>   discriminate between them.
> - `scripts/benchmark_retrieval.py`: 35 labelled queries — the 7 fields `expense_ratio, exit_load,
>   min_sip, min_lumpsum, benchmark, riskometer, aum` across all 5 schemes, phrased as natural
>   questions. Build a SECOND corpus with `fixed_size_chunks(doc, 180, 40)` and embed it into a
>   scratch Chroma collection at `data/chroma_bench` (delete that directory first with
>   `shutil.rmtree`). Never touch `data/chroma`. Embed both corpora with the same `get_model()`,
>   run dense-only top-8 retrieval for each query, and count a hit when the gold `field_name` is
>   present in the returned chunk metadata. Report top-1, top-3 and top-5 hit rates per strategy,
>   write per-query rows to `data/retrieval_benchmark.json`, and print a comparison table. The
>   script's own output must state that the benchmark is dense-only and small-sample, so it
>   measures the chunking difference and not the shipped hybrid configuration.
> Verify: run both and report the two hit-rate columns. Do not modify `data/chroma` or any module
> under `mf_rag/`.

**Verification**
```powershell
.\.venv\Scripts\python.exe scripts\compare_chunking.py
.\.venv\Scripts\python.exe scripts\benchmark_retrieval.py
```

---

## P10 — Deliverables

**Goal:** regenerate the four submission artefacts from code, so they can never drift from the
running system.

**Create `scripts/make_sources.py`**
- Write `deliverables/SOURCES.csv` with columns `key, scheme_name, scheme_short, category, plan,
  page_name_on_site, url, publisher, access_type, content_type`, all derived from `SOURCES`.
- Write `deliverables/SOURCES.md` — a table of the 5 schemes, a "Secondary / educational links"
  section (help centre, mutual funds index, SEBI) marked *cited only for guidance answers, never
  for a fund fact*, a "Not used" section (no back-end screenshots, no third-party blogs, no PII
  requested/received/stored), and the disclaimer footer.

**Create `scripts/make_sample_qa.py`**
- 10 fixed `(question, topic)` pairs covering: 5 fact questions across different schemes and
  fields, ELSS lock-in, fund manager, statement guidance, an advice refusal, and a PII refusal.
- For each, call the **real** `get_assistant().ask(q)` and record question, topic, answer,
  `primary_citation`, `last_updated`, intent, kind, refused.
- Write `deliverables/SAMPLE_QA.json` and a `deliverables/SAMPLE_QA.md` with a header stating the
  corpus refresh timestamp, the disclaimer, a note that answers are extractive, and per-question
  sections showing Topic / Answer / Source / Last updated / Route.

**Create `deliverables/DISCLAIMER.md`** — the exact text of `DISCLAIMER` from `sources.py`, plus
where it appears in the UI.

**Write `README.md`** — quick start (3 commands), refresh instructions, test suite, CLI usage, the
pipeline diagram, per-stage explanation, the chunking comparison table **with its two caveats**, the
guardrail table, the source table, the project layout, and known limitations.

**Cursor prompt**

> Create the deliverable generators and the deliverables. Read `PRD.md` §11 and `architecture.md`
> §6.
> - `scripts/make_sources.py` writes `deliverables/SOURCES.csv` (columns `key, scheme_name,
>   scheme_short, category, plan, page_name_on_site, url, publisher, access_type, content_type`,
>   all derived from the `SOURCES` registry) and `deliverables/SOURCES.md` containing a 5-row
>   scheme table, a "Secondary / educational links" section that states these are cited ONLY for
>   guidance answers and never for a fund fact, a "Not used" section (no back-end screenshots, no
>   third-party blogs or aggregators as factual sources, no PII requested/received/stored), and the
>   disclaimer as a footer.
> - `scripts/make_sample_qa.py` runs 10 fixed (question, topic) pairs through the REAL assistant via
>   `get_assistant().ask(q)` — 5 fact questions across different schemes and fields, an ELSS
>   lock-in question, a fund-manager question, a statement-guidance question, an advice refusal
>   ("Should I buy HDFC Small Cap Fund?") and a PII refusal ("My PAN is ABCDE1234F, check my
>   returns"). Record question, topic, answer text, `primary_citation`, `last_updated`, intent,
>   kind and refused. Write `deliverables/SAMPLE_QA.json` and `deliverables/SAMPLE_QA.md`; the
>   Markdown must open with the corpus refresh timestamp, the disclaimer, and a note that answers
>   are extractive and each carries exactly one source link. Per question show Topic, Answer,
>   Source, Last updated from sources, and a Route line with `intent` / `kind` / `refused`. The PII
>   row must show `n/a (no source consulted)` for the timestamp.
> - Create `deliverables/DISCLAIMER.md` containing the exact `DISCLAIMER` string from
>   `mf_rag/sources.py` and a note on where it is displayed in the UI.
> - Write `README.md`: quick start with the 3 commands, `--refresh` usage, the test command and
>   test count, CLI usage, the 6-stage pipeline diagram, a paragraph per stage, the chunking
>   comparison table INCLUDING both honest caveats (the first metric tried was uninformative at
>   1.0 for every strategy; the benchmark is dense-only), the guardrail table, the source table
>   with the Flexi Cap naming quirk and the two-AUM-figure quirk, the project layout tree, and the
>   known-limitations list.
> - Run both generators and paste the resulting SOURCES.md and SAMPLE_QA.md so they can be checked.
> Never hand-write an answer in SAMPLE_QA.md — every answer must come from the running assistant.

**Exit gate**
- [x] `SAMPLE_QA.md` has exactly 10 entries, every answer traceable to the assistant —
      `TestDeliverables` re-asks all 10 and requires an exact match on answer, citation,
      kind, intent, refused and timestamp
- [x] Every fund fact cites its scheme page; the guidance answer cites only the help centre
- [x] The PII row has no timestamp (`n/a (no source consulted)`) and cites the privacy policy
- [x] `README.md` contains both chunking caveats, and its benchmark table is asserted to match
      `data/retrieval_benchmark.json` exactly

**Corrections applied during P10 implementation**

All four artefacts already existed, so P10 was verification. Three real problems were found:

1. **The README's benchmark table had rotted.** It claimed 24 fixed chunks, 2.86% top-1 and
   22.86% top-3, and had no top-5 column. The regenerated report says **26** chunks, 2.86% top-1,
   14.29% top-3, 74.29% top-5. The table also mislabelled the baseline as "180-**token**"
   windows when `fixed_size_baseline` splits on `text.split()`, i.e. **180 words**. Corrected,
   with the top-5 column added, and `test_readme_benchmark_table_matches_the_report` now builds
   the expected row strings from the JSON so a stale README fails the suite instead of
   misleading a reader.
2. **`make_sources.py` hard-coded three educational URLs** that duplicate `EDUCATIONAL_LINKS`, so
   `SOURCES.md` could drift from the links the assistant actually cites — the exact failure mode
   P10 exists to prevent. Now derived from the registry.
3. **The ELSS link was listed as a "guidance-only" source while being a primary source.**
   `EDUCATIONAL_LINKS["elss"]` is the HDFC ELSS scheme page — the same URL as table row 3 — yet it
   was printed under "cited only for guidance answers (never for a fund fact)". The generator now
   separates any registered link that is also a scheme page and labels it primary, so the
   document cannot contradict itself. Pinned by
   `test_sources_md_lists_every_scheme_and_no_orphan_urls`.

**On the PAN in the deliverable.** `SAMPLE_QA` row 10 must show the PII refusal, and this spec
fixes the question as `"My PAN is ABCDE1234F, check my returns"`, so a PAN-shaped string appears
in the question heading. That is intended and harmless (it is the standard dummy PAN). The test
that matters is `test_pii_identifier_never_reaches_an_answer`: the identifier must not appear in
any *answer*, which proves the refusal scrubs rather than echoes it.

**New class `TestDeliverables`** (12 tests) turns every exit-gate item above into an assertion,
so "regenerated from code" is enforced rather than merely intended. Suite is now **101 tests** after
P11 added two provenance regressions.

---

## P11 — Hardening and full-suite sign-off

**Goal:** close the known gaps, then prove the whole thing holds together.

**Task 1 — remove dead config (I-8)**
`config.FACTS_PATH` and `config.MIN_HYBRID_SCORE` are declared and never read. Delete them, and
change `FAQAssistant.__init__(self, retriever, top_k: int = 6)` to use
`MAX_CONTEXT_CHUNKS` from config, adding that constant. Then re-grep: every name in `config.py`
must appear in at least one other file.

**Task 2 — multi-worker safety note**
`_MODEL`, `_CLIENT` and `_retriever` are module globals. Correct for Streamlit's single process,
wrong for multi-worker serving. Add a one-line comment at each declaration saying so. Do **not**
refactor to a process pool in this phase.

**Task 3 — full suite + UI + CLI sign-off**
```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -t .   # 101 tests, OK
.\.venv\Scripts\python.exe scripts\test_ui.py
.\.venv\Scripts\python.exe scripts\ask.py "What is the exit load of HDFC ELSS Tax Saver Fund?"
.\.venv\Scripts\python.exe scripts\ask.py "Should I buy HDFC Small Cap Fund?"
.\.venv\Scripts\python.exe scripts\ask.py "My PAN is ABCDE1234F, check my returns"
.\.venv\Scripts\python.exe scripts\ask.py "What is the weather in Mumbai"
.\.venv\Scripts\python.exe scripts\build_index.py               # idempotency re-check
```

**Task 4 — determinism check**
Run the CLI on the same 5 questions twice. Answers must be byte-identical. Any difference means
something non-deterministic leaked in (dict ordering, a timestamp, an unseeded shuffle).

**Task 5 — offline check**
Disconnect the network, then run `build_index.py` (no `--refresh`) and 5 CLI questions. Both must
succeed using only `data/raw/*.html` and `data/chroma/`.

**Exit gate — Definition of Done**
- [x] 101 tests, `OK`
- [x] No unused constant in `config.py`
- [x] `streamlit run app.py` serves with no exceptions
- [x] Same question twice → identical answer
- [x] Index rebuild and 5 questions work with the network off
- [x] All 5 deliverables present and regenerable by their scripts
- [x] No stray temp files; `data/chroma_bench` gitignored

### P11 results

**Task 1 — dead config removed.** `FACTS_PATH` and `MIN_HYBRID_SCORE` deleted. `MAX_CONTEXT_CHUNKS`
already existed in `config.py` but was never read, so the instruction to "add" it was stale — the
fix was to *use* it. `FAQAssistant.__init__` now defaults to `top_k=MAX_CONTEXT_CHUNKS` instead of
a bare literal `6`. Re-grep confirms all 22 remaining `config.py` names are referenced elsewhere.

**Task 2 — multi-worker notes added** at `_MODEL`, `_CLIENT` (`embed_store.py`) and `_retriever`
(`retrieve.py`).

**Task 4 — determinism.** All 5 questions byte-identical across two separate processes
(ELSS exit load, Small Cap advice refusal, PII, unanswerable weather, Flexi Cap expense ratio).

**Task 5 — offline.** Rather than toggling the machine's adapter, egress was blocked at the socket
layer inside the process (`socket.connect` / `create_connection` / `getaddrinfo` raise, plus
`HF_HUB_OFFLINE=1`). Patching the *connect methods* rather than the `socket` class matters:
`ssl.SSLSocket` subclasses `socket.socket`, so replacing the class breaks the import outright.
Under that block, `build_index.py` exited 0 and all 5 questions answered correctly.

#### Defect found and fixed: `retrieved_at` was a fabricated freshness claim

The idempotency re-check in Task 3 failed, and the cause was a real provenance bug rather than a
cosmetic one. `load_documents` read cached pages from `data/raw/*.html` but `parse_document`
stamped every document with `now_iso()` — the time of the *rebuild*.

Consequences, all of them wrong:
- The UI's `Last updated from sources:` caption (the product's only currency signal) claimed pages
  had just been verified whenever the app was rebuilt or restarted. A page fetched at 08:12 was
  presented as verified at 11:02. The number was a fabrication, not a measurement.
- `build_index.py` was not idempotent: `chunks.jsonl` changed on every run purely by clock.
- Every offline run would have reported the most recent possible freshness for the stalest data.

Fixed by adding `ingest.fetched_at(path)`, which derives the stamp from the raw file's mtime — the
honest record of when the page was actually fetched — and threading it through
`parse_document(..., retrieved_at=...)`. Network fetches still use `now_iso()`, correctly, because
they *are* happening now. A cache-only rebuild is now byte-identical: `chunks.jsonl` hashes to
`A7A7178C…` on every run.

The drift was caught by the pre-existing `test_every_sample_answer_is_traceable_to_the_assistant`
(9 failures comparing generated `SAMPLE_QA` against the live assistant), which is the traceability
check doing exactly its job. Deliverables were regenerated. Two regression tests added:
`test_cached_pages_keep_their_real_fetch_time` and `test_build_is_idempotent` — suite is now
**101 tests**.

**Benchmark isolation re-verified:** `scripts/benchmark_retrieval.py` leaves both `chunks.jsonl`
and every file under `data/chroma/` byte-identical. Structure-aware top-1 74.29%, top-3/top-5
94.29% over 35 labelled queries at top-8 candidates.

---

## Appendix A — Traceability

| Phase | PRD requirement | Architecture section | Tests added |
|---|---|---|---|
| P0 | FR-1.1 (partial), §5 scope | §2, §3, §7 | — |
| P1 | FR-1.1 – FR-1.5 | §4.1 | `TestIngest` (7, incl. 2 added in P11) |
| P2 | FR-2.1 – FR-2.5 | §4.2 | `TestChunking` (6) |
| P3 | FR-3.1 – FR-3.3, FR-4.1 – FR-4.3 | §4.3, §4.4 | `TestStore` (6) |
| P4 | FR-5.1 – FR-5.5 | §4.5 | `TestRetrieval` (7) |
| P5 | FR-7 (PII rung), NFR privacy | §10 | `TestPiiGuard` (4) |
| P6 | FR-6.1 – FR-6.5, FR-7 (rungs 2–4) | §4.6, §5 | `TestRouting` (5), `TestAnswers` (17) |
| P7 | FR-6.5 (stamp), pipeline wiring | §3 | `TestPipeline` (9) |
| P8 | FR-8.1 – FR-8.7 | §3 | `TestUiSource` (7) |
| P9 | G6 (architecture demonstrable) | §4.2, §9 | `TestEvaluationHarness` (8) |
| P10 | §11 deliverables, AC-15 – AC-17 | §6 | `TestDeliverables` (12) |
| P11 | I-8, §11 debt items 1–2 | §7, §11 | provenance + idempotency regressions, in `TestIngest` |

---

## Addendum — evaluation-only LLM tooling (post-P11)

An LLM client was added for **evaluation only**, never for answering. `mf_rag/llm.py` reads `.env`
and supports OpenAI-compatible (OpenAI, Groq, Azure, LM Studio, Ollama), Anthropic and Gemini. It
is deliberately the only module that touches credentials.

**D1 is unchanged.** `answer.py`, `pipeline.py`, `retrieve.py`, `ingest.py` and `chunking.py` do not
import it, and `TestParaphraseHarness::test_answer_path_does_not_import_the_llm_module` fails if any
of them ever does. That test is the enforcement mechanism for the decision, not a comment about it.

**What it is for.** All 35 benchmark queries come from one template, so the headline retrieval number
was measured on a single phrasing style. `scripts/gen_paraphrases.py` produces rewordings of the same
35 gold labels; `scripts/benchmark_paraphrase.py` scores them through the **shipped hybrid retriever**,
which also closes the "the hybrid configuration has not been scored" gap the dense-only benchmark's own
caveat text admitted to.

Gold labels never move, so any change in hit rate is retrieval failing on wording, not a relabelling
artefact. Results are committed to `data/paraphrase_benchmark.json`; the cache in
`data/paraphrase_queries.json` is committed too, so the evaluation needs no key and no network.
`--seed` regenerates it deterministically with no key; the LLM path overwrites it and records
`generated_by` so the two are distinguishable.

**Finding, stated carefully.** Templated phrasing scores 94.29% top-1, but conversational phrasing
scores 17.14%. That gap is *not* mostly a retrieval defect and must not be quoted as one: the
`colloquial` and `verbose` seeds deliberately omit the field's own vocabulary, so they test implicit
intent mapping ("penalty for redeeming early" → exit load) rather than rewording, which no lexical or
dense retriever can do from surface tokens. The artifact's own `caveats` field and README both say so,
and `TestParaphraseHarness::test_report_carries_the_interpretive_caveat` fails if that warning is
removed. The defensible reading, from the miss breakdown: **scheme resolution is essentially solved**
(0 wrong schemes in five of six styles), and the real weakness is **field disambiguation** — picking
`expense_ratio` over `exit_load` when the query drops the field's words, because the field label is the
only thing separating those chunks. Fixing that means changing chunking or adding a query-expansion
step, both of which alter the retrieval design and are out of scope here.

Suite is now **110 tests** (101 + 7 paraphrase-harness + 2 clickable-example, see the FR-8.2 note
in the P8 section above).

---

## Appendix B — Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `PipelineMissing: No index found` | P3 never ran, or `data/chroma` was deleted | `python scripts\build_index.py` |
| `test_answers_are_grounded_in_retrieved_chunks` fails | someone reformatted a value in `_fact_sentence` | quote `cite_text` verbatim; never re-parse or re-format the number |
| An intent routes to a neighbouring field | a broad `INTENTS` pattern was inserted above a specific one | restore the ordering; `exit_load_history` before `exit_load`, `sub_category` before `category` |
| `PII is refused but the question was innocent` | a regex is too greedy — usually phone or Aadhaar | add digit lookbehind/lookahead; check against the `TestPiiGuard` negative list |
| Dense top-1 collapses after a chunking change | tokenizer fell back to word counting | `transformers` not importable; check `count_tokens_minilm` did not cache `False` |
| AUM answer shows `9,86,236.84 Cr` | the `total_aum` field leaked into the `aum` intent | `aum` must map only to `aum`; `test_aum_reports_scheme_size_not_amc_total` |
| Section-filtered search returns nothing | the fact moved to a different `section` string | section names are a closed set; keep `SECTION_FOR_INTENT` and the ingest section strings in sync |
| Benchmark numbers move between runs | the benchmark re-fetched, or the scratch store was not cleared | `shutil.rmtree(data/chroma_bench)` first; never pass `--refresh` |
| Streamlit is slow to start | model reloading per rerun | confirm `@st.cache_resource` wraps `get_assistant` and `_MODEL` is a module global |

## Appendix C — Cursor prompt library

Reusable instructions for work that spans phases.

**Add a new scheme**
> Append a `Source` to `SOURCES` in `mf_rag/sources.py` with `key`, `scheme_name`,
> `scheme_short`, `category`, `plan`, `url` and `tags`. Do not change any other module. Re-run
> `scripts\build_index.py` and confirm the new scheme's chunks appear, the chunk count rises by
> roughly 33, and `python scripts\ask.py "expense ratio of <new scheme>"` cites the new URL. Then
> regenerate `deliverables\SOURCES.*` with `scripts\make_sources.py`.

**Add a new field**
> 1. `mf_rag/ingest.py`: add the label to `LABEL_FIELDS` (or the relevant `contact_fields` map) and
>    a `_parse_*` rule that pairs it with the following line, appending a `Fact` with a `section`.
> 2. `mf_rag/answer.py`: add an `INTENTS` entry with a generous pattern and the field in its
>    `fields` tuple, placed **above** any broader pattern that would swallow it; add a
>    `FIELD_QUERY_HINT` entry.
> 3. Add a test asserting the field is extracted on all 5 pages and that the question returns it
>    with the correct citation.
> No change is needed in chunking, embedding, storage or retrieval — they are field-agnostic.

**Add a new refusal class**
> 1. `mf_rag/answer.py`: add the pattern and the refusal template with a `{link}` slot.
> 2. Insert the check into `FAQAssistant.ask` at the correct rung — PII first, then advice, then
>    performance, then projection.
> 3. Return `Answer(..., kind="<new>_refusal", refused=True, citations=[link], retrieved_at=self._stamp_for(link))`.
> 4. Add the new kind to `render_answer`'s `st.warning` branch in `app.py`.
> 5. Add a test that the class is refused, carries exactly one citation, and still has a timestamp.

**Change a retrieval parameter**
> Edit only `mf_rag/config.py`. Then re-run `scripts\benchmark_retrieval.py` and report whether
> dense top-1/top-3 moved. Note in `README.md` if the shipped configuration now differs from the
> benchmarked one. Never inline a numeric literal into `retrieve.py` or `answer.py`.
