"""Test suite for the HDFC facts-only FAQ assistant.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import csv
import json
import re
import sys
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mf_rag import memory
from mf_rag.config import MEMORY_TURNS
from mf_rag.ingest import load_documents
from mf_rag.answer import FAQAssistant, ADVICE_RE, Answer, detect_intent, detect_scheme, strip_scheme_names
from mf_rag.embed_store import load_chunks
from mf_rag.pipeline import EXAMPLE_QUESTIONS, PipelineMissing, get_assistant
from mf_rag.pipeline import ask as pipeline_ask
from mf_rag.pii import scan
from mf_rag.retrieve import HybridRetriever
from mf_rag.sources import SOURCES, SOURCES_BY_KEY, DISCLAIMER, EDUCATIONAL_LINKS

SOURCE_URLS = {s.url for s in SOURCES}
SHORTS = {s.scheme_short for s in SOURCES}

EXPECTED_FIELDS = (
    "category", "sub_category", "riskometer", "nav", "aum", "expense_ratio",
    "rating", "min_sip", "min_lumpsum", "exit_load", "stamp_duty",
    "tax_implication", "benchmark", "objective", "launch_date", "fund_house",
    "custodian", "rta",
)

ASSISTANT: FAQAssistant | None = None


def assistant() -> FAQAssistant:
    global ASSISTANT
    if ASSISTANT is None:
        ASSISTANT = FAQAssistant(HybridRetriever(load_chunks()))
    return ASSISTANT


def sentences(text: str) -> int:
    return len([p for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p.strip()])


class TestIngest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.documents = load_documents()

    def test_five_sources_loaded(self) -> None:
        self.assertEqual(len(self.documents), 5)

    def test_every_source_has_all_core_fields(self) -> None:
        for document in self.documents:
            present = {f.field_name for f in document.facts}
            missing = [name for name in EXPECTED_FIELDS if name not in present]
            self.assertEqual(missing, [], f"{document.scheme_name} missing {missing}")

    def test_elss_has_lock_in_and_nil_exit_load(self) -> None:
        document = next(d for d in self.documents if d.source.key == "hdfc_elss")
        self.assertIn("3Y", document.value("lock_in"))
        self.assertEqual(document.value("exit_load").strip().lower(), "nil")

    def test_content_window_excludes_nav_and_footer_boilerplate(self) -> None:
        for document in self.documents:
            self.assertNotIn("Download the App", document.lines)
            self.assertNotIn("Invest in Stocks", document.lines)
            self.assertNotIn("Top Gainers Stocks", document.lines)

    def test_conflicts_prefer_visible_page(self) -> None:
        document = next(d for d in self.documents if d.source.key == "hdfc_large_cap")
        self.assertTrue(any("riskometer" in c for c in document.conflicts))
        self.assertEqual(document.value("riskometer"), "Very High Risk")

    def test_cached_pages_keep_their_real_fetch_time(self) -> None:
        """A page read from data/raw/ was NOT retrieved at build time.

        Stamping it with now() made the assistant tell the user a source had just been
        verified when it was read off disk, and made build_index.py non-idempotent.
        """
        from mf_rag.ingest import fetched_at

        newest_raw = max(
            (path.stat().st_mtime for path in (ROOT / "data" / "raw").glob("*.html")), default=0
        )
        for document in self.documents:
            with self.subTest(key=document.source.key):
                self.assertTrue(document.retrieved_at, "every document needs a retrieval stamp")
                self.assertEqual(
                    document.retrieved_at,
                    fetched_at(ROOT / "data" / "raw" / f"{document.source.key}.html"),
                    "stamp must be the raw file's mtime, not the time of this test run",
                )
        self.assertLess(
            max(datetime.fromisoformat(d.retrieved_at).timestamp() for d in self.documents),
            newest_raw + 1,
            "a cached page was stamped later than the file it came from",
        )

    def test_build_is_idempotent(self) -> None:
        """Two cache-only loads must produce identical retrieval stamps."""
        from mf_rag.ingest import load_documents

        first = {d.source.key: d.retrieved_at for d in load_documents()}
        second = {d.source.key: d.retrieved_at for d in load_documents()}
        self.assertEqual(first, second)


class TestChunking(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from mf_rag.chunking import chunk_corpus, count_tokens_minilm
        from mf_rag.ingest import load_documents

        cls.documents = load_documents()
        cls.chunks = chunk_corpus(cls.documents, count_tokens_minilm)
        cls.by_id = {c.chunk_id: c for c in cls.chunks}

    def test_fact_chunks_are_atomic(self) -> None:
        facts = [c for c in self.chunks if c.kind == "fact"]
        self.assertTrue(facts)
        for chunk in facts:
            match = next(
                (
                    f
                    for d in self.documents
                    for f in d.facts
                    if f.field_name == chunk.field_name
                    and f.label == chunk.heading
                    and f.value
                    and f.value in chunk.cite_text
                ),
                None,
            )
            self.assertIsNotNone(
                match, f"fact chunk dropped its value: {chunk.cite_text[:80]}"
            )
            # Atomic means the cite text is exactly this one label/value pair, so a
            # retrieved fact can never drag a second field's number along with it.
            expected = f"{match.label}: {match.value}"
            if match.as_of:
                expected += f" (as on {match.as_of})"
            self.assertEqual(chunk.cite_text, expected)

    def test_no_chunk_exceeds_hard_max(self) -> None:
        from mf_rag.chunking import CHUNK_TOKEN_HARD_MAX

        offenders = [c for c in self.chunks if c.token_count > CHUNK_TOKEN_HARD_MAX]
        self.assertEqual(
            offenders,
            [],
            "chunks over the hard max: "
            + "; ".join(
                f"{c.token_count}tok [{c.section}] {c.heading[:40]}" for c in offenders
            ),
        )

    def test_cite_text_appears_in_source(self) -> None:
        for chunk in self.chunks:
            if chunk.kind != "fact":
                continue
            self.assertNotIn(" (continued)", chunk.cite_text)
            self.assertTrue(chunk.cite_text.rstrip("."), chunk.cite_text)

    def test_prose_chunks_carry_heading(self) -> None:
        prose = [c for c in self.chunks if c.kind == "prose"]
        self.assertTrue(prose)
        for chunk in prose:
            self.assertTrue(
                chunk.text.startswith(f"{chunk.scheme_name} ({chunk.category}). "),
                chunk.text[:90],
            )
            self.assertIn(chunk.heading, chunk.text)

    def test_chunk_ids_are_stable(self) -> None:
        from mf_rag.chunking import chunk_corpus, count_tokens_minilm

        again = chunk_corpus(load_documents(), count_tokens_minilm)
        self.assertEqual([c.chunk_id for c in again], [c.chunk_id for c in self.chunks])

    def test_fixed_baseline_is_marked(self) -> None:
        from mf_rag.chunking import fixed_size_chunks

        document = self.documents[0]
        fixed = fixed_size_chunks(document, size=180, overlap=40)
        self.assertTrue(fixed)
        for chunk in fixed:
            self.assertEqual(chunk.kind, "fixed")
            self.assertEqual(chunk.section, "mixed")
            self.assertEqual(chunk.heading, "mixed window")
        # The baseline must never leak into the production corpus.
        self.assertNotIn("fixed", {c.kind for c in self.chunks})


class TestStore(unittest.TestCase):
    """Stage 3/4: embedding + persistent vector store. Needs a built index."""

    @classmethod
    def setUpClass(cls) -> None:
        from mf_rag.config import CHUNKS_PATH

        if not CHUNKS_PATH.exists():
            raise unittest.SkipTest(
                "no index built - run: python scripts/build_index.py"
            )
        cls.chunks = load_chunks()

    def test_index_has_all_chunks(self) -> None:
        from mf_rag.embed_store import get_collection

        self.assertTrue(self.chunks)
        self.assertEqual(get_collection().count(), len(self.chunks))

    def test_embedding_dim_is_384(self) -> None:
        from mf_rag.config import EMBEDDING_DIM
        from mf_rag.embed_store import embed_texts

        self.assertEqual(len(embed_texts(["dimension probe"])[0]), EMBEDDING_DIM)

    def test_vectors_are_normalised(self) -> None:
        import math

        from mf_rag.embed_store import embed_texts

        for vector in embed_texts(["normalisation probe", "second probe"]):
            norm = math.sqrt(sum(component * component for component in vector))
            self.assertAlmostEqual(norm, 1.0, places=5)

    def test_metadata_carries_citation_fields(self) -> None:
        from mf_rag.embed_store import get_collection

        stored = get_collection().get(include=["metadatas"])["metadatas"]
        self.assertEqual(len(stored), len(self.chunks))
        for metadata in stored:
            for field in ("source_url", "retrieved_at", "chunk_id", "cite_text", "scheme_short"):
                self.assertTrue(metadata.get(field), f"missing {field} in {metadata}")
            self.assertIn(metadata["source_url"], SOURCE_URLS)

    def test_upsert_is_idempotent(self) -> None:
        from mf_rag.embed_store import get_collection, upsert_chunks

        before = get_collection().count()
        upsert_chunks(self.chunks)
        self.assertEqual(get_collection().count(), before)
        self.assertEqual(get_collection().count(), len(self.chunks))

    def test_collection_uses_cosine(self) -> None:
        from mf_rag.config import CHROMA_COLLECTION, EMBEDDING_MODEL
        from mf_rag.embed_store import get_collection

        metadata = get_collection().metadata or {}
        self.assertEqual(CHROMA_COLLECTION, "hdfc_mutual_funds")
        self.assertEqual(metadata.get("hnsw:space"), "cosine")
        self.assertEqual(metadata.get("embed_model"), EMBEDDING_MODEL)

    def test_index_exists_true_after_build(self) -> None:
        from mf_rag.embed_store import index_exists

        self.assertTrue(index_exists())


class TestRetrieval(unittest.TestCase):
    """Stage 5: hybrid dense + BM25 retrieval fused with RRF."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.retriever = HybridRetriever(load_chunks())
        cls.results = cls.retriever.search("expense ratio of HDFC ELSS Tax Saver Fund")

    def test_dense_and_bm25_both_contribute(self) -> None:
        self.assertTrue(self.results)
        self.assertTrue(
            any(c.vector_rank is not None for c in self.results),
            "no result came from the dense list - hybrid degraded to BM25 only",
        )
        self.assertTrue(
            any(c.bm25_rank is not None for c in self.results),
            "no result came from the BM25 list - hybrid degraded to dense only",
        )

    def test_scheme_boost_prefers_named_scheme(self) -> None:
        for scheme, query in [
            ("ELSS", "exit load of HDFC ELSS Tax Saver Fund"),
            ("Small Cap", "minimum SIP for HDFC Small Cap Fund"),
            ("Flexi Cap", "benchmark of HDFC Flexi Cap"),
            ("Large Cap", "riskometer of HDFC Large Cap Fund"),
            ("Balanced Advantage", "AUM of HDFC Balanced Advantage Fund"),
        ]:
            with self.subTest(scheme=scheme):
                top = self.retriever.search(query)[0]
                self.assertEqual(top.scheme_short, scheme)

    def test_rrf_matches_manual_computation(self) -> None:
        from mf_rag.config import RRF_K

        query = "minimum SIP for HDFC Small Cap Fund"
        hits = self.retriever.search(query, top_k=8)
        self.assertTrue(hits)
        for chunk in hits:
            expected = 0.0
            if chunk.vector_rank is not None:
                expected += 1.0 / (RRF_K + chunk.vector_rank)
            if chunk.bm25_rank is not None:
                expected += 1.0 / (RRF_K + chunk.bm25_rank)
            # Boosts are derived from the recorded reasons, so this asserts the
            # score is exactly RRF + the boosts the retriever claims to have applied.
            boost = 1.0
            if any(r.startswith("scheme match") for r in chunk.reasons):
                boost *= 1.35
            if any(r.startswith("field boost") for r in chunk.reasons):
                boost *= 1.05
            self.assertAlmostEqual(
                chunk.rrf_score, expected * boost, places=9, msg=f"reasons={chunk.reasons}"
            )

    def test_section_filter_restricts_results(self) -> None:
        hits = self.retriever.search("largest holdings by weight", section="Holdings")
        self.assertTrue(hits)
        for chunk in hits:
            self.assertEqual(chunk.section, "Holdings")

    def test_section_filter_on_empty_section(self) -> None:
        self.assertEqual(self.retriever.search("expense ratio", section="Nonexistent"), [])

    def test_scheme_alias_resolution(self) -> None:
        from mf_rag.answer import detect_scheme as answer_detect
        from mf_rag.retrieve import detect_scheme

        cases = {
            "what is the 80C benefit": "ELSS",
            "tell me about the tax saver fund": "ELSS",
            "expense ratio of my bluechip": "Large Cap",
            "ter for the flexicap": "Flexi Cap",
            "nav of hdfc equity fund": "Flexi Cap",
            "aum of HDFC Balanced Advantage Fund": "Balanced Advantage",
            "what is the weather in mumbai": None,
        }
        for query, expected in cases.items():
            with self.subTest(query=query):
                # The answer stage and the retriever must never disagree.
                self.assertEqual(detect_scheme(query), expected)
                self.assertEqual(answer_detect(query), expected)

    def test_top_k_is_respected(self) -> None:
        for k in (1, 3, 4, 6):
            with self.subTest(k=k):
                self.assertLessEqual(
                    len(self.retriever.search("expense ratio of HDFC Large Cap", top_k=k)), k
                )

    def test_results_are_sorted_by_rrf_desc(self) -> None:
        scores = [c.rrf_score for c in self.retriever.search("exit load", top_k=8)]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_reasons_explain_each_hit(self) -> None:
        for chunk in self.results:
            self.assertTrue(chunk.reasons, f"no retrieval reason recorded for {chunk.chunk_id}")
            self.assertTrue(chunk.source_url in SOURCE_URLS)


class TestPiiGuard(unittest.TestCase):
    POSITIVES = [
        ("my pan is ABCDE1234F", "PAN"),
        ("aadhaar 2345 6789 0123", "Aadhaar"),
        ("aadhaar 234567890123", "Aadhaar"),
        ("mail me at a.b@example.com", "email address"),
        ("call 9876543210", "phone number"),
        ("call +91 98765 43210", "phone number"),
        ("what is the otp", "OTP"),
        ("my folio number is 123", "account or folio number"),
        ("my demat number 12345678", "account or folio number"),
        ("IFSC HDFC0001234", "IFSC"),
        ("card 4111 1111 1111 1111", "card number"),
        ("passport A2096457", "passport number"),
    ]

    NEGATIVES = [
        # The eight question types named in the brief.
        "What is the expense ratio of HDFC Large Cap Fund?",
        "Is there a lock-in period on HDFC ELSS Tax Saver Fund?",
        "What is the minimum SIP for HDFC Small Cap Fund?",
        "What is the exit load of HDFC ELSS Tax Saver Fund?",
        "What is the riskometer and benchmark of HDFC Flexi Cap?",
        "How do I download my capital-gains statement?",
        "What is the NAV and fund size of HDFC Balanced Advantage Fund?",
        "Who manages HDFC Large Cap Fund?",
        # The three example questions the UI offers.
        "What is the expense ratio of HDFC Flexi Cap Fund?",
        "Is there a lock-in period on HDFC ELSS Tax Saver Fund?",
        "What is the minimum SIP for HDFC Small Cap Fund?",
        # Greetings and pleasantries.
        "hi",
        "hello there",
        "thanks",
        # Published figures must not trip a numeric rule.
        "Fund size is 986236.84 Cr and NAV is 65.4321",
        "expense ratio 1.03% and ter 0.5%",
        "exit load nil, 3Y lock-in, min SIP 500",
        "rating 4 stars, 326 holdings",
        "benchmark NIFTY 50 TRI index",
        "stamp duty 0.5%",
    ]

    def test_detects_identifiers(self) -> None:
        for text, kind in self.POSITIVES:
            with self.subTest(kind=kind, text=text):
                result = scan(text)
                self.assertTrue(result.is_pii, f"{text!r} should be flagged")
                self.assertIn(kind, result.kinds)

    def test_does_not_flag_ordinary_questions(self) -> None:
        for text in self.NEGATIVES:
            with self.subTest(text=text):
                result = scan(text)
                self.assertFalse(result.is_pii, f"{text!r} wrongly flagged as {result.kinds}")

    def test_scrub_redacts_every_match(self) -> None:
        from mf_rag.pii import REDACTED, scrub

        sample = (
            "pan ABCDE1234F, aadhaar 2345 6789 0123, mail a.b@example.com, "
            "call 9876543210, card 4111 1111 1111 1111, IFSC HDFC0001234, passport A2096457"
        )
        scrubbed = scrub(sample)
        for literal in (
            "ABCDE1234F",
            "2345 6789 0123",
            "a.b@example.com",
            "9876543210",
            "4111 1111 1111 1111",
            "HDFC0001234",
            "A2096457",
        ):
            self.assertNotIn(literal, scrubbed, f"{literal!r} survived scrubbing")
        # Scrubbing must be self-consistent: nothing left that any rule can match.
        self.assertFalse(scan(scrubbed).is_pii, f"residue still matches: {scan(scrubbed).kinds}")
        self.assertIn(REDACTED, scrubbed)

    def test_refusal_message_cites_privacy_policy(self) -> None:
        from mf_rag.pii import PiiResult
        from mf_rag.sources import EDUCATIONAL_LINKS

        message = PiiResult(is_pii=True, kinds=["PAN"]).message
        self.assertIn(EDUCATIONAL_LINKS["privacy"], message)
        self.assertNotIn("{", message, "refusal template left an unfilled slot")
        for banned in ("PAN,", "Aadhaar, account or folio"):
            self.assertIn(banned, message)

    def test_scrub_leaves_ordinary_questions_untouched(self) -> None:
        from mf_rag.pii import scrub

        for text in self.NEGATIVES:
            with self.subTest(text=text):
                self.assertEqual(scrub(text), text)


class TestRouting(unittest.TestCase):
    def test_detect_scheme_aliases(self) -> None:
        cases = {
            "expense ratio of HDFC Large Cap Fund": "Large Cap",
            "elss lock in period": "ELSS",
            "minimum sip for flexi cap": "Flexi Cap",
            "benchmark of hdfc balanced advantage": "Balanced Advantage",
            "small cap aum": "Small Cap",
        }
        for query, expected in cases.items():
            self.assertEqual(detect_scheme(query), expected, query)

    def test_detect_intent_prefers_specific_over_generic(self) -> None:
        self.assertEqual(detect_intent("How do I download my statement?")[0], "statement")
        self.assertEqual(detect_intent("historic change in exit load")[0], "exit_load_history")
        self.assertEqual(detect_intent("what is the expense ratio")[0], "expense_ratio")
        self.assertEqual(detect_intent("elss lock in period")[0], "lock_in")

    def test_intent_phrasings_route_to_the_right_field(self) -> None:
        """Regression guard: a matched intent must answer its own field, not a neighbour."""
        cases = {
            "When was HDFC Small Cap Fund launched?": "launch_date",
            "What is the launch date of HDFC ELSS?": "launch_date",
            "What is the date of incorporation of HDFC Large Cap?": "launch_date",
            "How old is HDFC Flexi Cap?": "launch_date",
            "What is the RTA of HDFC Large Cap?": "rta",
            "Who is the registrar and transfer agent of HDFC ELSS?": "rta",
            "Who is the custodian of HDFC ELSS?": "custodian",
            "What is the sub-category of HDFC Balanced Advantage?": "sub_category",
            "What is the fund category of HDFC Small Cap?": "category",
            "Which AMC manages HDFC Large Cap?": "fund_house",
            "What is the rating of HDFC Flexi Cap?": "rating",
        }
        for query, expected in cases.items():
            self.assertEqual(detect_intent(query)[0], expected, query)

    def test_answers_report_the_asked_field_not_a_neighbour(self) -> None:
        """The label in the answer must be the field the user actually asked for."""
        cases = [
            ("When was HDFC Small Cap Fund launched?", "Launch Date"),
            ("What is the RTA of HDFC Large Cap?", "Registrar & Transfer Agent"),
            ("Who is the custodian of HDFC ELSS?", "Custodian"),
            ("What is the sub-category of HDFC Balanced Advantage?", "Sub-category"),
            ("What is the fund category of HDFC Small Cap?", "Fund category"),
            ("Which AMC manages HDFC Large Cap?", "Fund house"),
            ("What is the rating of HDFC Flexi Cap?", "Rating"),
            ("What is the NAV of HDFC Small Cap?", "NAV"),
            ("What is the AUM of HDFC Balanced Advantage?", "Fund size"),
            ("What is the stamp duty on HDFC ELSS?", "Stamp duty"),
        ]
        for query, label in cases:
            answer = assistant().ask(query)
            self.assertEqual(answer.kind, "fact", query)
            self.assertIn(label, answer.text, f"{query} -> {answer.text}")

    def test_aum_reports_scheme_size_not_amc_total(self) -> None:
        """The pages carry both scheme AUM and an AMC-wide 'Total AUM'; only scheme size is valid."""
        answer = assistant().ask("What is the AUM of HDFC Balanced Advantage Fund?")
        self.assertIn("Fund size", answer.text)
        self.assertNotIn("Total AUM", answer.text)

    def test_advice_patterns(self) -> None:
        for query in [
            "Should I buy HDFC Small Cap Fund?",
            "Which HDFC fund is best for me?",
            "Is now a good time to sell my ELSS?",
            "How much should I invest in Flexi Cap?",
            "recommend a fund",
            # Adjective-before-noun order, which the noun-first patterns miss.
            "which is the best fund",
            "which one should I buy",
            "which of these is suitable for me",
            "my portfolio is too risky",
            "help me choose",
            "market timing",
        ]:
            self.assertTrue(ADVICE_RE.search(query), query)

    def test_performance_and_projection_patterns(self) -> None:
        from mf_rag.answer import COMPUTE_RE, PERFORMANCE_RE

        for query in [
            "what is the CAGR of HDFC Large Cap",
            "which fund is the top performer",
            "what is the ranking of these funds",
            "what is the alpha and sharpe ratio",
            "has HDFC ELSS outperformed",
        ]:
            self.assertTrue(PERFORMANCE_RE.search(query), query)
        for query in [
            "calculate the returns over 5 years",
            "project the value in 10 years",
            "what if I invest for 20 years",
            "estimate my future corpus",
        ]:
            self.assertTrue(COMPUTE_RE.search(query), query)

    def test_guardrails_do_not_fire_on_fact_questions(self) -> None:
        """A refusal pattern that over-matches turns a facts assistant into a wall."""
        from mf_rag.answer import COMPUTE_RE, PERFORMANCE_RE

        for query in [
            "What is the expense ratio of HDFC Large Cap Fund?",
            "Is there a lock-in period on HDFC ELSS Tax Saver Fund?",
            "What is the minimum SIP for HDFC Small Cap Fund?",
            "What is the exit load of HDFC ELSS?",
            "What is the riskometer of HDFC Flexi Cap?",
            "What is the benchmark of HDFC Balanced Advantage Fund?",
            "How do I download my capital-gains statement?",
            "What is the NAV of HDFC Large Cap?",
            "What is the tax implication of HDFC ELSS?",
            "Who manages HDFC Large Cap Fund?",
            "What are the top holdings of HDFC Small Cap Fund?",
            "What is the stamp duty on investment?",
            "What is the launch date of HDFC Large Cap?",
            "Who is the custodian?",
            "What is the rating of HDFC Balanced Advantage Fund?",
            "What is the investment objective?",
            "What is the exit load history of HDFC ELSS?",
        ]:
            with self.subTest(query=query):
                self.assertIsNone(ADVICE_RE.search(query), f"advice matched: {query}")
                self.assertIsNone(PERFORMANCE_RE.search(query), f"performance matched: {query}")
                self.assertIsNone(COMPUTE_RE.search(query), f"compute matched: {query}")


class TestAnswers(unittest.TestCase):
    def test_fact_answers_cite_exactly_one_source(self) -> None:
        queries = [
            "What is the expense ratio of HDFC Large Cap Fund?",
            "What is the expense ratio of HDFC ELSS Tax Saver Fund?",
            "What is the exit load of HDFC Small Cap Fund?",
            "What is the minimum SIP for HDFC Balanced Advantage Fund?",
            "What is the benchmark of HDFC Flexi Cap Fund?",
            "What is the riskometer level of HDFC Small Cap Fund?",
            "What is the NAV of HDFC Large Cap?",
            "What is the AUM of HDFC Balanced Advantage Fund?",
            "What is the investment objective of HDFC Small Cap Fund?",
        ]
        for query in queries:
            answer = assistant().ask(query)
            self.assertEqual(len(answer.citations), 1, query)
            self.assertIn(answer.citations[0], SOURCE_URLS, query)
            self.assertEqual(answer.kind, "fact", query)
            self.assertFalse(answer.refused, query)

    def test_answers_are_at_most_three_sentences(self) -> None:
        queries = [
            "What is the expense ratio of HDFC Large Cap Fund?",
            "What is the exit load of HDFC Small Cap Fund?",
            "Who manages HDFC Large Cap Fund?",
            "What are the top holdings of HDFC Flexi Cap?",
            "Is there a lock-in period on HDFC ELSS?",
        ]
        for query in queries:
            answer = assistant().ask(query)
            self.assertLessEqual(sentences(answer.text), 3, f"{query} -> {answer.text}")

    def test_answers_are_grounded_in_retrieved_chunks(self) -> None:
        chunks = load_chunks()
        corpus_text = "\n".join(c.cite_text for c in chunks)
        queries = [
            "What is the expense ratio of HDFC Large Cap Fund?",
            "What is the exit load of HDFC ELSS?",
            "What is the benchmark of HDFC Balanced Advantage Fund?",
            "What is the minimum lump sum for HDFC ELSS?",
            "What is the riskometer level of HDFC Flexi Cap?",
        ]
        for query in queries:
            answer = assistant().ask(query)
            if answer.kind != "fact":
                continue
            body = answer.text.split(" - ", 1)[-1]
            checked = 0
            for piece in re.split(r"[.;]\s", body):
                piece = piece.strip().rstrip(".;,")
                if len(piece) < 12:
                    continue
                checked += 1
                self.assertIn(piece, corpus_text, f"{query}: ungrounded fragment {piece!r}")
            self.assertGreater(checked, 0, f"{query}: nothing was checked")

    def test_per_scheme_answers_do_not_cross_contaminate(self) -> None:
        expensed = {
            "HDFC Large Cap Fund": "1.03%",
            "HDFC ELSS Tax Saver Fund": "1.21%",
            "HDFC Small Cap Fund": "0.78%",
            "HDFC Balanced Advantage Fund": "0.78%",
        }
        for scheme, expected in expensed.items():
            answer = assistant().ask(f"What is the expense ratio of {scheme}?")
            self.assertIn(expected, answer.text, scheme)
            self.assertEqual(
                answer.citations[0], SOURCES_BY_KEY[_key_for(scheme)].url, scheme
            )

    def test_manager_answer_names_primary_and_others(self) -> None:
        answer = assistant().ask("Who manages HDFC Large Cap Fund?")
        self.assertIn("Rahul Baijal", answer.text)
        self.assertIn("Dhruv Muchhal", answer.text)
        self.assertLessEqual(sentences(answer.text), 3)

    def test_holdings_answer_is_grounded(self) -> None:
        answer = assistant().ask("What are the top holdings of HDFC Flexi Cap?")
        self.assertIn("ICICI Bank Ltd", answer.text)
        self.assertEqual(answer.citations[0], SOURCES_BY_KEY["hdfc_flexi_cap"].url)

    def test_statement_question_returns_guidance(self) -> None:
        answer = assistant().ask("How do I download my capital gains statement?")
        self.assertEqual(answer.kind, "guidance")
        self.assertIn("help", answer.citations[0])
        self.assertIn("PAN", answer.text)

    def test_advice_is_refused_with_educational_link(self) -> None:
        for query in [
            "Should I buy HDFC Small Cap Fund?",
            "Which HDFC fund is best for me?",
            "Is now a good time to sell my ELSS?",
        ]:
            answer = assistant().ask(query)
            self.assertTrue(answer.refused, query)
            self.assertEqual(answer.kind, "advice_refusal", query)
            self.assertTrue(answer.citations, query)

    def test_performance_and_projection_are_refused(self) -> None:
        for query in [
            "What are the returns of HDFC Large Cap?",
            "Calculate the 10 year return of HDFC Small Cap",
            "Estimate the 5 year SIP projection",
        ]:
            answer = assistant().ask(query)
            self.assertTrue(answer.refused, query)
            self.assertIn(answer.kind, ("performance_refusal", "compute_refusal"), query)

    def test_pii_is_refused_before_retrieval(self) -> None:
        for query in [
            "My PAN is ABCDE1234F, check my returns",
            "Call me on 9876543210 about my folio",
            "email me at me@example.com",
        ]:
            answer = assistant().ask(query)
            self.assertEqual(answer.kind, "pii", query)
            self.assertTrue(answer.refused, query)
            self.assertEqual(answer.chunks, [], query)

    def test_out_of_scope_query(self) -> None:
        for query in ["What is the weather today?", "Tell me a joke about stocks"]:
            answer = assistant().ask(query)
            self.assertEqual(answer.kind, "out_of_scope", query)

    def test_answers_never_leak_identifiers(self) -> None:
        patterns = [r"[A-Z]{5}\d{4}[A-Z]", r"\b\d{10}\b", r"[\w.+-]+@[\w-]+\.[\w.-]+"]
        for query in [
            "What is the expense ratio of HDFC Large Cap Fund?",
            "My PAN is ABCDE1234F and phone is 9876543210",
            "Who manages HDFC ELSS?",
        ]:
            answer = assistant().ask(query)
            for pattern in patterns:
                self.assertIsNone(re.search(pattern, answer.text), f"{query} -> {answer.text}")

    def test_every_answer_carries_timestamp_when_sourced(self) -> None:
        answer = assistant().ask("What is the expense ratio of HDFC Large Cap Fund?")
        self.assertTrue(answer.retrieved_at)
        self.assertIn("Last updated from sources", answer.render())

    def test_render_omits_timestamp_but_keeps_privacy_link_for_pii_refusal(self) -> None:
        answer = assistant().ask("My PAN is ABCDE1234F")
        self.assertNotIn("Last updated", answer.render())
        self.assertIn("Source: https://groww.in/privacy-policy", answer.render())
        self.assertNotIn("ABCDE1234F", answer.render())
        self.assertLessEqual(sentences(answer.text), 3)

    def test_every_answer_path_returns_a_citation(self) -> None:
        queries = [
            "What is the expense ratio of HDFC Large Cap Fund?",
            "What is the lock-in period on HDFC ELSS?",
            "How do I download my statement?",
            "Should I buy HDFC Small Cap?",
            "What are the returns of HDFC Large Cap?",
            "Calculate 5 year projection for HDFC Flexi Cap",
            "My PAN is ABCDE1234F",
            "What is the weather today?",
        ]
        for query in queries:
            answer = assistant().ask(query)
            self.assertTrue(answer.citations, f"{query} returned no citation")
            for url in answer.citations:
                self.assertTrue(url.startswith("https://"), query)

    def test_greeting(self) -> None:
        answer = assistant().ask("hi")
        self.assertEqual(answer.kind, "greeting")

    def test_scheme_name_does_not_hijack_intent(self) -> None:
        """Words inside a scheme name are metadata, not the question.

        "HDFC ELSS *Tax* Saver Fund" used to be routed as a *tax* question, so
        "What is the RTA of HDFC ELSS Tax Saver Fund?" fell through to a scope
        message instead of the RTA.
        """
        cases = [
            ("What is the RTA of HDFC ELSS Tax Saver Fund?", "rta"),
            ("What is the tax implication of HDFC ELSS Tax Saver Fund?", "tax"),
            ("What is the tax of HDFC ELSS?", "tax"),
            ("What is the expense ratio of HDFC Large Cap Fund?", "expense_ratio"),
            ("Who manages HDFC Large Cap Fund?", "manager"),
            ("What is the exit load history of HDFC ELSS?", "exit_load_history"),
            ("What is the fund size AUM of HDFC Balanced Advantage Fund?", "aum"),
        ]
        for query, expected in cases:
            with self.subTest(query=query):
                scheme = detect_scheme(query)
                self.assertIsNotNone(scheme, query)
                intent, _ = detect_intent(strip_scheme_names(query, scheme))
                self.assertEqual(intent, expected, query)
                self.assertEqual(assistant().ask(query).kind, "fact", query)

    def test_every_field_resolves_for_every_scheme(self) -> None:
        """No field/scheme pair may silently fall back to a scope message."""
        for scheme in ["HDFC Large Cap Fund", "HDFC ELSS Tax Saver Fund", "HDFC Small Cap Fund"]:
            for field in [
                "expense ratio",
                "exit load",
                "lock-in period",
                "minimum SIP",
                "minimum lump sum",
                "riskometer",
                "benchmark",
                "NAV",
                "fund size AUM",
                "RTA",
                "custodian",
                "stamp duty",
                "rating",
                "launch date",
                "sub-category",
                "investment objective",
            ]:
                query = f"What is the {field} of {scheme}?"
                with self.subTest(query=query):
                    self.assertEqual(assistant().ask(query).kind, "fact", query)

    def test_unresolved_scheme_returns_scope_message(self) -> None:
        """A fund outside the corpus must get a scope message, never another fund's number.

        Note: a name that *contains* a known alias ("HDFC Ultra Large Cap Fund") still
        resolves to that scheme, because alias matching is substring-based. That is a
        deliberate trade-off, not covered here.
        """
        for query in [
            "What is the expense ratio of HDFC Parag Parhat Fund?",
            "Tell me about HDFC Floating Rate Fund",
            "What is the NAV of HDFC Money Market Fund?",
            "What is the exit load of HDFC Dividend Yield Fund?",
        ]:
            with self.subTest(query=query):
                answer = assistant().ask(query)
                self.assertEqual(answer.kind, "out_of_scope", answer.text)
                self.assertEqual(len(answer.citations), 1)

    def test_resolved_and_unnamed_schemes_still_answer(self) -> None:
        """The unresolved-scheme guard must not fire on questions it should not touch."""
        for query in [
            "What is the expense ratio of HDFC Large Cap Fund?",
            "What is the expense ratio of HDFC ELSS Tax Saver Fund?",
            "Is there a lock-in period on HDFC ELSS Tax Saver Fund?",
            "Who manages HDFC Large Cap Fund?",
            "What are the top holdings of HDFC Small Cap Fund?",
            "What is the expense ratio?",
            "What is the minimum SIP?",
            "Who is the custodian?",
            "hi",
        ]:
            with self.subTest(query=query):
                self.assertNotEqual(assistant().ask(query).kind, "out_of_scope", query)

    def test_empty_query_cites_and_stamps(self) -> None:
        for query in ["", "   "]:
            with self.subTest(query=query):
                answer = assistant().ask(query)
                self.assertEqual(answer.kind, "empty")
                self.assertEqual(len(answer.citations), 1, answer.text)
                self.assertTrue(answer.retrieved_at, "empty answer text contains a link but no stamp")

    def test_sentence_cap_holds_for_every_answer_kind(self) -> None:
        for query in [
            "What is the expense ratio of HDFC Large Cap Fund?",
            "Who manages HDFC Large Cap Fund?",
            "What are the top holdings of HDFC Flexi Cap?",
            "How do I download my capital-gains statement?",
            "Should I buy HDFC Small Cap Fund?",
            "My PAN is ABCDE1234F, check my returns",
            "what is the weather in Mumbai",
            "what is the CAGR of HDFC Large Cap",
            "hi",
            "",
        ]:
            with self.subTest(query=query):
                answer = assistant().ask(query)
                self.assertLessEqual(sentences(answer.text), 3, f"{query!r} -> {answer.text}")

    def test_pre_retrieval_responses_carry_no_stamp(self) -> None:
        """Only answers that consulted no page may omit the timestamp."""
        for query, kind, citations in [
            ("My PAN is ABCDE1234F", "pii", 1),      # cites the privacy policy
            ("hi", "greeting", 0),                    # cites nothing: it is a greeting
        ]:
            with self.subTest(query=query):
                answer = assistant().ask(query)
                self.assertEqual(answer.kind, kind)
                self.assertEqual(answer.retrieved_at, "")
                self.assertEqual(answer.chunks, [], "no retrieval should have happened")
                self.assertEqual(len(answer.citations), citations)


class TestPipeline(unittest.TestCase):
    """`mf_rag/pipeline.py` is the one entry point used by both the CLI and the UI."""

    def test_get_assistant_is_cached(self) -> None:
        self.assertIs(get_assistant(), get_assistant())

    def test_pipeline_missing_message_names_the_script(self) -> None:
        with mock.patch("mf_rag.pipeline.index_exists", return_value=False):
            get_assistant.cache_clear()
            try:
                with self.assertRaises(PipelineMissing) as ctx:
                    get_assistant()
            finally:
                get_assistant.cache_clear()
        self.assertIn("build_index.py", str(ctx.exception))

    def test_three_example_questions(self) -> None:
        self.assertEqual(len(EXAMPLE_QUESTIONS), 3)
        for question in EXAMPLE_QUESTIONS:
            with self.subTest(question=question):
                answer = pipeline_ask(question)
                self.assertEqual(len(answer.citations), 1, question)
                self.assertTrue(answer.citations[0].startswith("https://"), question)
                self.assertTrue(answer.retrieved_at, question)
                self.assertFalse(answer.refused, question)

    def test_ask_module_level_helper(self) -> None:
        answer = pipeline_ask("What is the expense ratio of HDFC Flexi Cap Fund?")
        self.assertIsInstance(answer, Answer)
        self.assertEqual(answer.kind, "fact")
        self.assertEqual(answer.intent, "expense_ratio")

    def test_example_questions_cover_distinct_fields(self) -> None:
        """The three demos must not all answer the same field."""
        intents = {pipeline_ask(q).intent for q in EXAMPLE_QUESTIONS}
        self.assertEqual(len(intents), 3, intents)

    def test_get_assistant_raises_when_index_is_missing(self) -> None:
        """The real index is present here, so this asserts the happy path stays intact."""
        self.assertIsInstance(get_assistant(), FAQAssistant)

    def test_rupee_sign_renders_in_the_cli(self) -> None:
        """The corpus contains U+20B9, which a default cp1252 Windows console cannot
        encode. That made the CLI raise UnicodeEncodeError and die on the rupee
        questions, so both the data and the console fix are pinned here."""
        answer = pipeline_ask("What is the minimum SIP for HDFC Small Cap Fund?")
        self.assertIn("₹", answer.text, "expected a rupee-denominated fact")

        source = (ROOT / "scripts" / "ask.py").read_text(encoding="utf-8")
        self.assertIn("reconfigure", source, "CLI must force a UTF-8 stream")
        self.assertIn("65001", source, "CLI must switch the Windows console to UTF-8")

    def test_interactive_banner_includes_disclaimer(self) -> None:
        """The banner used to be followed by an unreachable `print(DISCLAIMER)`,
        because every branch of the REPL loop returned first."""
        source = (ROOT / "scripts" / "ask.py").read_text(encoding="utf-8")
        banner_prints = source.index('print("HDFC Mutual Fund FAQ assistant')
        disclaimer_print = source.index("print(DISCLAIMER)")
        loop_start = source.index("while True:")
        self.assertLess(disclaimer_print, loop_start, "DISCLAIMER must print before the REPL loop")

    def test_cli_declares_its_contract(self) -> None:
        source = (ROOT / "scripts" / "ask.py").read_text(encoding="utf-8")
        for expected in ['nargs="*"', '"--debug"', "PipelineMissing", "sys.path.insert"]:
            self.assertIn(expected, source)


class TestUiSource(unittest.TestCase):
    """Source-level checks on `app.py`, so the main suite never boots Streamlit.

    The behavioural equivalent lives in `scripts/test_ui.py` (AppTest).
    """

    def setUp(self) -> None:
        self.source = (ROOT / "app.py").read_text(encoding="utf-8")

    def test_app_has_facts_only_note(self) -> None:
        self.assertIn("DISCLAIMER", self.source)
        self.assertIn("st.caption(DISCLAIMER)", self.source)
        self.assertIn("Facts-only", DISCLAIMER)
        self.assertIn("no investment advice", DISCLAIMER.lower())
        self.assertIn("Facts only, no buy/sell advice", self.source)
        self.assertIn("No return calculations or comparisons", self.source)

    def test_app_lists_three_examples(self) -> None:
        self.assertIn("EXAMPLE_QUESTIONS", self.source)
        self.assertEqual(len(EXAMPLE_QUESTIONS), 3)
        for question in EXAMPLE_QUESTIONS:
            with self.subTest(question=question):
                self.assertTrue(
                    any(word in question for word in ("expense ratio", "lock-in", "SIP")), question
                )
                self.assertTrue(pipeline_ask(question).citations, question)

    def test_example_questions_are_clickable_not_plain_text(self) -> None:
        """FR-8.2 says the 3 examples are clickable/suggested.

        The list used to be rendered as `st.markdown` bullets, so the requirement was
        unmet and `test_app_lists_three_examples` could not catch it: that test only ever
        checked the pipeline's constant, never how app.py draws it. Assert both halves -
        a button per example, and no leftover markdown list of the examples.
        """
        self.assertIn("def example_questions(", self.source)
        self.assertIn("st.button(question", self.source)
        self.assertIn("pending_question", self.source)
        self.assertNotRegex(
            self.source,
            r'st\.markdown\(f?"- \{question\}',
            "examples must be buttons, not markdown bullets",
        )

    def test_clicked_example_is_answered_and_not_repeated(self) -> None:
        """A click hands the question over via pending_question, which must be consumed
        once: Streamlit reruns top-to-bottom, so a stale value would re-answer the same
        example on the next rerun (e.g. when the trace checkbox is ticked)."""
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
        self.assertEqual([e.value for e in at.exception], [])

        labels = [b.label for b in at.button]
        for question in EXAMPLE_QUESTIONS:
            with self.subTest(question=question):
                self.assertIn(question, labels, "one clickable button per example")

        at.button[labels.index(EXAMPLE_QUESTIONS[0])].click().run()
        self.assertEqual([e.value for e in at.exception], [])

        rendered = [m.value for m in at.markdown]
        self.assertEqual(
            sum(1 for m in rendered if m == EXAMPLE_QUESTIONS[0]),
            1,
            "the clicked question must appear exactly once",
        )
        self.assertTrue(
            any("Expense ratio" in m for m in rendered), "the click must produce a real answer"
        )
        self.assertTrue(
            [c for c in at.get("link_button") if c.label == "Source"],
            "the answered example must still carry its source link",
        )

        # A rerun with no new question must not re-answer the same example.
        at.checkbox[0].check().run()
        after = [m.value for m in at.markdown]
        self.assertEqual(
            sum(1 for m in after if m == EXAMPLE_QUESTIONS[0]),
            1,
            "pending_question was not cleared, so the example re-answered on rerun",
        )

    def test_app_handles_missing_index(self) -> None:
        self.assertIn("if not index_exists():", self.source)
        self.assertIn("st.error", self.source)
        self.assertIn("python scripts/build_index.py", self.source)
        self.assertIn("except PipelineMissing", self.source)
        # the guard must run before the assistant is ever constructed
        self.assertLess(
            self.source.index("if not index_exists():"),
            self.source.index("bot = assistant()"),
            "the missing-index guard must come before the assistant is built",
        )

    def test_render_answer_switches_on_kind(self) -> None:
        body = self.source[self.source.index("def render_answer") :]
        self.assertIn('if answer.kind == "advice_refusal"', body)
        self.assertIn("st.info(answer.text)", body)
        self.assertIn('elif answer.kind in ("pii", "compute_refusal", "performance_refusal")', body)
        self.assertIn("st.warning(answer.text)", body)
        self.assertIn("st.markdown(answer.text)", body)
        self.assertIn('st.link_button("Source", url)', body)

    def test_timestamp_caption_is_guarded(self) -> None:
        """I-5: the PII refusal must show no timestamp, so the caption is conditional."""
        body = self.source[self.source.index("def render_answer") :]
        self.assertIn("if answer.retrieved_at:", body)
        caption = body.index("Last updated from sources")
        self.assertLess(
            body.index("if answer.retrieved_at:"),
            caption,
            "the caption must be inside the retrieved_at guard",
        )
        # and no unconditional caption elsewhere in the file
        self.assertEqual(self.source.count("Last updated from sources"), 1)

    def test_settings_precede_the_transcript(self) -> None:
        """Streamlit reruns top-to-bottom.

        An expander below the transcript always lags one run behind, so ticking
        'Show retrieval trace' would leave the answers above it untraced until the
        next question was asked. The Settings block must be rendered first.
        """
        self.assertLess(
            self.source.index('with st.expander("Settings")'),
            self.source.index("for question, answer in st.session_state.history"),
            "Settings must be evaluated before the transcript is rendered",
        )

    def test_uses_the_cached_pipeline_entry_point(self) -> None:
        self.assertIn("@st.cache_resource", self.source)
        self.assertIn("get_assistant()", self.source)
        self.assertIn("st.session_state.history", self.source)
        self.assertIn("st.chat_input", self.source)
        self.assertIn("Searching the corpus...", self.source)
        self.assertIn("st.rerun()", self.source)


class TestEvaluationHarness(unittest.TestCase):
    """Guards the two evaluation scripts' promises, without re-running them.

    Running the benchmark embeds ~200 chunks twice, so these are source-level and
    report-level checks that keep the main suite fast.
    """

    def setUp(self) -> None:
        self.bench_src = (ROOT / "scripts" / "benchmark_retrieval.py").read_text(encoding="utf-8")
        self.chunking_src = (ROOT / "scripts" / "compare_chunking.py").read_text(encoding="utf-8")
        self.bench = json.loads((ROOT / "data" / "retrieval_benchmark.json").read_text(encoding="utf-8"))
        self.chunking = json.loads((ROOT / "data" / "chunking_report.json").read_text(encoding="utf-8"))

    def test_benchmark_never_opens_the_production_store(self) -> None:
        """Chroma rewrites its SQLite and HNSW files on open+query.

        An earlier version queried data/chroma for the structure-aware leg, which
        silently rewrote every file in the shipped index. The benchmark must embed
        both corpora into its own scratch store instead.
        """
        for banned in ("get_client", "CHROMA_DIR", "hdfc_mutual_funds"):
            self.assertNotIn(banned, self.bench_src, f"benchmark must not reference {banned}")
        self.assertIn("build_scratch(structured, fixed)", self.bench_src)
        self.assertIn('"bench_struct"', self.bench_src)
        self.assertIn('"bench_fixed"', self.bench_src)

    def test_benchmark_writes_only_to_the_scratch_dir(self) -> None:
        self.assertIn('SCRATCH = DATA_DIR / "chroma_bench"', self.bench_src)
        self.assertIn("shutil.rmtree", self.bench_src)
        self.assertIn("(DATA_DIR / \"retrieval_benchmark.json\")", self.bench_src)

    def test_benchmark_reports_top1_top3_top5(self) -> None:
        for strategy in ("structure_aware", "fixed_size_180"):
            for depth in ("top1", "top3", "top5"):
                self.assertIn(depth, self.bench[strategy], f"{strategy}.{depth} missing")

    def test_benchmark_has_35_rows_per_strategy(self) -> None:
        rows = self.bench["per_query"]
        self.assertEqual(len(rows), 35)
        self.assertEqual({r["scheme"] for r in rows}.__len__(), 5)
        self.assertEqual({r["field"] for r in rows}.__len__(), 7)
        for row in rows:
            for key in ("struct_top1", "struct_top3", "struct_top5", "fixed_top1", "fixed_top3", "fixed_top5"):
                self.assertIn(key, row)

    def test_benchmark_states_caveats_in_words(self) -> None:
        caveats = self.bench["caveats"]
        self.assertGreater(len(caveats.split()), 60, "caveats must be prose, not a token")
        self.assertIn("DENSE-ONLY", caveats)
        self.assertIn("SMALL-SAMPLE", caveats)
        self.assertIn("at least this good", caveats.lower())
        self.assertIn("not a measurement", caveats.lower())

    def test_structure_aware_beats_fixed_windows(self) -> None:
        self.assertGreater(
            self.bench["structure_aware"]["top1"],
            self.bench["fixed_size_180"]["top1"] * 5,
            "structure-aware top-1 should be far above the fixed-window baseline",
        )

    def test_chunking_report_labels_its_metric_honestly(self) -> None:
        """cooccur_ratio is 1.0 for every strategy, so the file must say it does not
        discriminate rather than letting a reader mistake 1.0 for a perfect score."""
        self.assertIn("metric_is_diagnostic_only", self.chunking)
        self.assertIn("does NOT", self.chunking["metric_is_diagnostic_only"])
        self.assertIn("benchmark_retrieval.py", self.chunking["metric_is_diagnostic_only"])
        self.assertIn("real_evidence", self.chunking)

    def test_chunking_report_confirms_every_strategy_scores_one(self) -> None:
        strategies = [k for k in self.chunking if k.startswith(("fixed_size_", "chosen_"))]
        self.assertGreaterEqual(len(strategies), 5)
        for name in strategies:
            self.assertEqual(
                self.chunking[name]["cooccur_ratio"], 1.0, f"{name} is the reason the metric is not diagnostic"
            )
        self.assertEqual(self.chunking["chosen_structure_aware"]["chunks_over_hard_max"], 0)


class TestParaphraseHarness(unittest.TestCase):
    """The evaluation-only LLM tooling. None of this may reach the answer path."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cache = json.loads((ROOT / "data" / "paraphrase_queries.json").read_text(encoding="utf-8"))
        cls.report = json.loads((ROOT / "data" / "paraphrase_benchmark.json").read_text(encoding="utf-8"))

    def test_answer_path_does_not_import_the_llm_module(self) -> None:
        """D1: the assistant has no language model. This is the guard for that decision.

        If a future change wires mf_rag.llm into answering, this fails on purpose.
        """
        for name in ("answer.py", "pipeline.py", "retrieve.py", "ingest.py", "chunking.py"):
            source = (ROOT / "mf_rag" / name).read_text(encoding="utf-8")
            with self.subTest(module=name):
                self.assertNotRegex(source, r"^\s*(from|import)\s+.*\bllm\b", f"{name} must not import the LLM client")

    def test_harness_needs_no_api_key(self) -> None:
        """The committed cache must make evaluation reproducible with no credentials."""
        from mf_rag.llm import LlmUnavailable, load_env

        self.assertIn("generated_by", self.cache)
        self.assertTrue(self.cache["generated_by"], "cache must record what produced it")
        for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY"):
            self.assertFalse(load_env().get(name), f"{name} must not be committed")

    def test_cache_covers_the_same_35_gold_pairs(self) -> None:
        from scripts.benchmark_retrieval import BENCH_FIELDS
        from scripts.gen_paraphrases import build_pairs

        expected = {f"{p['scheme']}|{p['field']}" for p in build_pairs()}
        self.assertEqual(set(self.cache["queries"]), expected)
        self.assertEqual(len(expected), 35)
        self.assertEqual(len(BENCH_FIELDS), 7)

    def test_every_style_gets_a_distinct_paraphrase(self) -> None:
        """Guards a real bug: an early seed ignored the style key and wrote one wording
        into all five style slots, making the whole comparison vacuous."""
        styles = list(self.cache["styles"])
        self.assertEqual(len(styles), 5)
        for key, entry in self.cache["queries"].items():
            with self.subTest(query=key):
                self.assertEqual(set(entry["paraphrases"]), set(styles))
                texts = [entry["paraphrases"][s][0] for s in styles]
                for text in texts:
                    self.assertTrue(text.strip(), f"{key}: empty paraphrase")
                    self.assertNotEqual(text, entry["template_query"], f"{key}: paraphrase is just the template")
                self.assertEqual(len(set(texts)), len(texts), f"{key}: styles produced duplicate wordings")

    def test_paraphrases_still_name_the_gold_scheme(self) -> None:
        """A rewrite that dropped the fund name would test scheme resolution, not phrasing."""
        for key, entry in self.cache["queries"].items():
            for style, texts in entry["paraphrases"].items():
                for text in texts:
                    with self.subTest(query=key, style=style):
                        self.assertIn(entry["scheme"], text, f"{key}/{style} lost the fund name")

    def test_report_is_internally_consistent(self) -> None:
        summary = self.report["summary"]
        self.assertIn("template", summary, "the author's phrasing is the baseline")
        for style, stats in summary.items():
            with self.subTest(style=style):
                self.assertEqual(stats["n"], 35)
                for depth in ("top1", "top3", "top5"):
                    self.assertTrue(0.0 <= stats[depth] <= 1.0)
                self.assertLessEqual(stats["top1"], stats["top3"])
                self.assertLessEqual(stats["top3"], stats["top5"])

    def test_report_carries_the_interpretive_caveat(self) -> None:
        """A 17% row is meaningless without the key that says the seed omits field words."""
        caveats = self.report["caveats"].lower()
        self.assertIn("colloquial", caveats)
        self.assertIn("without naming the field", caveats)
        self.assertIn("never as an accuracy figure", caveats)
        self.assertIn("how_to_read", self.report["measurement"])


class TestDeliverables(unittest.TestCase):
    """The submission artefacts must not drift from the running system.

    P10's whole point is that SOURCES / SAMPLE_QA / README are generated from code.
    These tests fail if a hand-edited number or URL stops matching the system.
    """

    def setUp(self) -> None:
        d = ROOT / "deliverables"
        self.qa = json.loads((d / "SAMPLE_QA.json").read_text(encoding="utf-8"))
        self.qa_md = (d / "SAMPLE_QA.md").read_text(encoding="utf-8")
        self.sources_md = (d / "SOURCES.md").read_text(encoding="utf-8")
        self.sources_csv = (d / "SOURCES.csv").read_text(encoding="utf-8")
        self.disclaimer_md = (d / "DISCLAIMER.md").read_text(encoding="utf-8")
        self.readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.bench = json.loads((ROOT / "data" / "retrieval_benchmark.json").read_text(encoding="utf-8"))

    def test_sample_qa_has_exactly_ten_entries(self) -> None:
        self.assertEqual(len(self.qa), 10)
        for index, question in enumerate(self.qa, start=1):
            self.assertIn(f"### {index}. {question['question']}", self.qa_md)

    def test_every_sample_answer_is_traceable_to_the_assistant(self) -> None:
        """No hand-written answers: re-ask each question and require an exact match."""
        for row in self.qa:
            with self.subTest(question=row["question"]):
                live = pipeline_ask(row["question"])
                self.assertEqual(row["answer"], live.text)
                self.assertEqual(row["citation"], live.primary_citation)
                self.assertEqual(row["kind"], live.kind)
                self.assertEqual(row["intent"], live.intent)
                self.assertEqual(row["refused"], live.refused)
                self.assertEqual(row["last_updated"], live.retrieved_at)

    def test_fund_facts_cite_a_scheme_page_and_guidance_only_educational(self) -> None:
        scheme_urls = {source.url for source in SOURCES}
        for row in self.qa:
            with self.subTest(question=row["question"]):
                if row["kind"] == "fact":
                    self.assertIn(row["citation"], scheme_urls)
                elif row["kind"] == "guidance":
                    self.assertNotIn(row["citation"], scheme_urls)
                    self.assertIn(row["citation"], set(EDUCATIONAL_LINKS.values()))

    def test_pii_row_has_no_timestamp(self) -> None:
        pii = [row for row in self.qa if row["kind"] == "pii"]
        self.assertEqual(len(pii), 1)
        self.assertEqual(pii[0]["last_updated"], "")
        self.assertEqual(pii[0]["citation"], EDUCATIONAL_LINKS["privacy"])
        self.assertIn("n/a (no source consulted)", self.qa_md)

    def test_pii_identifier_never_reaches_an_answer(self) -> None:
        """The spec mandates the PII demo question, so the identifier appears in the
        question heading. It must never appear in any ANSWER - the refusal has to
        scrub it, not echo it back."""
        for row in self.qa:
            with self.subTest(question=row["topic"]):
                self.assertNotIn("ABCDE1234F", row["answer"])
        # and the deliverable must not quote the refusal verbatim either
        refusal = [row for row in self.qa if row["kind"] == "pii"][0]["answer"]
        self.assertNotIn("ABCDE1234F", refusal)

    def test_every_row_has_exactly_one_citation(self) -> None:
        for row in self.qa:
            with self.subTest(question=row["question"]):
                self.assertTrue(row["citation"].startswith("https://"), row["question"])

    def test_readme_benchmark_table_matches_the_report(self) -> None:
        """Hand-copied numbers rot. The README table must match the JSON exactly."""
        s = self.bench["structure_aware"]
        f = self.bench["fixed_size_180"]
        for expected in [
            f"| **Structure-aware** (atomic facts + prose) | {self.bench['corpus_chunks']['structure_aware']} | **{s['top1']:.2%}** | **{s['top3']:.2%}** | {s['top5']:.2%} |",
            f"| Fixed 180-word windows (40-word overlap) | {self.bench['corpus_chunks']['fixed_180']} | {f['top1']:.2%} | {f['top3']:.2%} | {f['top5']:.2%} |",
        ]:
            self.assertIn(expected, self.readme, f"README table is stale, expected a row like:\n{expected}")

    def test_readme_carries_both_chunking_caveats(self) -> None:
        self.assertIn("cooccur_ratio", self.readme)
        self.assertIn("1.0 for every", self.readme)
        self.assertIn("dense-only", self.readme)
        self.assertIn("not a measurement", self.readme)
        # and it must not quote the top-5 column as if it supported the design
        self.assertIn("Read the top-1 column", self.readme)

    def test_sources_md_lists_every_scheme_and_no_orphan_urls(self) -> None:
        for source in SOURCES:
            self.assertIn(source.scheme_name, self.sources_md)
            self.assertIn(source.url, self.sources_md)
        self.assertIn("## Not used", self.sources_md)
        self.assertIn("No PII of any kind is requested, received or stored.", self.sources_md)
        # a scheme page must never be labelled as a guidance-only link
        for url in (source.url for source in SOURCES):
            guidance = self.sources_md.split("## Secondary / educational links")[1].split("## Not used")[0]
            before_primary_note = guidance.split("primary** scheme")[0]
            self.assertNotIn(url, before_primary_note, f"{url} is a scheme page, not a guidance link")

    def test_sources_csv_matches_the_registry(self) -> None:
        rows = list(csv.DictReader(self.sources_csv.splitlines()))
        self.assertEqual(len(rows), 5)
        for row, source in zip(rows, SOURCES):
            self.assertEqual(row["key"], source.key)
            self.assertEqual(row["scheme_name"], source.scheme_name)
            self.assertEqual(row["url"], source.url)
            self.assertEqual(row["plan"], source.plan)

    def test_disclaimer_md_quotes_the_exact_disclaimer(self) -> None:
        self.assertIn(DISCLAIMER, self.disclaimer_md)
        self.assertIn("app.py", self.disclaimer_md)

    def test_disclaimer_appears_in_all_three_surfaces(self) -> None:
        # the Markdown deliverables quote the text; app.py imports the constant
        # rather than duplicating the string, so reference the name there
        self.assertIn(DISCLAIMER, self.sources_md)
        self.assertIn(DISCLAIMER, self.qa_md)
        app = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn("st.caption(DISCLAIMER)", app)
        self.assertNotIn(DISCLAIMER, app, "app.py must import the constant, not copy the text")


class TestConversationMemory(unittest.TestCase):
    """Follow-up questions should resolve the fund named in a recent turn.

    The safety properties matter more than the feature: memory may only ever choose which
    scheme retrieval is allowed to consider. It must not widen the scope gate, must not let
    a comparator collapse onto one fund, and must never carry an identifier forward.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.elss = next(s for s in SOURCES if s.scheme_short == "ELSS")
        cls.flexi = next(s for s in SOURCES if s.scheme_short == "Flexi Cap")
        first = assistant().ask(f"What is the expense ratio of {cls.elss.scheme_name}?")
        cls.history = [(f"What is the expense ratio of {cls.elss.scheme_name}?", first)]

    # --- the feature ---------------------------------------------------
    def test_followup_resolves_scheme_from_earlier_turn(self) -> None:
        answer = assistant().ask("What about its lock-in?", self.history)
        self.assertEqual(answer.kind, "fact")
        self.assertIn(self.elss.scheme_name, answer.text)
        self.assertTrue(answer.notes, "using memory should be recorded in the trace")

    def test_short_bare_followup_resolves(self) -> None:
        answer = assistant().ask("And the exit load?", self.history)
        self.assertEqual(answer.kind, "fact")
        self.assertIn("Exit load", answer.text)

    def test_scheme_is_recovered_from_the_answer_side(self) -> None:
        # The user never names the fund; only the assistant's answer does.
        opening = f"What is the exit load of {self.elss.scheme_name}?"
        self.assertNotIn("elss", opening.split("?")[0].replace(self.elss.scheme_name, "").lower())
        answer = assistant().ask("And the expense ratio?", [(opening, assistant().ask(opening))])
        self.assertEqual(answer.kind, "fact")
        self.assertIn(self.elss.scheme_name, answer.text)

    def test_only_the_last_window_of_turns_is_remembered(self) -> None:
        old = [(f"What is the expense ratio of {self.elss.scheme_name}?", None)]
        recent = [(f"What is the exit load of {self.flexi.scheme_name}?", None)]
        resolution = memory.resolve(
            "What about its lock-in?", old + recent + recent, window=2
        )
        self.assertEqual(resolution.scheme, self.flexi.scheme_short)

    def test_recent_turns_caps_the_window(self) -> None:
        self.assertEqual(len(memory.recent_turns(list(range(25)))), MEMORY_TURNS)
        self.assertEqual(len(memory.recent_turns(list(range(3)))), 3)
        self.assertEqual(memory.recent_turns([]), [])
        self.assertEqual(MEMORY_TURNS, 10)

    # --- memory must not change the guards ----------------------------
    def test_explicit_scheme_in_the_query_beats_memory(self) -> None:
        answer = assistant().ask(
            f"What is the exit load of {self.flexi.scheme_name}?", self.history
        )
        self.assertIn(self.flexi.scheme_name, answer.text)
        self.assertFalse(answer.notes, "an explicit scheme must not be attributed to memory")

    def test_out_of_scope_question_stays_out_of_scope(self) -> None:
        for question in (
            "What is the weather in Mumbai?",
            "Tell me a joke",
            "What about the fund manager in Mumbai?",
        ):
            with self.subTest(question=question):
                self.assertEqual(assistant().ask(question).kind, "out_of_scope")
                with_memory = assistant().ask(question, self.history)
                self.assertEqual(
                    with_memory.kind,
                    "out_of_scope",
                    "conversation memory must not widen the scope gate",
                )

    def test_comparative_question_is_not_treated_as_a_follow_up(self) -> None:
        for question in (
            "Which of them has the lowest exit load?",
            "What is the fund with the highest returns?",
            "Compare all the funds",
        ):
            with self.subTest(question=question):
                self.assertFalse(memory.is_follow_up(question))
                self.assertFalse(memory.resolve(question, self.history).used)

    def test_pronoun_markers_need_word_boundaries(self) -> None:
        # "it" must not be found inside "exit", "with" or "unit".
        for question in (
            "What is the exit load?",
            "What is the minimum unit size?",
            "Tell me the fund with the lowest fees",
        ):
            with self.subTest(question=question):
                self.assertFalse(memory.is_follow_up(question))

    def test_follow_up_markers_are_detected(self) -> None:
        for question in (
            "What about its lock-in?",
            "And the exit load?",
            "lock-in?",
            "How about this fund?",
        ):
            with self.subTest(question=question):
                self.assertTrue(memory.is_follow_up(question))

    def test_refusals_still_fire_before_memory(self) -> None:
        expected = {
            "Should I buy it?": "advice_refusal",
            "What if I invested 5000 a month?": "compute_refusal",
        }
        for question, kind in expected.items():
            with self.subTest(question=question):
                answer = assistant().ask(question, self.history)
                self.assertEqual(answer.kind, kind)
                self.assertEqual(answer.chunks, [])

    def test_pii_is_still_refused_when_history_is_present(self) -> None:
        answer = assistant().ask(
            "My PAN is ABCDE1234F, what is the expense ratio?", self.history
        )
        self.assertEqual(answer.kind, "pii")
        self.assertEqual(answer.chunks, [])

    def test_identifiers_in_history_never_reach_retrieval(self) -> None:
        opening = "my PAN is ABCDE1234F and folio 12345678"
        history = [(opening, assistant().ask(opening))]
        answer = assistant().ask("What about its expense ratio?", history)
        blob = answer.text + "".join(chunk.cite_text for chunk in answer.chunks)
        for secret in ("ABCDE1234F", "12345678"):
            self.assertNotIn(secret, blob)

    def test_turn_that_is_only_identifiers_is_skipped(self) -> None:
        resolution = memory.resolve("What about its lock-in?", [("PAN ABCDE1234F", None)])
        self.assertFalse(resolution.used)

    # --- the feature must be invisible without history -----------------
    def test_no_history_is_byte_identical(self) -> None:
        for question in (
            "What is the expense ratio of the HDFC ELSS Tax Saver Fund?",
            "What is the weather in Mumbai?",
            "Should I buy it?",
            "Tell me a joke",
        ):
            with self.subTest(question=question):
                without = assistant().ask(question)
                with_empty = assistant().ask(question, [])
                self.assertEqual(without.render(), with_empty.render())
                self.assertEqual(without.kind, with_empty.kind)
                self.assertEqual(without.citations, with_empty.citations)
                self.assertEqual(without.notes, with_empty.notes)

    def test_ask_signature_keeps_history_optional(self) -> None:
        self.assertEqual(assistant().ask("Tell me a joke").kind, "out_of_scope")
        self.assertEqual(pipeline_ask("Tell me a joke").kind, "out_of_scope")

    # --- retriever filter ----------------------------------------------
    def test_retriever_scheme_filter_is_a_hard_filter(self) -> None:
        retriever = assistant().retriever
        unfiltered = retriever.search("expense ratio", top_k=6)
        self.assertGreater(len(unfiltered), 1)
        self.assertGreater(
            len({chunk.scheme_short for chunk in unfiltered}), 1, "expected a mixed result set"
        )
        filtered = retriever.search(
            "expense ratio", top_k=6, allowed_schemes={self.elss.scheme_short}
        )
        self.assertTrue(filtered)
        self.assertEqual({chunk.scheme_short for chunk in filtered}, {self.elss.scheme_short})

    def test_unknown_scheme_filter_returns_nothing(self) -> None:
        self.assertEqual(
            assistant().retriever.search("expense ratio", allowed_schemes={"No Such Fund"}), []
        )

    def test_section_and_scheme_filters_intersect(self) -> None:
        retriever = assistant().retriever
        result = retriever.search(
            "expense ratio", section="Key facts", allowed_schemes={self.elss.scheme_short}
        )
        self.assertTrue(result, "expected the two filters to leave a non-empty set")
        for chunk in result:
            self.assertEqual(chunk.section, "Key facts")
            self.assertEqual(chunk.scheme_short, self.elss.scheme_short)

    def test_conflicting_filters_return_nothing(self) -> None:
        # Derive a section the remembered fund has and the other one does not, so the two
        # filters genuinely cannot both be satisfied.
        other_sections = {
            c.section
            for c in assistant().retriever.chunks
            if c.scheme_short == self.flexi.scheme_short
        }
        exclusive = next(
            c.section
            for c in assistant().retriever.chunks
            if c.scheme_short == self.elss.scheme_short and c.section not in other_sections
        )
        self.assertEqual(
            assistant().retriever.search(
                "expense ratio", section=exclusive, allowed_schemes={self.flexi.scheme_short}
            ),
            [],
            "an empty intersection must return nothing rather than everything",
        )

    # --- architecture ---------------------------------------------------
    def test_memory_does_not_leak_into_the_answer_path(self) -> None:
        source = (ROOT / "mf_rag" / "answer.py").read_text(encoding="utf-8")
        answer_module = (ROOT / "mf_rag" / "answer.py").read_text(encoding="utf-8")
        self.assertNotIn("from .llm", source)
        self.assertNotIn("import llm", answer_module)
        # Memory is imported lazily inside ask(); it must never be a top-level import that
        # could become a cycle, and it must not import answer back.
        memory_source = (ROOT / "mf_rag" / "memory.py").read_text(encoding="utf-8")
        self.assertNotIn("from .answer", memory_source)
        self.assertNotIn("import answer", memory_source)


def _key_for(scheme_fragment: str) -> str:
    for source in SOURCES:
        if scheme_fragment.lower() in source.scheme_name.lower():
            return source.key
    raise AssertionError(scheme_fragment)


if __name__ == "__main__":
    unittest.main(verbosity=2)
