"""STAGE 6 - Answer composition, deliberately extractive.

Every sentence is copied from a retrieved chunk, so the assistant cannot invent
a fee, a date or a ratio. It routes the query to an intent, pulls the matching
labelled fact out of the retrieved context, and cites the one source page.

Guardrails: advice and portfolio questions are refused, performance questions are
not computed, answers are capped at three sentences, and every answer carries a
single source link plus a "Last updated from sources" stamp.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from .config import MAX_ANSWER_SENTENCES, MAX_CONTEXT_CHUNKS
from .retrieve import HybridRetriever, RetrievedChunk
from .sources import EDUCATIONAL_LINKS, SOURCES

ADVICE_PATTERNS = (
    r"\bshould\s+(?:i|we)\b",
    r"\b(?:should|ought\s+to)\s+(?:buy|sell|redeem|switch|invest|purchase|hold|exit|enter)\b",
    r"\b(?:is\s+it\s+(?:a\s+)?good|worth\s+(?:investing|buying|it))\b",
    r"\bwhich\s+(?:fund|scheme|mutual\s+fund)\s+(?:should|do\s+i|to)\b",
    r"\b(?:best|top|good)\s+(?:fund|scheme|mutual\s+fund)\s+(?:to|for)\b",
    r"\b(?:recommend|suggestion|suggest)\b",
    r"\b(?:can|i\s+should)\s+i\s+(?:allocate|invest|put)\b",
    r"\bhow\s+much\s+(?:should|do)\s+i\b",
    r"\b(?:tell|advise)\s+me\s+what\s+to\b",
    r"\bis\s+now\s+a\s+good\s+time\b",
    r"\bbook\s+profits?\b",
    r"\bmarket\s+timing\b",
    r"\bmy\s+portfolio\b",
    r"\bhow\s+should\s+i\s+split\b",
    r"\bswitch\s+(?:from|to)\b.*\bto\b",
    r"\bcompare\s+.*\b(?:better|best|which\s+one)\b",
    # "which is the best fund" puts the adjective before the noun, so the patterns
    # above (which expect the noun first) miss it. Cover both word orders.
    r"\bwhich\b[^?]{0,30}\b(?:is|are|was|were)\b[^?]{0,25}\b(?:best|top|good|suitable|right|finest)\b",
    r"\bwhich\s+(?:one|of\s+(?:these|them|my|the))\b[^?]{0,30}\b(?:best|should|to\s+buy|to\s+pick)\b",
    r"\bwhich\b[^?]{0,40}\b(?:fund|scheme|etf)\b[^?]{0,30}\b(?:best|good|suitable|right)\b",
    r"\b(?:best|good|suitable|right)\b[^?]{0,20}\b(?:fund|scheme|etf)\b[^?]{0,20}\bfor\s+me\b",
    r"\b(?:is|are)\b[^?]{0,30}\b(?:safe|risky|good)\b[^?]{0,20}\bfor\s+(?:me|my|retirement|child)\b",
    r"\b(?:am\s+i|should\s+i)\s+(?:too\s+\w+\s+to|ready\s+to)\b",
    r"\bhelp\s+me\s+choose\b",
)
ADVICE_RE = re.compile("|".join(ADVICE_PATTERNS), re.I)

PERFORMANCE_RE = re.compile(
    r"\b(?:return|returns|performance|profit|growth|cagr|alpha|sharpe|"
    # Match the verb whatever the subject is ("has HDFC ELSS outperformed"), not
    # only the literal "has it outperformed".
    r"(?:out|under|over)perform\w*|top\s+performer|rank(?:ing)?)\b",
    re.I,
)
COMPUTE_RE = re.compile(
    r"\b(?:calculate|compute|project|estimate|forecast|what\s+if|simulate|"
    r"how\s+much\s+(?:would|will|can)\s+i)\b",
    re.I,
)

INTENTS: tuple[tuple[str, re.Pattern[str], tuple[str, ...]], ...] = (
    (
        "exit_load_history",
        re.compile(
            r"\bexit\s+load\s+(?:history|revision|change[ds]?)\b|"
            r"\bhistoric(?:al)?\s+exit\s+load\b|\bchange[d]?\s+in\s+exit\s+load\b|"
            r"\bexit\s+load\s+has\s+(?:changed|been)\b",
            re.I,
        ),
        (),
    ),
    (
        "statement",
        re.compile(
            r"\bstatement\b|\bdownload\b|\bcas\b|\bpassbook\b|"
            r"\baccount\s+statement\b|\bcapital\s+gains\s+statement\b|"
            r"\bhow\s+do\s+i\s+(?:get|download|see)\b",
            re.I,
        ),
        (),
    ),
    ("expense_ratio", re.compile(r"\bexpense\s+ratio\b|\bter\b|\bcharges?\b.*\bfee\b|\bfee\b", re.I), ("expense_ratio",)),
    ("exit_load", re.compile(r"\bexit\s+load\b|\bredemption\s+charge\b|\bload\b", re.I), ("exit_load",)),
    ("lock_in", re.compile(r"\block[\s-]?in\b|\belss\b.*\b(?:period|year|months?)\b|\b3\s*year\b|\b80\s*c\b", re.I), ("lock_in",)),
    ("min_sip", re.compile(r"\bmin(?:imum)?\.?\s*(?:for\s*)?sip\b|\bminimum\s+sip\b|\bsip\s+amount\b|\bhow\s+much\s+(?:for\s+)?sip\b", re.I), ("min_sip",)),
    ("min_lumpsum", re.compile(r"\blump\s?sum\b|\bminimum\s+investment\b|\bmin(?:imum)?\.?\s*for\s+(?:1st|first|2nd|second)\b", re.I), ("min_lumpsum", "min_additional")),
    ("benchmark", re.compile(r"\bbenchmark\b|\bindex\b.*\btrack\w*\b|\bwhich\s+index\b", re.I), ("benchmark",)),
    ("riskometer", re.compile(r"\briskometer\b|\brisk\s+level\b|\brisk\s+rating\b|\bhow\s+risky\b|\bvery\s+high\s+risk\b", re.I), ("riskometer",)),
    ("rating", re.compile(r"\brating\b|\bstars?\b|\bhow\s+many\s+stars\b", re.I), ("rating",)),
    ("nav", re.compile(r"\bnav\b|\bunit\s+price\b", re.I), ("nav",)),
    ("aum", re.compile(r"\baum\b|\bfund\s+size\b|\bsize\s+of\s+the\s+fund\b|\bcorpus\b", re.I), ("aum",)),
    ("stamp_duty", re.compile(r"\bstamp\s+duty\b", re.I), ("stamp_duty",)),
    ("tax", re.compile(r"\btax\b|\bcapital\s+gains?\b|\b80\s*c\b|\bstcg\b|\bltcg\b|\b12\.5\b|\b20%\b", re.I), ("tax_implication",)),
    ("objective", re.compile(r"\bobjective\b|\bwhat\s+does\s+(?:it|the\s+scheme)\s+do\b|\baims?\s+to\b|\bstrategy\b", re.I), ("objective",)),
    ("manager", re.compile(r"\bmanager\b|\bwho\s+manages\b|\bfund\s+manager\b", re.I), ()),
    ("holdings", re.compile(r"\bholdings\b|\bportfolio\b|\btop\s+\d+\s+stocks?\b", re.I), ()),
    (
        "launch_date",
        re.compile(
            r"\blaunch\s+date\b|\blaunched\b|\blaunch(?:ing)?\s+(?:of\s+the\s+scheme\b)?|"
            r"\bwhen\s+(?:was|did|is|were)\b[^?]{0,60}\b(?:launch|start|commence|inception)|"
            r"\bdate\s+of\s+(?:launch|incorporation)\b|\bincorporat\w*\b|"
            r"\binception\s+date\b|\bgo\s+live\b|\bhow\s+old\s+is\b",
            re.I,
        ),
        ("launch_date", "incorporation_date"),
    ),
    (
        "rta",
        re.compile(
            r"\brta\b|\bregistrar\b|\bregistrars\b|\btransfer\s+agent\b|"
            r"\bregistrars?\s+and\s+transfer\b",
            re.I,
        ),
        ("rta",),
    ),
    ("custodian", re.compile(r"\bcustodian\b|\bwho\s+(?:holds|custodies)\b", re.I), ("custodian",)),
    (
        "sub_category",
        re.compile(r"\bsub[\s-]?categor\w+\b|\bse[c]?tor\s+class\w*\b|\bsegment\b", re.I),
        ("sub_category",),
    ),
    (
        "category",
        re.compile(r"\b(?:fund\s+)?categor\w+\b|\bscheme\s+type\b|\bwhat\s+type\s+of\s+fund\b", re.I),
        ("category",),
    ),
    ("fund_house", re.compile(r"\bfund\s+house\b|\bwhich\s+amc\b|\bamc\b|\bwho\s+manages\s+the\s+fund\b", re.I), ("fund_house",)),
)

ADVICE_REFUSAL = (
    "I only answer published facts about HDFC Mutual Fund schemes, so I can't say whether you "
    "should buy, sell or hold anything. For a suitability view, speak to a SEBI-registered "
    "investment adviser. Educational reference: {link}"
)

PERFORMANCE_REFUSAL = (
    "I don't calculate or compare returns. The scheme page lists the published figures and links "
    "to the official factsheet: {link}"
)

COMPUTE_REFUSAL = (
    "I don't project or estimate future values. I can only report the published facts on the scheme "
    "page: {link}"
)

OUT_OF_SCOPE = (
    "I could not find that on the five HDFC scheme pages I cover. I can answer published facts such "
    "as expense ratio, exit load, minimum SIP, lock-in, riskometer, benchmark, NAV and fund size. "
    "Source list: {link}"
)

GREETING = (
    "Hi! I answer published facts about five HDFC Mutual Fund direct-growth schemes: "
    "expense ratio, exit load, minimum SIP, ELSS lock-in, riskometer, benchmark, NAV and fund size. "
    "Facts-only, no investment advice."
)


FIELD_QUERY_HINT: dict[str, str] = {
    "expense_ratio": "expense ratio",
    "exit_load": "exit load",
    "lock_in": "lock-in period ELSS 3Y Lock-in",
    "min_sip": "Min. for SIP minimum SIP amount",
    "min_lumpsum": "Min. for 1st investment minimum lump sum",
    "min_additional": "Min. for 2nd investment minimum additional investment",
    "benchmark": "Fund benchmark index",
    "riskometer": "Riskometer risk level",
    "rating": "Rating",
    "nav": "NAV",
    "aum": "Fund size AUM",
    "tax_implication": "Tax implication capital gains",
    "stamp_duty": "Stamp duty on investment",
    "objective": "Investment Objective scheme seeks",
    "launch_date": "Launch Date",
    "incorporation_date": "Date of Incorporation",
    "custodian": "Custodian",
    "rta": "Registrar and Transfer Agent RTA",
    "fund_house": "Fund house",
    "category": "Fund category",
    "sub_category": "Sub-category",
}

SECTION_QUERY_HINT: dict[str, str] = {
    "manager": "Compare Fund management fund manager also manages tenure",
    "holdings": "Holdings top holdings largest portfolio weight",
    "exit_load_history": "Exit load revision history historic exit load changes",
}

SECTION_FOR_INTENT: dict[str, str] = {
    "manager": "Fund management",
    "holdings": "Holdings",
    "exit_load_history": "Fees, exit load and tax",
}

STOPWORDS = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "of", "for", "to", "in", "on", "at", "by",
        "and", "or", "it", "its", "this", "that", "my", "me", "i", "you", "your", "do", "does",
        "did", "can", "could", "would", "should", "what", "whats", "how", "much", "many", "tell",
        "give", "please", "about", "with", "from", "be", "been", "has", "have", "had", "there",
    }
)
MIN_LEXICAL_OVERLAP = 0.5


def content_tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS}


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
    def primary_citation(self) -> str:
        return self.citations[0] if self.citations else ""

    def render(self) -> str:
        parts = [self.text]
        if self.citations:
            parts.append("Source: " + " | ".join(self.citations))
        if self.retrieved_at:
            parts.append(f"Last updated from sources: {self.retrieved_at}")
        return "\n\n".join(parts)


def _sentences(text: str, limit: int) -> str:
    parts = [p for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p.strip()]
    if not parts:
        return ""
    kept = parts[:limit]
    joined = " ".join(kept).rstrip()
    if len(parts) > limit and not joined.endswith((".", "!", "?")):
        joined += "."
    return joined


MANAGER_HEAD_RE = re.compile(r"^Fund manager:\s*(?P<name>[^(]+?)\s*$")
TENURE_RE = re.compile(r"^(?P<name>[^(]{2,60}?)\s*\((?P<tenure>[^)]+)\)")


def _parse_manager(cite_text: str) -> tuple[str, str]:
    match = TENURE_RE.match(cite_text.strip())
    if not match:
        return "", ""
    return match.group("name").strip(), match.group("tenure").strip()


def detect_scheme(query: str) -> str | None:
    """Resolve a scheme short name from a query.

    Delegates to the retriever so the answer stage and the scheme boost can never
    disagree about which scheme a question is about.
    """
    from .retrieve import detect_scheme as _detect_scheme

    return _detect_scheme(query)


def strip_scheme_names(query: str, scheme: str | None) -> str:
    """Remove the resolved scheme name from the question before intent detection.

    Scheme names are metadata, not the question. Without this, the words in a name
    hijack routing: "RTA of HDFC ELSS *Tax* Saver Fund" is detected as a *tax*
    question because the name contains the word "Tax". Longest match wins so that
    "tax saver" is removed before the bare alias "elss".
    """
    if scheme is None:
        return query

    from .retrieve import SCHEME_ALIASES

    names: list[str] = []
    for source in SOURCES:
        if source.key == scheme:
            names.extend((source.display_name, source.scheme_name, source.scheme_short))
    names.extend(alias for alias, key in SCHEME_ALIASES.items() if key == scheme)

    for name in sorted({n for n in names if n}, key=len, reverse=True):
        query = re.sub(re.escape(name), " ", query, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", query).strip()


def detect_intent(query: str) -> tuple[str, tuple[str, ...]]:
    for name, pattern, fields in INTENTS:
        if pattern.search(query):
            return name, fields
    return "unknown", ()


def _scheme_vocabulary() -> frozenset[str]:
    """Every word that legitimately appears in one of the five scheme names.

    Derived from the registry so it cannot drift when a source is added.
    """
    from .retrieve import SCHEME_ALIASES

    words: set[str] = set()
    for source in SOURCES:
        for name in (source.scheme_name, source.scheme_short, source.display_name, source.category, source.plan):
            words.update(re.findall(r"[a-z]+", name.lower()))
    for alias in SCHEME_ALIASES:
        words.update(re.findall(r"[a-z]+", alias.lower()))
    return frozenset(words)


SCHEME_VOCABULARY = _scheme_vocabulary()


def names_unresolved_scheme(query: str, scheme: str | None) -> bool:
    """True when the question names a fund that is not in the corpus.

    Without this, "expense ratio of HDFC Parag Parhat Fund" resolves no scheme and
    then happily returns some *other* fund's expense ratio, which is worse than a
    scope message: it is a confident, cited, wrong number.

    The test is a proper noun that is neither sentence-initial nor part of any
    registered scheme name. Lower-case questions are unaffected, and a question
    that does resolve is unaffected.
    """
    if scheme is not None:
        return False
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9&]*", query)
    for token in tokens[1:]:  # tokens[0] may be capitalised only as sentence start
        lowered = token.lower()
        if token.isupper():
            continue
        if token[0].isupper() and lowered not in SCHEME_VOCABULARY:
            return True
    return False


def _pick_fact_chunk(chunks: Sequence[RetrievedChunk], fields: Sequence[str], scheme: str | None) -> RetrievedChunk | None:
    for field_name in fields:
        candidates = [c for c in chunks if c.field_name == field_name]
        if scheme:
            scoped = [c for c in candidates if c.scheme_short == scheme]
            if scoped:
                return scoped[0]
        if candidates:
            return candidates[0]
    return None


def _fact_sentence(chunk: RetrievedChunk, *, scoped: bool = False) -> str:
    if chunk.kind != "fact":
        return _sentences(chunk.cite_text, MAX_ANSWER_SENTENCES)
    base = chunk.cite_text.rstrip(".")
    if scoped:
        return f"On the {chunk.scheme_name} page: {base}."
    return f"{chunk.scheme_name} - {base}."


def _link_for(scheme: str | None) -> str:
    for source in SOURCES:
        if source.scheme_short == scheme:
            return source.url
    return SOURCES[0].url


def _educational(name: str) -> str:
    return EDUCATIONAL_LINKS.get(name, SOURCES[0].url)


class FAQAssistant:
    def __init__(self, retriever: HybridRetriever, top_k: int = MAX_CONTEXT_CHUNKS) -> None:
        self.retriever = retriever
        self.top_k = top_k
        self._stamps: dict[str, str] = {}
        self._vocabulary: set[str] = set()
        for chunk in retriever.chunks:
            self._stamps[chunk.source_url] = chunk.retrieved_at
            self._vocabulary.update(content_tokens(chunk.cite_text))
            self._vocabulary.update(content_tokens(chunk.heading))
        self._corpus_stamp = max(self._stamps.values(), default="")

    def _stamp_for(self, url: str) -> str:
        return self._stamps.get(url, self._corpus_stamp)

    def _targeted(
        self, cleaned: str, scheme: str | None, hint: str, section: str | None = None,
        allowed_schemes: set[str] | None = None,
    ) -> list[RetrievedChunk]:
        parts = [cleaned]
        if scheme:
            for source in SOURCES:
                if source.scheme_short == scheme:
                    parts.append(source.display_name)
        parts.append(hint)
        return self.retriever.search(
            " ".join(parts), top_k=self.top_k, section=section, allowed_schemes=allowed_schemes
        )

    def _in_scope(self, query: str) -> bool:
        tokens = content_tokens(query)
        if not tokens:
            return False
        overlap = len(tokens & self._vocabulary) / len(tokens)
        return overlap >= MIN_LEXICAL_OVERLAP

    def _manager_answer(
        self, scheme: str | None, traced: Sequence[RetrievedChunk]
    ) -> tuple[str, str, list[RetrievedChunk]] | None:
        pool = [
            c
            for c in self.retriever.chunks
            if c.section == "Fund management" and (scheme is None or c.scheme_short == scheme)
        ]
        if not pool:
            return None
        chosen_scheme = scheme or (traced[0].scheme_short if traced else pool[0].scheme_short)
        pool = [c for c in pool if c.scheme_short == chosen_scheme]
        pool.sort(key=lambda c: c.ordinal)
        parsed = [(*_parse_manager(c.cite_text), c) for c in pool]
        parsed = [p for p in parsed if p[0]]
        if not parsed:
            return None
        primary_name, primary_tenure, primary_chunk = parsed[0]
        others = [name for name, _tenure, _c in parsed[1:]]
        scheme_name = primary_chunk.scheme_name
        text = f"{scheme_name} is managed by {primary_name} ({primary_tenure})."
        if others:
            joined = (
                ", ".join(others[:-1]) + f" and {others[-1]}" if len(others) > 1 else others[0]
            )
            text += f" The page also lists {joined}."
        return text, primary_chunk.source_url, list(traced) or [primary_chunk]

    def ask(self, query: str, history: Sequence[tuple[str, Any]] | None = None) -> Answer:
        from . import pii as pii_module

        original = query.strip()
        pii_result = pii_module.scan(original)
        if pii_result.is_pii:
            return Answer(
                text=pii_module.REFUSAL.format(link=_educational("privacy")),
                citations=[_educational("privacy")],
                kind="pii",
                refused=True,
                pii_kinds=pii_result.kinds,
                notes=["No source page was retrieved and nothing was stored."],
            )

        cleaned = pii_module.scrub(original)
        if not cleaned:
            link = _educational("terms")
            return Answer(
                text=OUT_OF_SCOPE.format(link=link),
                citations=[link],
                retrieved_at=self._corpus_stamp,
                kind="empty",
            )

        if len(cleaned) < 3:
            return Answer(text=GREETING, kind="greeting", intent="greeting")

        if ADVICE_RE.search(cleaned):
            return Answer(
                text=ADVICE_REFUSAL.format(link=_educational("terms")),
                citations=[_educational("terms")],
                retrieved_at=self._corpus_stamp,
                kind="advice_refusal",
                refused=True,
                intent="advice",
            )

        scheme = detect_scheme(cleaned)
        # Decided from the query alone, before any memory is consulted: does this question
        # name a fund we do not carry? Memory must not be able to switch this guard off by
        # handing us a scheme, or a follow-up would get a confident answer to a fund that
        # was never asked about. `names_unresolved_scheme` returns False whenever a scheme
        # is already set, so it has to be evaluated here, not after the block below.
        unresolved_name = names_unresolved_scheme(cleaned, scheme)
        # Conversation memory: only ever used when the query names no scheme itself, and
        # only to filter which scheme's chunks retrieval may consider. It is resolved after
        # every guardrail above, so PII, advice and compute refusals are unchanged, and it
        # cannot reach the out-of-scope check, which still runs on `cleaned` below.
        memory_note: tuple[str, ...] = ()
        if scheme is None and history:
            from . import memory as memory_module

            resolution = memory_module.resolve(cleaned, history)
            if resolution.used:
                scheme = resolution.scheme
                memory_note = resolution.notes
                memory_allowed = {scheme}
            else:
                memory_allowed = None
        else:
            memory_allowed = None

        intent, fields = detect_intent(strip_scheme_names(cleaned, scheme))

        if intent == "statement":
            link = _educational("statements")
            return Answer(
                text=(
                    "Account statements are not published on a fund page - they are generated inside your "
                    "own account. The official help centre covers downloading statements, capital gains "
                    "statements and tax documents. This assistant never asks for your PAN, folio number or OTP."
                ),
                citations=[link],
                retrieved_at=self._corpus_stamp,
                kind="guidance",
                intent=intent,
            )

        if unresolved_name:
            link = _educational("terms")
            return Answer(
                text=OUT_OF_SCOPE.format(link=link),
                citations=[link],
                retrieved_at=self._corpus_stamp,
                kind="out_of_scope",
                intent=intent,
            )

        if intent in SECTION_QUERY_HINT:
            section = SECTION_FOR_INTENT[intent]
            chunks = self._targeted(cleaned, scheme, SECTION_QUERY_HINT[intent], section=section)
            if scheme:
                scoped = [c for c in chunks if c.scheme_short == scheme]
                if scoped:
                    chunks = scoped
            if intent == "manager":
                manager_answer = self._manager_answer(scheme, chunks)
                if manager_answer is not None:
                    text, url, traced = manager_answer
                    return Answer(
                        text=text,
                        citations=[url],
                        retrieved_at=self._stamp_for(url),
                        kind="fact",
                        intent=intent,
                        chunks=traced,
                    )
                chunks = []
            elif intent == "holdings":
                hit = next((c for c in chunks if c.heading.startswith("Top")), None)
                chunks = [hit] if hit else []
            else:
                hit = next(
                    (c for c in chunks if "exit load revision history" in c.heading.lower()), None
                )
                chunks = [hit] if hit else []
            if chunks:
                hit = chunks[0]
                text = _sentences(hit.cite_text, 2)
                if scheme is None:
                    text = f"{hit.scheme_name} - {text}"
                return Answer(
                    text=text,
                    citations=[hit.source_url],
                    retrieved_at=self._stamp_for(hit.source_url),
                    kind="fact",
                    intent=intent,
                    chunks=chunks,
                )

        if COMPUTE_RE.search(cleaned) and intent in ("unknown", "performance"):
            link = _link_for(scheme)
            return Answer(
                text=COMPUTE_REFUSAL.format(link=link),
                citations=[link],
                retrieved_at=self._stamp_for(link),
                kind="compute_refusal",
                refused=True,
                intent="compute",
            )

        chunks = self.retriever.search(cleaned, top_k=self.top_k, allowed_schemes=memory_allowed)

        if fields:
            chosen = _pick_fact_chunk(chunks, fields, scheme)
            if chosen is None:
                for field_name in fields:
                    hint = FIELD_QUERY_HINT.get(field_name)
                    if not hint:
                        continue
                    retry = self._targeted(cleaned, scheme, hint, allowed_schemes=memory_allowed)
                    chosen = _pick_fact_chunk(retry, (field_name,), scheme)
                    if chosen is not None:
                        chunks = retry
                        break
            if chosen is not None:
                return Answer(
                    text=_fact_sentence(chosen, scoped=scheme is None),
                    citations=[chosen.source_url],
                    retrieved_at=self._stamp_for(chosen.source_url),
                    intent=intent,
                    kind="fact",
                    chunks=chunks,
                    notes=list(memory_note),
                )

        if PERFORMANCE_RE.search(cleaned) and not fields:
            link = _link_for(scheme)
            return Answer(
                text=PERFORMANCE_REFUSAL.format(link=link),
                citations=[link],
                retrieved_at=self._stamp_for(link),
                kind="performance_refusal",
                refused=True,
                intent="performance",
            )

        if intent == "unknown" and not self._in_scope(cleaned):
            return Answer(
                text=OUT_OF_SCOPE.format(link=_educational("terms")),
                citations=[_educational("terms")],
                retrieved_at=self._corpus_stamp,
                kind="out_of_scope",
                intent=intent,
            )

        if not chunks:
            return Answer(
                text=OUT_OF_SCOPE.format(link=_educational("terms")),
                citations=[_educational("terms")],
                retrieved_at=self._corpus_stamp,
                kind="out_of_scope",
                intent=intent,
            )

        if scheme:
            scoped = [c for c in chunks if c.scheme_short == scheme]
            if scoped:
                chunks = scoped
        best = chunks[0]
        if not best.field_name:
            return Answer(
                text=OUT_OF_SCOPE.format(link=_educational("terms")),
                citations=[_educational("terms")],
                retrieved_at=self._corpus_stamp,
                kind="out_of_scope",
                intent=intent,
            )
        return Answer(
            text=_fact_sentence(best, scoped=scheme is None),
            citations=[best.source_url],
            retrieved_at=self._stamp_for(best.source_url),
            intent=intent,
            kind="fact",
            chunks=chunks,
            notes=list(memory_note),
        )
