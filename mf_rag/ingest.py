"""STAGE 1 - Ingestion (loading). Fetch public pages into structured, citable documents.

Extraction is schema-driven rather than generic: the page has no semantic DOM
landmarks and its label/value pairs are split across sibling elements, so each
field is located by a known label and paired with the value that follows it.
The visible page always wins over the embedded __NEXT_DATA__ payload.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

import requests
from bs4 import BeautifulSoup, Tag

from .config import CLEAN_DIR, LOG_PATH, RAW_DIR
from .sources import SOURCES, Source

BLOCK_TAGS = frozenset(
    {
        "address", "article", "aside", "blockquote", "div", "dl", "dd", "dt",
        "fieldset", "figcaption", "figure", "footer", "form", "h1", "h2", "h3",
        "h4", "h5", "h6", "header", "hr", "li", "main", "nav", "ol", "p", "pre",
        "section", "table", "tbody", "td", "tfoot", "th", "thead", "tr", "ul",
    }
)
DROP_TAGS = frozenset({"script", "style", "noscript", "svg", "template", "iframe"})
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

MAX_HOLDINGS = 25
TOP_LEVEL_CATEGORIES = frozenset({"Equity", "Hybrid", "Debt"})
LOCK_IN_RE = re.compile(r"^(?:ELSS\s*•\s*)?(\d+\s*[YMY])\s*Lock-?in", re.I)
END_MARKERS = ("Contact Us", "Download the App", "Show More", "Others:", "© 2016-")
RISK_RE = re.compile(r"^(Very High|High|Moderately High|Moderate|Low)\s+Risk$", re.I)
DATE_RE = re.compile(r"^\d{1,2}\s+[A-Z][a-z]{2}\s+\d{4}$")
NAV_DATE_RE = re.compile(r"^NAV:\s*(.+)$", re.I)
BENCHMARK_RE = re.compile(r"^Fund benchmark\s+(.+)$", re.I)
CONTACT_LABEL_RE = re.compile(
    r"^(Fund house|Rank \(total assets\)|Total AUM|Date of Incorporation|Launch Date|"
    r"Custodian|Registrar & Transfer Agent|Phone|E-mail|Website|Address|Scheme Information Document\(SID\))\s*(.*)$"
)
TENURE_RE = re.compile(r"^[A-Z][a-z]{2}\s+\d{4}\s*-\s*(Present|\d{4})$")
MANAGER_INITIALS_RE = re.compile(r"^[A-Z]{2,3}$")
MONEY_RE = re.compile(r"^(₹[\d,]+(?:\.\d+)?\s*(?:Cr)?|[\d,]+\.\d+%|\d+(?:\.\d+)?%)$")

LABEL_FIELDS: dict[str, str] = {
    "Min. for SIP": "min_sip",
    "Fund size (AUM)": "aum",
    "Expense ratio": "expense_ratio",
    "Rating": "rating",
    "Min. for 1st investment": "min_lumpsum",
    "Min. for 2nd investment": "min_additional",
    "Min. for withdrawal": "min_withdrawal",
}


@dataclass
class Fact:
    field_name: str
    label: str
    value: str
    section: str
    as_of: str | None = None

    def sentence(self) -> str:
        if self.as_of:
            return f"{self.label}: {self.value} (as on {self.as_of})."
        return f"{self.label}: {self.value}."


@dataclass
class ProseBlock:
    heading: str
    text: str
    section: str


@dataclass
class Document:
    source: Source
    url: str
    title: str
    lines: list[str]
    facts: list[Fact] = field(default_factory=list)
    prose: list[ProseBlock] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)
    links: dict[str, str] = field(default_factory=dict)
    retrieved_at: str = ""
    conflicts: list[str] = field(default_factory=list)

    @property
    def scheme_name(self) -> str:
        return self.source.scheme_name

    @property
    def scheme_short(self) -> str:
        return self.source.scheme_short

    def fact(self, field_name: str) -> Fact | None:
        for item in self.facts:
            if item.field_name == field_name:
                return item
        return None

    def value(self, field_name: str, default: str = "") -> str:
        found = self.fact(field_name)
        return found.value if found else default


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def fetched_at(path: Path) -> str:
    """When the cached page was actually fetched, i.e. the mtime of the raw file.

    A page read from `data/raw/` was NOT retrieved now. Stamping it with the current
    time would make the assistant tell the user a source was last verified at the
    moment of the rebuild, which is a claim about freshness that was never true, and
    it would make `build_index.py` non-idempotent. The file mtime is the honest answer.
    """
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")


def fetch(url: str, *, retries: int = 3, timeout: int = 45) -> str:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(
                url,
                headers={"User-Agent": USER_AGENT, "Accept-Language": "en-IN,en;q=0.9"},
                timeout=timeout,
            )
            response.raise_for_status()
            return response.text
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url}: {last}")


def _normalise(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text.replace("\u00a0", " ").replace("\u200b", "")).strip()


def _has_text_bearing_block(element: Tag) -> bool:
    return any(
        isinstance(child, Tag) and child.get_text(" ", strip=True)
        for child in element.find_all(list(BLOCK_TAGS))
    )


def block_leaf_lines(soup: BeautifulSoup) -> list[str]:
    for tag in soup.find_all(list(DROP_TAGS)):
        tag.decompose()
    lines: list[str] = []
    for element in soup.find_all(list(BLOCK_TAGS)):
        if not isinstance(element, Tag) or _has_text_bearing_block(element):
            continue
        line = _normalise(element.get_text(" ", strip=True))
        if line and len(line) <= 1500:
            lines.append(line)
    return lines


def content_window(lines: list[str], scheme_name: str) -> list[str]:
    start = 0
    for index, line in enumerate(lines):
        if line.lower() == scheme_name.lower() and index + 1 < len(lines):
            if lines[index + 1] in TOP_LEVEL_CATEGORIES or LOCK_IN_RE.match(lines[index + 1]):
                start = index
                break
    else:
        for index, line in enumerate(lines):
            if line.lower() == scheme_name.lower():
                start = index
                break
    window = lines[start:]
    for index, line in enumerate(window):
        if line in END_MARKERS:
            return window[:index]
    return window


def extract_payload(html: str) -> dict[str, Any]:
    match = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.S)
    if not match:
        return {}
    try:
        data = json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}
    return data.get("props", {}).get("pageProps", {}).get("mfServerSideData", {}) or {}


def extract_links(soup: BeautifulSoup) -> dict[str, str]:
    links: dict[str, str] = {}
    for anchor in soup.find_all("a", href=True):
        text = _normalise(anchor.get_text(" ", strip=True)).lower()
        href = anchor["href"]
        if "scheme information document" in text and "sid" not in links:
            links["sid"] = href
        if "fact sheet" in text or "factsheet" in text:
            links.setdefault("factsheet", href)
        if "hdfcfund.com" in href and "hdfc_site" not in links:
            links["hdfc_site"] = href
    return links


def _next_value(lines: list[str], index: int, stop: int) -> tuple[str, int]:
    cursor = index + 1
    while cursor < stop and not lines[cursor]:
        cursor += 1
    if cursor < stop:
        return lines[cursor], cursor
    return "", index


def _parse_key_facts(lines: list[str], facts: list[Fact]) -> None:
    seen: set[str] = set()
    for index, line in enumerate(lines):
        field_name = LABEL_FIELDS.get(line)
        if field_name and field_name not in seen:
            value, _ = _next_value(lines, index, len(lines))
            if value:
                seen.add(field_name)
                facts.append(Fact(field_name=field_name, label=line, value=value, section="Key facts"))
        nav_match = NAV_DATE_RE.match(line)
        if nav_match and "nav" not in seen:
            value, _ = _next_value(lines, index, len(lines))
            if value:
                seen.add("nav")
                facts.append(
                    Fact("nav", "NAV", value, "Key facts", as_of=nav_match.group(1).strip())
                )
        risk_match = RISK_RE.match(line)
        if risk_match and "riskometer" not in seen:
            seen.add("riskometer")
            facts.append(Fact("riskometer", "Riskometer", line, "Key facts"))


def _parse_category(lines: list[str], facts: list[Fact], scheme_name: str) -> None:
    head = lines[:10]
    for index, line in enumerate(head):
        lock_match = LOCK_IN_RE.match(line)
        if lock_match:
            facts.append(
                Fact(
                    "lock_in",
                    "Lock-in period",
                    f"{lock_match.group(1).replace(' ', '')} (as stated on the scheme page)",
                    "Key facts",
                )
            )
    for index, line in enumerate(head):
        if line in TOP_LEVEL_CATEGORIES and index + 1 < len(head):
            facts.append(Fact("category", "Fund category", line, "Key facts"))
            facts.append(Fact("sub_category", "Sub-category", head[index + 1], "Key facts"))
            return


def _parse_fees_and_tax(lines: list[str], facts: list[Fact], prose: list[ProseBlock]) -> None:
    anchor = next((i for i, ln in enumerate(lines) if ln == "Exit load, stamp duty and tax"), None)
    if anchor is None:
        return
    stop = next(
        (i for i in range(anchor + 1, len(lines)) if lines[i] in ("Check past data", "Compare similar funds")),
        len(lines),
    )
    index = anchor + 1
    while index < stop:
        line = lines[index]
        if line == "Exit load":
            value, cursor = _next_value(lines, index, stop)
            if value:
                facts.append(Fact("exit_load", "Exit load", value, "Fees, exit load and tax"))
                index = cursor + 1
                continue
        elif line.startswith("Exit load of "):
            facts.append(Fact("exit_load", "Exit load", line, "Fees, exit load and tax"))
        elif line.startswith("Stamp duty on investment:"):
            value = line.split(":", 1)[1].strip()
            facts.append(Fact("stamp_duty", "Stamp duty on investment", value, "Fees, exit load and tax"))
        elif line == "Tax implication":
            value, _ = _next_value(lines, index, stop)
            facts.append(Fact("tax_implication", "Tax implication", value, "Fees, exit load and tax"))
        index += 1


def _parse_exit_load_history(lines: list[str], prose: list[ProseBlock]) -> None:
    anchor = next((i for i, ln in enumerate(lines) if ln == "Exit Load"), None)
    if anchor is None:
        return
    stop = next(
        (i for i in range(anchor + 1, len(lines)) if lines[i] == "Exit load, stamp duty and tax"),
        len(lines),
    )
    pairs: list[str] = []
    index = anchor + 1
    while index < stop - 1:
        if DATE_RE.match(lines[index]) and lines[index + 1].startswith("Exit load of "):
            pairs.append(f"{lines[index]}: {lines[index + 1]}")
            index += 2
        else:
            index += 1
    if pairs:
        prose.append(
            ProseBlock(
                heading="Exit load revision history",
                text="Historic exit load changes listed on the scheme page: " + " | ".join(pairs) + ".",
                section="Fees, exit load and tax",
            )
        )


def _parse_glossary(lines: list[str], prose: list[ProseBlock]) -> None:
    definitions = {
        "Annualised returns": "Average of the yearly returns of a mutual fund over a given period.",
        "Absolute returns": "The total return of a mutual fund over a given period.",
        "Expense ratio": "A fee payable to a mutual fund house for managing your mutual fund investments.",
        "Tax": "A percentage of your capital gains payable to the government upon exiting your mutual fund investments.",
        "Exit load": "A fee payable to a mutual fund house for exiting a fund (fully or partially) before the completion of a specified period from the date of investment.",
        "Stamp duty": "A form of tax payable for the purchase or sale of an asset or security.",
    }
    for index, line in enumerate(lines):
        if line in definitions and index + 1 < len(lines):
            body = lines[index + 1]
            if body.startswith("A ") or body.startswith("The "):
                prose.append(ProseBlock(f"Glossary: {line}", body, "Glossary"))


def _parse_about(lines: list[str], scheme_name: str, facts: list[Fact], prose: list[ProseBlock]) -> None:
    contact_fields = {
        "Fund house": "fund_house",
        "Rank (total assets)": "rank",
        "Total AUM": "total_aum",
        "Date of Incorporation": "incorporation_date",
        "Launch Date": "launch_date",
        "Custodian": "custodian",
        "Registrar & Transfer Agent": "rta",
    }
    for index, line in enumerate(lines):
        if line == f"About {scheme_name}" and index + 1 < len(lines):
            prose.append(ProseBlock("About the scheme", lines[index + 1], "About the scheme"))
            if index + 2 < len(lines) and not lines[index + 2].startswith("Investment Objective"):
                prose.append(ProseBlock("About the scheme (continued)", lines[index + 2], "About the scheme"))
        elif line == "Investment Objective" and index + 1 < len(lines):
            facts.append(Fact("objective", "Investment objective", lines[index + 1], "Objective"))
        benchmark_match = BENCHMARK_RE.match(line)
        if benchmark_match:
            facts.append(Fact("benchmark", "Benchmark", benchmark_match.group(1).strip(), "Benchmark"))
        contact_match = CONTACT_LABEL_RE.match(line)
        if contact_match:
            label, value = contact_match.group(1), contact_match.group(2).strip()
            if label == "Scheme Information Document(SID)":
                continue
            field_name = contact_fields.get(label)
            if field_name and not value:
                value, _ = _next_value(lines, index, len(lines))
            if field_name and value:
                facts.append(Fact(field_name, label, value, "Scheme details"))


def _parse_managers(lines: list[str], prose: list[ProseBlock]) -> None:
    anchor = next((i for i, ln in enumerate(lines) if ln == "Fund management"), None)
    if anchor is None:
        return
    stop = next(
        (i for i in range(anchor + 1, len(lines)) if lines[i].startswith("About HDFC")),
        len(lines),
    )
    index = anchor + 1
    while index < stop:
        if (
            MANAGER_INITIALS_RE.match(lines[index])
            and index + 1 < stop
            and TENURE_RE.match(lines[index + 2] if index + 2 < stop else "")
        ):
            name = lines[index + 1]
            tenure = lines[index + 2]
            details: list[str] = []
            probe = index + 3
            while probe < stop and not MANAGER_INITIALS_RE.match(lines[probe]) and not lines[probe].startswith("Also manages"):
                if lines[probe] in ("View details", "Education", "Experience"):
                    probe += 1
                    continue
                details.append(lines[probe])
                probe += 1
            bio = " ".join(details)
            prose.append(
                ProseBlock(
                    heading=f"Fund manager: {name}",
                    text=f"{name} ({tenure}). {bio}".strip(),
                    section="Fund management",
                )
            )
            index = probe
        else:
            index += 1


def _parse_holdings(lines: list[str], prose: list[ProseBlock]) -> None:
    anchor = next((i for i, ln in enumerate(lines) if ln.startswith("Holdings (")), None)
    if anchor is None:
        return
    total_match = re.search(r"\(\s*(\d+)\s*\)", lines[anchor])
    total = int(total_match.group(1)) if total_match else 0
    stop = next(
        (i for i in range(anchor + 1, len(lines)) if lines[i] in ("Expense ratio", "Exit load", "Exit load, stamp duty and tax")),
        len(lines),
    )
    rows: list[str] = []
    index = anchor + 1
    while index < stop and len(rows) < MAX_HOLDINGS:
        weight = lines[index]
        if weight.endswith("%") and re.match(r"^[\d,.]+%$", weight) and index >= 3:
            name = lines[index - 3]
            if name and not MONEY_RE.match(name) and len(name) > 1:
                rows.append(f"{name} ({weight})")
        index += 1
    if rows:
        prose.append(
            ProseBlock(
                heading=f"Top {len(rows)} holdings (page lists {total} holdings)",
                text="Largest holdings by weight as shown on the scheme page: " + "; ".join(rows) + ".",
                section="Holdings",
            )
        )


def detect_conflicts(document: Document) -> list[str]:
    payload = document.payload
    if not payload:
        return []
    notes: list[str] = []
    visible_risk = document.value("riskometer")
    payload_risk = payload.get("nfo_risk")
    if visible_risk and payload_risk and visible_risk.lower() not in payload_risk.lower():
        notes.append(
            f"riskometer: page shows '{visible_risk}', embedded payload says '{payload_risk}' (page wins)"
        )
    payload_manager = payload.get("fund_manager")
    if payload_manager and payload_manager not in "\n".join(document.lines):
        notes.append(
            f"fund_manager: embedded payload '{payload_manager}' absent from visible page (page wins)"
        )
    payload_launch = payload.get("launch_date")
    page_launch = document.value("launch_date")
    if payload_launch and page_launch and payload_launch.replace("-", " ") not in page_launch:
        notes.append(
            f"launch_date: embedded payload '{payload_launch}' differs from page '{page_launch}' (page wins)"
        )
    return notes


def parse_document(
    source: Source, html: str, title: str, retrieved_at: str | None = None
) -> Document:
    soup = BeautifulSoup(html, "lxml")
    display = source.display_name
    lines = content_window(block_leaf_lines(soup), display)
    document = Document(
        source=source,
        url=source.url,
        title=title,
        lines=lines,
        payload=extract_payload(html),
        links=extract_links(soup),
        retrieved_at=retrieved_at or now_iso(),
    )
    _parse_key_facts(lines, document.facts)
    _parse_category(lines, document.facts, display)
    _parse_fees_and_tax(lines, document.facts, document.prose)
    _parse_exit_load_history(lines, document.prose)
    _parse_glossary(lines, document.prose)
    _parse_about(lines, source.scheme_name, document.facts, document.prose)
    _parse_managers(lines, document.prose)
    _parse_holdings(lines, document.prose)
    document.conflicts = detect_conflicts(document)
    return document


def load_documents(*, refresh: bool = False, sleep: float = 1.2) -> list[Document]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    documents: list[Document] = []
    log: list[dict[str, Any]] = []

    for source in SOURCES:
        raw_path = RAW_DIR / f"{source.key}.html"
        if raw_path.exists() and not refresh:
            html = raw_path.read_text(encoding="utf-8")
            origin = "cache"
            retrieved_at = fetched_at(raw_path)
        else:
            html = fetch(source.url)
            raw_path.write_text(html, encoding="utf-8")
            origin = "network"
            retrieved_at = now_iso()
            time.sleep(sleep)

        soup = BeautifulSoup(html, "lxml")
        title = _normalise(soup.title.get_text()) if soup.title else source.scheme_name
        document = parse_document(source, html, title, retrieved_at=retrieved_at)

        (CLEAN_DIR / f"{source.key}.txt").write_text(
            "\n".join(
                [
                    f"SOURCE_URL: {source.url}",
                    f"SCHEME: {source.scheme_name}",
                    f"CATEGORY: {source.category}",
                    f"RETRIEVED_AT: {document.retrieved_at}",
                    "",
                    "== FACTS ==",
                    *[f"[{f.section}] {f.field_name} | {f.label} | {f.value}" for f in document.facts],
                    "",
                    "== PROSE ==",
                    *[f"[{p.section}] {p.heading}\n{p.text}" for p in document.prose],
                ]
            ),
            encoding="utf-8",
        )
        documents.append(document)
        log.append(
            {
                "key": source.key,
                "url": source.url,
                "origin": origin,
                "lines": len(document.lines),
                "facts": len(document.facts),
                "prose_blocks": len(document.prose),
                "conflicts": document.conflicts,
                "retrieved_at": document.retrieved_at,
            }
        )

    LOG_PATH.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    return documents


def documents_to_json(documents: Iterable[Document]) -> list[dict[str, Any]]:
    return [
        {
            "key": d.source.key,
            "scheme_name": d.scheme_name,
            "scheme_short": d.source.scheme_short,
            "category": d.source.category,
            "plan": d.source.plan,
            "url": d.url,
            "title": d.title,
            "retrieved_at": d.retrieved_at,
            "conflicts": d.conflicts,
            "links": d.links,
            "facts": [
                {"field": f.field_name, "label": f.label, "value": f.value, "section": f.section, "as_of": f.as_of}
                for f in d.facts
            ],
            "prose": [{"heading": p.heading, "text": p.text, "section": p.section} for p in d.prose],
        }
        for d in documents
    ]
