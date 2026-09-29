"""Generate the source-list deliverables (CSV + Markdown) from the source registry."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mf_rag.config import DELIVERABLES_DIR
from mf_rag.sources import AMC_NAME, DISCLAIMER, EDUCATIONAL_LINKS, SOURCES

# Labels for EDUCATIONAL_LINKS. The URLs themselves are read from the registry so
# SOURCES.md cannot drift away from what the assistant actually cites.
EDUCATIONAL_LABELS = {
    "statements": "Groww help centre (statements, tax documents)",
    "terms": "Groww mutual funds index",
    "riskometer": "SEBI (regulator, riskometer context)",
    "elss": "HDFC ELSS Tax Saver Fund page (cited for ELSS guidance only)",
    "privacy": "Groww privacy policy (cited only for the PII refusal)",
}

COLUMNS = (
    "key",
    "scheme_name",
    "scheme_short",
    "category",
    "plan",
    "page_name_on_site",
    "url",
    "publisher",
    "access_type",
    "content_type",
)


def main() -> int:
    DELIVERABLES_DIR.mkdir(parents=True, exist_ok=True)

    with (DELIVERABLES_DIR / "SOURCES.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for source in SOURCES:
            writer.writerow(
                {
                    "key": source.key,
                    "scheme_name": source.scheme_name,
                    "scheme_short": source.scheme_short,
                    "category": source.category,
                    "plan": source.plan,
                    "page_name_on_site": source.display_name,
                    "url": source.url,
                    "publisher": "Groww (public scheme page)",
                    "access_type": "public, no login",
                    "content_type": "scheme factsheet-style page (fees, riskometer, benchmark, holdings)",
                }
            )

    lines = [
        "# Source list",
        "",
        f"AMC: **{AMC_NAME}** - 5 schemes, all Direct Growth plans.",
        "",
        "All five sources are public scheme pages. No login, no paywall, no third-party blogs, no",
        "brokerage or back-office screenshots. The assistant cites the exact page an answer came from.",
        "",
        "| # | Scheme | Category | Plan | Name shown on page | URL |",
        "|---|--------|----------|------|--------------------|-----|",
    ]
    for index, source in enumerate(SOURCES, start=1):
        lines.append(
            f"| {index} | {source.scheme_name} | {source.category} | {source.plan} | "
            f"{source.display_name} | {source.url} |"
        )

    lines += [
        "",
        "## Secondary / educational links",
        "",
        "These are cited only for guidance-type answers (never for a fund fact):",
        "",
    ]
    primary_urls = {source.url for source in SOURCES}
    reused: list[str] = []
    for name, url in sorted(EDUCATIONAL_LINKS.items()):
        label = EDUCATIONAL_LABELS.get(name, name)
        if url in primary_urls:
            # EDUCATIONAL_LINKS also points at one scheme page, for ELSS guidance.
            # It is a primary source, so listing it here would contradict the rule above.
            reused.append(f"- {label}: {url}")
            continue
        lines.append(f"- {label}: {url}")
    if reused:
        lines += [
            "",
            "The following are also registered as guidance links, but they are **primary** scheme",
            "pages (they appear in the table above), so a fund fact may legitimately cite them:",
            "",
        ]
        lines += reused
    lines += [
        "",
        "## Not used",
        "",
        "- No screenshots of any app back-end.",
        "- No third-party blogs, aggregators or forums as factual sources.",
        "- No PII of any kind is requested, received or stored.",
        "",
        "---",
        "",
        DISCLAIMER,
        "",
    ]
    (DELIVERABLES_DIR / "SOURCES.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote SOURCES.csv and SOURCES.md for {len(SOURCES)} sources")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
