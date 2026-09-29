"""Source registry for the HDFC Mutual Fund facts-only RAG corpus."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Source:
    key: str
    scheme_name: str
    scheme_short: str
    category: str
    plan: str
    url: str
    page_name: str = ""
    isin: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def display_name(self) -> str:
        return self.page_name or self.scheme_name


AMC_NAME = "HDFC Mutual Fund"
AMC_SHORT = "HDFC"
DISCLAIMER = (
    "Facts-only. No investment advice. This assistant reproduces published facts "
    "from public fund pages and does not recommend, rate, or compare funds for purchase."
)
EDUCATIONAL_LINKS = {
    "statements": "https://groww.in/help",
    "elss": "https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth",
    "riskometer": "https://www.sebi.gov.in/",
    "terms": "https://groww.in/mutual-funds",
    "privacy": "https://groww.in/privacy-policy",
}

SOURCES: tuple[Source, ...] = (
    Source(
        key="hdfc_large_cap",
        scheme_name="HDFC Large Cap Fund Direct Growth",
        scheme_short="Large Cap",
        category="Equity Large Cap",
        plan="Direct Growth",
        url="https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
        tags=("large cap", "equity", "bluechip", "top 100"),
    ),
    Source(
        key="hdfc_flexi_cap",
        scheme_name="HDFC Equity Fund Direct Growth",
        scheme_short="Flexi Cap",
        category="Equity Flexi Cap",
        plan="Direct Growth",
        url="https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth",
        page_name="HDFC Flexi Cap Direct Plan Growth",
        tags=("flexi cap", "equity fund", "multi cap", "hdfc equity fund"),
    ),
    Source(
        key="hdfc_elss",
        scheme_name="HDFC ELSS Tax Saver Fund Direct Plan Growth",
        scheme_short="ELSS",
        category="Equity ELSS",
        plan="Direct Plan Growth",
        url="https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth",
        tags=("elss", "80c", "80d", "tax saver", "lock in", "3 year"),
    ),
    Source(
        key="hdfc_small_cap",
        scheme_name="HDFC Small Cap Fund Direct Growth",
        scheme_short="Small Cap",
        category="Equity Small Cap",
        plan="Direct Growth",
        url="https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth",
        tags=("small cap", "equity", "sme", "high risk"),
    ),
    Source(
        key="hdfc_balanced_advantage",
        scheme_name="HDFC Balanced Advantage Fund Direct Growth",
        scheme_short="Balanced Advantage",
        category="Hybrid Balanced Advantage",
        plan="Direct Growth",
        url="https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth",
        tags=("balanced advantage", "hybrid", "equity savings", "arbitrage"),
    ),
)

SOURCES_BY_KEY: dict[str, Source] = {s.key: s for s in SOURCES}
