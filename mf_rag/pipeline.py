"""Pipeline wiring: load the persisted index and expose a ready-to-use assistant."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from .answer import Answer, FAQAssistant
from .embed_store import index_exists, load_chunks
from .retrieve import HybridRetriever

EXAMPLE_QUESTIONS = (
    "What is the expense ratio of HDFC Flexi Cap Fund?",
    "Is there a lock-in period on HDFC ELSS Tax Saver Fund?",
    "What is the minimum SIP for HDFC Small Cap Fund?",
)


class PipelineMissing(Exception):
    pass


@lru_cache(maxsize=1)
def get_assistant() -> FAQAssistant:
    if not index_exists():
        raise PipelineMissing(
            "No index found. Run: python scripts/build_index.py"
        )
    return FAQAssistant(HybridRetriever(load_chunks()))


def ask(query: str, history: list[tuple[str, Any]] | None = None) -> Answer:
    """Answer `query`. `history` is prior (question, answer) turns; when it is supplied,
    a follow-up that names no fund can still be resolved to one. Omit it and behaviour is
    exactly as before."""
    return get_assistant().ask(query, history=history)
