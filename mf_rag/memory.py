"""Conversation memory for retrieval: resolve follow-up referents from recent turns.

The assistant answers one question at a time. Without any memory, a follow-up that omits
the fund name - "what about its lock-in?", "and the exit load?" - names no scheme, so
`detect_scheme` returns None and the request is answered `out_of_scope`. That is a
frequent, fixable failure, and it is a *retrieval* problem rather than a generation one.

So memory is used for exactly one thing: deciding which scheme's chunks retrieval is
allowed to consider. It never contributes text to the answer. The answer is still a
verbatim span from a retrieved chunk, still one citation, still at most three sentences
(architecture.md decision D1).

Two properties this module is responsible for:

1. **No identifiers are ever carried forward.** A PAN or phone number typed two turns ago
   must not re-enter a retrieval query or an answer. Remembered text is run through
   `pii.scrub` before it is examined, and a turn that scrubs to nothing is skipped.

2. **Memory cannot widen the scope gate.** Resolution only chooses a filter; the
   out-of-scope lexical check still runs on the user's actual words, so "what is the
   weather in Mumbai" is still refused after a long ELSS discussion.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from . import pii as pii_module
from .config import MEMORY_TURNS

# Words that mark a question as referring back to something already discussed. A query
# only borrows a scheme from history when it is clearly a follow-up, so an unrelated
# question never silently inherits the last topic.
FOLLOW_UP_PHRASES = (
    "what about",
    "how about",
    "and the",
    "and its",
    "same for",
)

# Single words that only count when they stand alone. A plain substring test would match
# "it" inside "with", "exit" or "unit", which is why these are compiled with boundaries.
FOLLOW_UP_WORDS = (
    "it",
    "its",
    "that",
    "this",
    "these",
    "those",
    "them",
    "they",
    "he",
    "she",
    "same",
    "there",
)

FOLLOW_UP_MARKERS = FOLLOW_UP_PHRASES + FOLLOW_UP_WORDS

_FOLLOW_UP_WORD_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(word) for word in FOLLOW_UP_WORDS) + r")\b"
)

# A question that ranges over the whole corpus is not a follow-up, however it is phrased.
# "the fund with the highest returns" contains no pronoun, but the moment a pronoun is
# allowed to pull in a remembered scheme, a comparative like "which of them has the lowest
# exit load" would silently collapse to one fund and be answered confidently from it. These
# words veto memory outright. Queries that also name a fund never reach here at all, because
# `ask` only consults memory when the query itself resolved no scheme.
GLOBAL_QUERY_RE = re.compile(
    r"\b(?:all|every|each|any|compare|versus|vs|difference|better|worse|"
    r"best|worst|highest|lowest|cheapest|shortest|longest|most|least|top)\b"
)


@dataclass(frozen=True)
class MemoryResolution:
    """Outcome of looking for a referent in recent turns."""

    scheme: str | None = None
    turn_index: int | None = None
    considered: int = 0
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def used(self) -> bool:
        return self.scheme is not None


def _turn_texts(turn: Any) -> list[str]:
    """Every scrap of text in one history turn, so the scheme can be recovered from either
    side of the exchange.

    A user can name the fund in the question ("expense ratio of the HDFC ELSS fund?") or the
    assistant can be the only place the fund was ever named ("expense ratio?" -> an answer
    headed "HDFC ELSS Tax Saver Fund Direct Plan Growth"). Reading only the question would
    miss the second case, which is a very common way to start a conversation.

    Accepts a (question, answer) pair, a bare question string, or an object exposing
    `.question`/`.text`/`.query`, so callers can pass their own history shape unadapted.
    The `Answer` object the app already stores is covered by the `.text` branch.
    """
    if isinstance(turn, str):
        return [turn]
    if isinstance(turn, (tuple, list)):
        parts: list[str] = []
        for item in turn:
            parts.extend(_turn_texts(item))
        return parts
    for attribute in ("question", "text", "query"):
        value = getattr(turn, attribute, None)
        if isinstance(value, str) and value:
            return [value]
    return []


def is_follow_up(query: str) -> bool:
    """True when the query looks like it refers back to earlier context.

    A pronoun is the strong signal. Two extra allowances, because real follow-ups are not
    always well-formed: a very short question ("lock-in?") and one that asks about the same
    field as the previous turn are both treated as follow-ups.

    A comparative or corpus-wide question is never a follow-up - see `GLOBAL_QUERY_RE`.
    """
    if GLOBAL_QUERY_RE.search(query.lower()):
        return False
    lowered = f" {query.lower().strip()} "
    if _FOLLOW_UP_WORD_RE.search(lowered):
        return True
    if any(phrase in lowered for phrase in FOLLOW_UP_PHRASES):
        return True
    words = query.split()
    return len(words) <= 3


def recent_turns(history: Sequence[Any] | None, window: int = MEMORY_TURNS) -> list[Any]:
    """The last `window` turns, oldest first. This is the entire memory."""
    if not history:
        return []
    return list(history)[-window:]


def resolve(query: str, history: Sequence[Any] | None, *, window: int = MEMORY_TURNS) -> MemoryResolution:
    """Find the scheme a follow-up refers to, newest turn first.

    Deliberately does not import `answer.detect_scheme`; `answer` imports this module, and a
    cycle would be a runtime error rather than a test failure. The scheme vocabulary is
    taken from `sources.SOURCES` instead, which is the same source of truth.
    """
    from .sources import SOURCES

    turns = recent_turns(history, window)
    if not turns:
        return MemoryResolution(notes=("no earlier turns to consider",))
    if not is_follow_up(query):
        return MemoryResolution(considered=len(turns), notes=("query does not read as a follow-up",))

    names: list[tuple[str, str]] = []
    for source in SOURCES:
        candidates = {source.scheme_short, source.scheme_name, source.display_name}
        if source.page_name:
            candidates.add(source.page_name)
        for alias in candidates:
            if alias:
                names.append((alias.lower(), source.scheme_short))
    # Longest alias first so "HDFC Large Cap Fund" wins over "Large Cap".
    names.sort(key=lambda pair: len(pair[0]), reverse=True)

    for index in range(len(turns) - 1, -1, -1):
        # PII is stripped from remembered text before it is even pattern-matched, so an
        # identifier can never influence which fund is chosen, let alone reach retrieval.
        remembered = " ".join(pii_module.scrub(part) for part in _turn_texts(turns[index]))
        if not remembered.strip():
            continue  # this turn was nothing but identifiers
        lowered = remembered.lower()
        for alias, scheme_short in names:
            if alias in lowered:
                return MemoryResolution(
                    scheme=scheme_short,
                    turn_index=index,
                    considered=len(turns),
                    notes=(f"scheme taken from turn {index + 1} of the last {len(turns)}",),
                )

    return MemoryResolution(considered=len(turns), notes=("no earlier turn named a scheme",))
