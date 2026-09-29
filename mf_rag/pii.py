"""PII guard. Rejects queries containing identifiers the assistant must not accept or store."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

PAN_RE = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
# The lookarounds stop this shape from matching a 4-digit group that is merely part
# of a longer 4-4-4-4 run, which is how a 16-digit card number used to be partially
# consumed as an Aadhaar and left residue behind after scrubbing.
AADHAAR_RE = re.compile(r"(?<!\d{4}\s)(?<!\d)[2-9]\d{3}\s?\d{4}\s?\d{4}(?!\s\d{4})(?!\d)")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\d)")
OTP_RE = re.compile(r"\b(?:otp|verification code|one time password)\b", re.I)
CARD_RE = re.compile(r"\b(?:\d{4}[\s-]?){3}\d{4}\b")
ACCOUNT_RE = re.compile(
    r"\b(?:account\s*(?:no|number)|acct\s*no|folio\s*(?:no|number)|demat\s*(?:no|number))\b", re.I
)
IFSC_RE = re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b")
PASSPORT_RE = re.compile(r"\b[A-PR-WY][0-9]{7}\b")

REDACTED = "[redacted]"

REFUSAL = (
    "I can't accept personal identifiers. Please don't share PAN, Aadhaar, account or folio "
    "numbers, OTPs, IFSC, card numbers, email addresses or phone numbers. "
    "This assistant only answers published fund facts and never needs personal data, so nothing "
    "from that message was stored - see {link} for how personal data is handled."
)

# Order matters for two reasons: `scan` reports kinds in this order, and `scrub`
# builds one alternation from it, so a more specific shape must precede a looser
# one that could otherwise consume a slice of it. Card (16 digits) precedes
# Aadhaar (12 digits in 4-4-4 groups) for exactly that reason.
RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("PAN", PAN_RE),
    ("card number", CARD_RE),
    ("Aadhaar", AADHAAR_RE),
    ("email address", EMAIL_RE),
    ("phone number", PHONE_RE),
    ("OTP", OTP_RE),
    ("account or folio number", ACCOUNT_RE),
    ("IFSC", IFSC_RE),
    ("passport number", PASSPORT_RE),
)

# One pass, so no rule can eat another's input and leave a partial tail behind.
SCRUB_RE = re.compile("|".join(f"(?:{pattern.pattern})" for _name, pattern in RULES))


@dataclass
class PiiResult:
    is_pii: bool
    kinds: list[str] = field(default_factory=list)

    @property
    def message(self) -> str:
        from .sources import EDUCATIONAL_LINKS

        return REFUSAL.format(link=EDUCATIONAL_LINKS["privacy"])


def scan(text: str) -> PiiResult:
    kinds = [name for name, pattern in RULES if pattern.search(text)]
    return PiiResult(is_pii=bool(kinds), kinds=kinds)


def scrub(text: str) -> str:
    """Replace every identifier in one pass so no rule can partially consume another's match."""
    return SCRUB_RE.sub(REDACTED, text)
