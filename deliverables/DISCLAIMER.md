# Disclaimer text used in the app

## Snippet shown in the UI

**Facts-only. No investment advice. This assistant reproduces published facts from public fund pages and does not recommend, rate, or compare funds for purchase.**

Rendered in `app.py` from `mf_rag.sources.DISCLAIMER`, shown under the page title and in the sidebar.

## Full text

This assistant answers factual questions about five HDFC Mutual Fund direct-growth schemes using only
publicly published scheme pages.

- It does not recommend, rate, rank or compare funds, and does not answer "should I buy, sell, switch
  or hold" questions. For suitability, consult a SEBI-registered investment adviser.
- It does not calculate, project or compare returns or future values. Where performance is asked
  about, it points to the scheme page and the official factsheet.
- Answers are copied from the cited page and capped at three sentences. Every answer carries one
  source link and a "Last updated from sources" timestamp.
- It does not accept or store PAN, Aadhaar, account or folio numbers, OTPs, IFSC, card numbers, email
  addresses or phone numbers. Queries containing these are refused and not stored.
- Figures reflect the page as at the timestamp shown and change daily (NAV, AUM, expense ratio) or
  periodically (exit load, benchmark, riskometer). Always confirm on the linked page before acting.

Mutual fund investments are subject to market risks. Read all scheme related documents carefully.
