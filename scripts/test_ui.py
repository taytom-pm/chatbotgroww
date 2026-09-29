"""UI smoke test using Streamlit's AppTest: renders the app and asks a question.

Run directly:  python scripts/test_ui.py
Exits non-zero on the first failed assertion.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest

from mf_rag.pipeline import EXAMPLE_QUESTIONS
from mf_rag.pipeline import ask as pipeline_ask

ANSWER_URL = "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth"


def _text(at) -> str:
    # Button labels are included because the FR-8.2 example questions are rendered as
    # clickable buttons rather than markdown, so they appear in no markdown or caption.
    return " ".join(
        [m.value for m in at.markdown]
        + [c.value for c in at.caption]
        + [i.value for i in at.info]
        + [w.value for w in at.warning]
        + [b.label for b in at.button]
    )


def run() -> None:
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    at.run()

    # 1. cold boot
    assert not at.exception, f"cold boot raised: {[str(e.value) for e in at.exception]}"
    assert len(at.chat_input) == 1, f"expected exactly 1 chat_input, found {len(at.chat_input)}"
    assert len(at.title) == 1, f"expected 1 title, found {len(at.title)}"
    assert "Facts-only" in _text(at) or "no investment advice" in _text(at).lower(), "missing facts-only note"
    for question in EXAMPLE_QUESTIONS:
        assert question in _text(at), f"welcome screen does not list example question: {question!r}"

    # 1b. FR-8.2: the examples must be clickable, not just listed as text. Uses its own
    # AppTest so the extra answer does not pollute the "exactly 1 Source link" checks below.
    at_click = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    labels = [b.label for b in at_click.button]
    for question in EXAMPLE_QUESTIONS:
        assert question in labels, f"example question is not a clickable button: {question!r}"
    at_click.button[labels.index(EXAMPLE_QUESTIONS[0])].click().run()
    assert not at_click.exception, f"clicking an example raised: {[str(e.value) for e in at_click.exception]}"
    # The welcome block is drawn before the pending question is consumed, so on this run
    # the button label and the chat message both contain the text. The transcript itself
    # must hold it exactly once - a stale pending_question would show it again on rerun.
    assert sum(1 for m in at_click.markdown if m.value == EXAMPLE_QUESTIONS[0]) == 1, (
        "clicking an example must add it to the transcript exactly once"
    )
    assert len([c for c in at_click.get("link_button") if c.label == "Source"]) == 1, (
        "clicking an example must produce exactly one cited answer"
    )

    # 2. ask a fact
    at.chat_input[0].set_value("What is the expense ratio of HDFC Large Cap Fund?").run()
    assert not at.exception, f"asking raised: {[str(e.value) for e in at.exception]}"
    assert "1.03%" in _text(at), "answer value 1.03% not rendered"
    assert "Last updated from sources" in _text(at), "missing retrieval timestamp caption"

    # 3. the Source link button, and the URL it points at
    link_buttons = at.get("link_button")
    assert len(link_buttons) == 1, f"expected exactly 1 link_button, found {len(link_buttons)}"
    assert link_buttons[0].label == "Source", link_buttons[0].label
    href = getattr(link_buttons[0], "href", None) or str(link_buttons[0].url or "")
    assert ANSWER_URL in href, f"Source button points at {href!r}"

    # 4. retrieval trace is opt-in (a "Settings" expander is always present)
    def trace_expander(test) -> list:
        return [e for e in test.get("expander") if "Retrieval trace" in (e.label or "")]

    assert not trace_expander(at), "trace expander should be hidden until requested"
    at.checkbox[0].check().run()
    assert not at.exception, f"trace toggle raised: {[str(e.value) for e in at.exception]}"
    assert trace_expander(at), "no 'Retrieval trace' expander after enabling the checkbox"
    assert any("rrf=" in m.value for m in at.markdown), "trace did not render rrf scores"
    assert any("dense=" in m.value for m in at.markdown), "trace did not render dense scores"
    assert any("bm25=" in m.value for m in at.markdown), "trace did not render bm25 scores"

    # 5. I-5 rendering contract: a PII refusal shows a warning and NO timestamp
    at2 = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    at2.run()
    at2.chat_input[0].set_value("My PAN is ABCDE1234F, check my returns").run()
    assert not at2.exception, f"PII path raised: {[str(e.value) for e in at2.exception]}"
    assert len(at2.warning) == 1, f"expected 1 warning for a PII refusal, found {len(at2.warning)}"
    assert "Last updated from sources" not in _text(at2), "PII refusal must not show a timestamp"
    assert len(at2.get("link_button")) == 1, "PII refusal must still cite the privacy policy"

    # 6. an advice request is refused, not answered. The welcome screen is also an
    #    st.info box, so assert the refusal text itself is present rather than counting.
    at3 = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    at3.run()
    at3.chat_input[0].set_value("Should I buy HDFC Small Cap Fund?").run()
    assert not at3.exception, f"advice path raised: {[str(e.value) for e in at3.exception]}"
    refusal_text = pipeline_ask("Should I buy HDFC Small Cap Fund?").text
    assert any(refusal_text in i.value for i in at3.info), "advice refusal not shown in an info box"
    assert not at3.warning, "an advice request must be an info, not a warning"

    # 7. clearing the conversation empties the transcript
    at.chat_input[0].set_value("What is the NAV of HDFC Small Cap Fund?").run()
    assert len(at.chat_message) >= 4, f"expected 4 chat messages, found {len(at.chat_message)}"
    clear = [b for b in at.button if b.label == "Clear conversation"][0]
    clear.click().run()
    assert not at.exception, f"clear raised: {[str(e.value) for e in at.exception]}"
    assert len(at.chat_message) == 0, "Clear conversation did not empty the transcript"

    print("OK - all 8 UI smoke checks passed (cold boot, clickable examples, fact answer, "
          "source link, trace toggle, PII, advice refusal, clear)")


if __name__ == "__main__":
    run()
