"""Streamlit UI: facts-only HDFC mutual fund FAQ assistant."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mf_rag.config import EMBEDDING_MODEL, RETRIEVE_FINAL_K
from mf_rag.embed_store import build_index, index_exists
from mf_rag.pipeline import EXAMPLE_QUESTIONS, PipelineMissing, get_assistant
from mf_rag.sources import AMC_NAME, DISCLAIMER, SOURCES

st.set_page_config(page_title="HDFC MF Facts Assistant", page_icon="📄", layout="centered")

WELCOME = (
    "Hi! I answer published facts about five HDFC Mutual Fund direct-growth schemes - "
    "expense ratio, exit load, minimum SIP, ELSS lock-in, riskometer, benchmark, NAV and fund size. "
    "Every answer is copied from the scheme page and comes with a source link."
)


@st.cache_resource(show_spinner="Loading embedding model and vector index...")
def assistant():
    return get_assistant()


def ensure_index() -> bool:
    """Build the index if it is missing. Returns False if the app cannot answer.

    A deployed copy of this app is built from the repo and has no shell, so the old
    "run: python scripts/build_index.py" error was advice nobody could act on - it is
    what a hosted deploy showed until the index was committed. The committed index is
    the fast path; this is the fallback for a fresh clone, a wiped container or a
    store that cannot be read, so the app is never permanently dead.
    """
    if index_exists():
        return True

    with st.status(
        "No index found. Building it now - first run only, usually 1-3 minutes.",
        expanded=True,
    ) as status:
        status.write("Fetching 5 scheme pages, then chunking, embedding and storing in ChromaDB.")
        try:
            stats = build_index()
        except Exception as exc:  # network, DNS, disk quota, a broken store
            # st.status's API is update(label=..., state=...), and write() takes no state.
            # Assigning `status.state` would only set a Python attribute and never repaint.
            status.update(label=f"Index build failed: {type(exc).__name__}", state="error")
            st.error(f"Could not build the index: {type(exc).__name__}: {exc}")
            st.caption(
                "This needs network access to groww.in. Retry by reloading the page; "
                "locally, run: python scripts/build_index.py"
            )
            return False
        status.update(label=f"Index built: {stats.summary()}", state="complete")

    if not index_exists():
        st.error("The index was built but is still not readable. Please reload the page.")
        return False
    return True


def init_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []
    if "show_trace" not in st.session_state:
        st.session_state.show_trace = False
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None


def example_questions() -> None:
    """Clickable example questions (FR-8.2).

    A click cannot prefill `st.chat_input`: that widget takes a `key` but no `value`, and
    assigning to its session_state key does not submit it (verified on 1.64 - the value
    does not reach the return). So a clicked example is handed over through
    `pending_question` and consumed below as if the user had typed it.
    """
    for question in EXAMPLE_QUESTIONS:
        if st.button(question, key=f"example::{question[:40]}", use_container_width=True):
            st.session_state.pending_question = question


def sidebar() -> None:
    with st.sidebar:
        st.header("Scope")
        st.caption(f"AMC: {AMC_NAME} - 5 schemes, Direct Growth plans")
        for source in SOURCES:
            st.markdown(f"- **{source.scheme_short}** - {source.category}")
        st.divider()
        st.subheader("Pipeline")
        st.code(
            "ingest -> chunk -> embed\n-> ChromaDB -> hybrid\nretrieve -> extractive answer",
            language="text",
        )
        st.caption(f"Embeddings: {EMBEDDING_MODEL}")
        st.caption(f"Top-k retrieved per question: {RETRIEVE_FINAL_K}")
        st.divider()
        st.subheader("Guardrails")
        st.markdown(
            "- Facts only, no buy/sell advice\n"
            "- No return calculations or comparisons\n"
            "- One source link per answer\n"
            "- Max 3 sentences per answer\n"
            "- Rejects PAN, Aadhaar, folio, OTP, phone, email"
        )
        st.divider()
        st.caption(DISCLAIMER)


def render_answer(answer, show_trace: bool) -> None:
    if answer.kind == "advice_refusal":
        st.info(answer.text)
    elif answer.kind in ("pii", "compute_refusal", "performance_refusal"):
        st.warning(answer.text)
    else:
        st.markdown(answer.text)

    if answer.citations:
        for url in answer.citations:
            st.link_button("Source", url)
    if answer.retrieved_at:
        st.caption(f"Last updated from sources: {answer.retrieved_at}")
    if show_trace and answer.chunks:
        with st.expander("Retrieval trace"):
            st.caption(
                f"intent={answer.intent} | kind={answer.kind} | refused={answer.refused}"
            )
            for note in answer.notes:
                st.caption(f"memory: {note}")
            for chunk in answer.chunks:
                st.markdown(
                    f"- `rrf={chunk.rrf_score:.5f}` `dense={chunk.vector_score}` "
                    f"`bm25={chunk.bm25_score}` **[{chunk.kind}/{chunk.field_name or chunk.heading}]** "
                    f"{chunk.scheme_short} - {chunk.cite_text[:90]}"
                )


def main() -> None:
    init_state()
    sidebar()

    st.title("HDFC Mutual Fund facts assistant")
    st.caption(DISCLAIMER)

    if not ensure_index():
        return

    try:
        bot = assistant()
    except PipelineMissing as exc:
        st.error(str(exc))
        st.code("python scripts/build_index.py", language="bash")
        return

    # Settings must be evaluated BEFORE the transcript is rendered. Streamlit reruns
    # top-to-bottom, so an expander placed at the bottom always lags one run behind:
    # ticking "Show retrieval trace" would leave the answers above it untraced until
    # the *next* question was asked.
    with st.expander("Settings"):
        st.session_state.show_trace = st.checkbox(
            "Show retrieval trace", value=st.session_state.show_trace
        )
        if st.session_state.history and st.button("Clear conversation"):
            st.session_state.history = []
            st.rerun()

    if not st.session_state.history:
        st.info(WELCOME)
        st.markdown("**Try one of these:**")
        example_questions()

    for question, answer in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            render_answer(answer, st.session_state.show_trace)

    typed = st.chat_input("Ask a published fact about these five schemes")
    clicked = st.session_state.pending_question
    asked = typed or clicked
    if asked:
        # Clear before asking so a rerun (trace toggle, clear chat) cannot re-answer the
        # same example question.
        st.session_state.pending_question = None
        with st.chat_message("user"):
            st.markdown(asked)
        with st.chat_message("assistant"):
            with st.spinner("Searching the corpus..."):
                # The current turn is appended afterwards, so this is exactly the prior
                # conversation. That is what lets "what about its lock-in?" resolve the
                # fund named a moment ago.
                answer = bot.ask(asked, history=st.session_state.history)
            render_answer(answer, st.session_state.show_trace)
        st.session_state.history.append((asked, answer))


if __name__ == "__main__":
    main()
