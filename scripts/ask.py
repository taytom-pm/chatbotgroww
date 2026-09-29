"""CLI: ask the facts-only assistant from a terminal."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mf_rag.pipeline import PipelineMissing, get_assistant
from mf_rag.sources import DISCLAIMER


def _use_utf8_console() -> None:
    """Make the terminal able to print the corpus.

    The chunk store contains the rupee sign (U+20B9) in every minimum-SIP and
    minimum-lump-sum fact. A Windows console defaults to cp1252, which cannot
    encode it, so `print(answer.render())` raised UnicodeEncodeError and killed
    the CLI on precisely the questions the demo is built around. Switch the
    console code page to UTF-8 and reconfigure the streams, degrading to
    replacement characters rather than crashing if either step is refused.
    """
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            ctypes.windll.kernel32.SetConsoleCP(65001)
        except Exception:
            pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def main() -> int:
    _use_utf8_console()
    parser = argparse.ArgumentParser(description="HDFC mutual fund facts-only FAQ assistant (CLI).")
    parser.add_argument("question", nargs="*", help="Question to answer. Omit to enter interactive mode.")
    parser.add_argument("--debug", action="store_true", help="Show retrieved chunks and scores.")
    args = parser.parse_args()

    try:
        assistant = get_assistant()
    except PipelineMissing as exc:
        print(exc, file=sys.stderr)
        return 1

    # Prior turns, for conversation memory. Only the interactive loop accumulates them; a
    # one-shot `ask.py "question"` has no earlier turns to remember.
    history: list[tuple[str, Any]] = []

    def respond(question: str) -> None:
        answer = assistant.ask(question, history=history)
        print()
        print(answer.render())
        if answer.notes:
            for note in answer.notes:
                print(f"[memory] {note}")
        if args.debug:
            print("\n-- retrieval trace --")
            print(f"intent={answer.intent} kind={answer.kind} refused={answer.refused}")
            for chunk in answer.chunks[:6]:
                print(
                    f"  rrf={chunk.rrf_score:.5f} vec={chunk.vector_score} bm25={chunk.bm25_score} "
                    f"[{chunk.kind}/{chunk.field_name or chunk.heading}] {chunk.scheme_short} :: {chunk.cite_text[:70]}"
                )
                print(f"      reasons={chunk.reasons}")
        history.append((question, answer))

    if args.question:
        respond(" ".join(args.question))
        return 0

    print("HDFC Mutual Fund FAQ assistant - facts only, no investment advice.")
    print(DISCLAIMER)
    print("Type 'exit' to quit.\n")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if question.lower() in ("exit", "quit", "q"):
            return 0
        if question:
            respond(question)


if __name__ == "__main__":
    raise SystemExit(main())
