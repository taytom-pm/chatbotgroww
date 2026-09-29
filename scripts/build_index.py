"""STAGE 1-4 driver: fetch, chunk, embed, store."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mf_rag.embed_store import build_index, load_chunks


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the ChromaDB index for the HDFC FAQ corpus.")
    parser.add_argument("--refresh", action="store_true", help="Re-fetch all source pages before indexing.")
    args = parser.parse_args()

    stats = build_index(refresh=args.refresh)
    chunks = load_chunks()
    kinds: dict[str, int] = {}
    for chunk in chunks:
        kinds[chunk.kind] = kinds.get(chunk.kind, 0) + 1

    print("Index built")
    print(f"  {stats.summary()}")
    print(f"  chunk kinds: {kinds}")
    print(f"  schemes: {len({c.scheme_name for c in chunks})}")
    print(f"  sections: {sorted({c.section for c in chunks})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
