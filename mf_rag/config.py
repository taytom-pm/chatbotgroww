"""Paths and tunable pipeline constants."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
CHROMA_DIR = DATA_DIR / "chroma"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
LOG_PATH = DATA_DIR / "ingest_log.json"

DELIVERABLES_DIR = ROOT / "deliverables"

CHROMA_COLLECTION = "hdfc_mutual_funds"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

TOKENIZER_MODEL = EMBEDDING_MODEL
MAX_EMBEDDING_TOKENS = 256
CHUNK_TOKEN_TARGET = 180
CHUNK_TOKEN_HARD_MAX = 250
CHUNK_TOKEN_OVERLAP = 40

RETRIEVE_VECTOR_K = 8
RETRIEVE_BM25_K = 8
RETRIEVE_FINAL_K = 4
RRF_K = 60

MAX_ANSWER_SENTENCES = 3
MAX_CONTEXT_CHUNKS = 6
MEMORY_TURNS = 10  # Q&A turns of conversation memory available to follow-up retrieval
