"""Optional LLM client, used ONLY for evaluation tooling. Never for answering.

The assistant itself has no language model (architecture.md, decision D1): answers are
extractive sentences copied verbatim from a retrieved chunk, so a wrong answer is a test
failure rather than a hallucination. This module exists so the *evaluation* side of the
project can be stronger without touching that guarantee:

  * generating rephrasings of the benchmark queries, to test whether retrieval survives
    wording the author did not write.

Nothing in the answer path imports this module. `mf_rag/answer.py` and `mf_rag/pipeline.py`
have no dependency on it, and adding one would be a design regression, not a feature.

Credentials come from `.env` (gitignored) or the process environment. The project works
fully with no key at all: `scripts/benchmark_paraphrase.py` reads a committed cache of
generated paraphrases, so the evaluation is reproducible and the test suite never needs
network access or a key.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .config import ROOT

ENV_PATH = ROOT / ".env"

# Providers in the order they are probed. The first with a key present wins, so setting
# one variable is enough. OpenAI-compatible covers OpenAI, Groq, Azure, LM Studio and
# Ollama, which is why a single branch handles most of the list.
_PROVIDER_ORDER = ("openai", "anthropic", "gemini", "groq", "local")


class LlmUnavailable(RuntimeError):
    """No usable provider was configured."""


@dataclass(frozen=True)
class LlmConfig:
    provider: str
    api_key: str
    model: str
    base_url: str
    timeout: int = 60


def load_env(path: Path = ENV_PATH) -> dict[str, str]:
    """Read `.env` into a dict. Missing or unreadable file yields {}.

    Deliberately not `load_dotenv()`: that mutates `os.environ` process-wide, and this
    project has no other consumer of the environment. A plain read keeps the blast
    radius to the caller.
    """
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip("'\"")
    return values


def resolve_config(env: dict[str, str] | None = None) -> LlmConfig:
    """Pick a provider from the environment. Raises if none is configured."""
    merged = {**load_env(), **dict(os.environ), **(env or {})}

    for provider in _PROVIDER_ORDER:
        if provider == "openai" and merged.get("OPENAI_API_KEY"):
            return LlmConfig("openai", merged["OPENAI_API_KEY"], merged.get("OPENAI_MODEL", "gpt-4o-mini"), "https://api.openai.com/v1")
        if provider == "groq" and merged.get("GROQ_API_KEY"):
            return LlmConfig("openai", merged["GROQ_API_KEY"], merged.get("GROQ_MODEL", "llama-3.1-8b-instant"), "https://api.groq.com/openai/v1")
        if provider == "local" and merged.get("LOCAL_LLM_MODEL"):
            # LOCAL_LLM_BASE_URL ships with a default, so it cannot signal "configured".
            # Requiring the model name makes opting in explicit and avoids reporting a
            # provider as available when no local server is actually listening.
            return LlmConfig("openai", merged.get("LOCAL_LLM_API_KEY", "not-needed"), merged["LOCAL_LLM_MODEL"], merged.get("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1").rstrip("/"))
        if provider == "anthropic" and merged.get("ANTHROPIC_API_KEY"):
            return LlmConfig("anthropic", merged["ANTHROPIC_API_KEY"], merged.get("ANTHROPIC_MODEL", "claude-sonnet-5"), "https://api.anthropic.com/v1")
        if provider == "gemini" and merged.get("GEMINI_API_KEY"):
            return LlmConfig("gemini", merged["GEMINI_API_KEY"], merged.get("GEMINI_MODEL", "gemini-2.0-flash"), "https://generativelanguage.googleapis.com/v1beta")

    raise LlmUnavailable(
        "No LLM provider configured. Set one of OPENAI_API_KEY, ANTHROPIC_API_KEY, "
        "GEMINI_API_KEY, GROQ_API_KEY, or LOCAL_LLM_MODEL for a local server, in .env. "
        "This is only needed to regenerate the paraphrase cache; the shipped assistant "
        "and the test suite do not need it."
    )


def _post(url: str, payload: dict, headers: dict[str, str], timeout: int) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:400]
        raise RuntimeError(f"LLM request failed: HTTP {exc.code} {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM request failed: {exc.reason}") from exc


def complete(prompt: str, *, config: LlmConfig | None = None, max_tokens: int = 700) -> str:
    """Single-turn completion. Temperature is 0 so the generated cache is as stable as the
    provider allows; it is cached on disk regardless, so evaluation never re-bills."""
    cfg = config or resolve_config()

    if cfg.provider == "openai":
        data = _post(
            f"{cfg.base_url}/chat/completions",
            {
                "model": cfg.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": max_tokens,
            },
            {"Authorization": f"Bearer {cfg.api_key}"},
            cfg.timeout,
        )
        return data["choices"][0]["message"]["content"].strip()

    if cfg.provider == "anthropic":
        data = _post(
            f"{cfg.base_url}/messages",
            {
                "model": cfg.model,
                "max_tokens": max_tokens,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
            },
            {"x-api-key": cfg.api_key, "anthropic-version": "2023-06-01"},
            cfg.timeout,
        )
        return "".join(block.get("text", "") for block in data.get("content", [])).strip()

    if cfg.provider == "gemini":
        url = f"{cfg.base_url}/models/{cfg.model}:generateContent?key={urllib.parse.quote(cfg.api_key)}"
        data = _post(
            url,
            {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0, "maxOutputTokens": max_tokens},
            },
            {},
            cfg.timeout,
        )
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()

    raise LlmUnavailable(f"unhandled provider: {cfg.provider}")


def parse_json_list(raw: str) -> list[str]:
    """Pull a JSON array of strings out of a model response.

    Models fence their JSON or prepend a sentence of commentary even when told not to, so
    this slices the first balanced array rather than trusting `raw` to be pure JSON.
    """
    start = raw.find("[")
    if start == -1:
        raise ValueError(f"no JSON array in response: {raw[:200]!r}")
    depth = 0
    for index in range(start, len(raw)):
        if raw[index] == "[":
            depth += 1
        elif raw[index] == "]":
            depth -= 1
            if depth == 0:
                parsed = json.loads(raw[start : index + 1])
                return [str(item).strip() for item in parsed if str(item).strip()]
    raise ValueError(f"unterminated JSON array: {raw[:200]!r}")
