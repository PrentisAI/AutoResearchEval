"""OpenRouter teacher-LLM client implementing ``reconstruct.base.LLMClient`` (§5, §11).

A dependency-light client (uses ``requests``) for OpenRouter's OpenAI-compatible
chat-completions endpoint. Used by the reconstruction techniques
(thought_completion, humpback_backtranslate, star_rationalize) to recover task
goals and implicit thoughts from real provenance.

Key resolution order: explicit ``api_key`` arg → ``OPENROUTER_API_KEY`` env →
a gitignored ``.env`` file in the repo root (``KEY=VALUE`` lines). Model:
explicit ``model`` arg → ``OPENROUTER_MODEL`` env/.env → a sane default.

Provenance (§11): the model id used is recorded on every reconstructed
trajectory via ``model_id`` (the reconstruction modules stamp it).
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "anthropic/claude-sonnet-4.6"
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path = _REPO_ROOT / ".env") -> dict[str, str]:
    """Parse a minimal ``KEY=VALUE`` .env (no export, no quotes handling beyond strip)."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _resolve(name: str, explicit: Optional[str], dotenv: dict[str, str]) -> Optional[str]:
    return explicit or os.environ.get(name) or dotenv.get(name)


class OpenRouterClient:
    """Minimal OpenRouter chat client. Satisfies the ``LLMClient`` protocol."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout: float = 90.0,
        max_retries: int = 4,
        reasoning: Optional[bool] = None,
    ):
        dotenv = _load_dotenv()
        key = _resolve("OPENROUTER_API_KEY", api_key, dotenv)
        if not key:
            raise RuntimeError(
                "No OpenRouter API key. Set OPENROUTER_API_KEY in the environment or in a "
                "gitignored .env file at the repo root (OPENROUTER_API_KEY=sk-or-...)."
            )
        self._key = key
        self._model = _resolve("OPENROUTER_MODEL", model, dotenv) or DEFAULT_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.n_calls = 0
        # Reasoning toggle for thinking models (e.g. DeepSeek V4). Explicit arg wins;
        # else OPENROUTER_REASONING env/.env: off/false/0/no -> disabled, on/true/1/yes
        # -> enabled, anything else / unset -> model default (no override sent).
        self._reasoning = reasoning
        if reasoning is None:
            r = (_resolve("OPENROUTER_REASONING", None, dotenv) or "").strip().lower()
            if r in ("off", "false", "0", "no", "disabled"):
                self._reasoning = False
            elif r in ("on", "true", "1", "yes", "enabled"):
                self._reasoning = True

    @property
    def model_id(self) -> str:
        return self._model

    def web_search(self, query: str, *, max_results: int = 4,
                   system: Optional[str] = None) -> dict:
        """Real literature/web search via OpenRouter's web plugin. Returns
        {answer, citations:[{title,url}]}. Used by the literature_search tool so an
        agent's investigation phase rests on real, cited sources (§19 acquisition)."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": query})
        payload = {
            "model": self._model,
            "plugins": [{"id": "web", "max_results": max_results}],
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            req = urllib.request.Request(OPENROUTER_URL, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                self.n_calls += 1
                msg = body["choices"][0]["message"]
                cites = [{"title": (a.get("url_citation", {}) or {}).get("title", ""),
                          "url": (a.get("url_citation", {}) or {}).get("url", "")}
                         for a in (msg.get("annotations") or []) if a.get("type") == "url_citation"
                         or "url_citation" in a]
                return {"answer": (msg.get("content") or "").strip(), "citations": cites}
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
                last_err = e
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"OpenRouter web_search failed after {self.max_retries} retries: {last_err}")

    def complete(self, prompt: str, *, system: Optional[str] = None) -> str:
        """Single-turn completion. Returns the assistant message text."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self._reasoning is not None:
            payload["reasoning"] = {"enabled": self._reasoning}
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
            # OpenRouter attribution headers (optional but recommended).
            "HTTP-Referer": "https://github.com/scicoder-data",
            "X-Title": "SciCoder data engine",
        }

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            req = urllib.request.Request(OPENROUTER_URL, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                self.n_calls += 1
                msg = body["choices"][0]["message"]
                content = msg.get("content") or msg.get("reasoning") or ""
                return content.strip()
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "ignore")[:500]
                # 401/403 = auth (don't retry); 429/5xx = transient (retry w/ backoff)
                if e.code in (401, 403):
                    raise RuntimeError(f"OpenRouter auth failed ({e.code}): {detail}") from e
                last_err = RuntimeError(f"OpenRouter HTTP {e.code}: {detail}")
                if e.code not in (408, 409, 429) and e.code < 500:
                    raise last_err from e
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
                last_err = e
            time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"OpenRouter request failed after {self.max_retries} retries: {last_err}")
