"""Provider-agnostic LLM client with a deterministic disk cache and a token/cost meter.

Configuration (environment variables, or an uncommitted `.env` at the repository root):
    AUDIT_LLM_PROVIDER   openai (default) | anthropic
    OPENAI_API_KEY / ANTHROPIC_API_KEY
    AUDIT_TEXT_MODEL     model for request classification and PDF text extraction
    AUDIT_VISION_MODEL   model for screenshot extraction

Every response is cached under code/cache/ keyed by sha256(provider, model, prompt, image
bytes), so a rerun makes zero API calls and returns identical answers. With no API key,
cached answers are still served; an uncached call raises LLMUnavailable.
"""

import base64
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
CACHE_DIR = CODE_DIR / "cache"

DEFAULT_MODELS = {
    "openai": {"text": "gpt-4.1-mini-2025-04-14", "vision": "gpt-4.1-2025-04-14"},
    "anthropic": {"text": "claude-haiku-5-5", "vision": "claude-sonnet-5-5"},
}

API_KEY_ENV = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}

# USD per 1M tokens (input, output), used only for the usage report's cost estimate.
PRICES = {
    "gpt-4.1-mini-2025-04-14": (0.40, 1.60),
    "gpt-4.1-2025-04-14": (2.00, 8.00),
    "gpt-4.1-nano-2025-04-14": (0.10, 0.40),
    "gpt-4o-mini-2024-07-18": (0.15, 0.60),
}


class LLMError(RuntimeError):
    """No provider could answer (no key, network/API failure, or unusable JSON after retries)."""


class LLMUnavailable(LLMError):
    """No API key for the provider and the answer is not cached."""


def load_dotenv(path=REPO_ROOT / ".env"):
    """Minimal .env reader (KEY=VALUE lines); never overrides variables already set."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class Meter:
    """Counts model calls and tokens.

    `calls`/`input_tokens`/`output_tokens` are API calls made by this process. Answers served from
    the cache also carry the usage of the call that originally produced them (`cached_*`), so a
    report can state the full model cost behind an output even when the rerun itself was free.
    """

    def __init__(self):
        self.calls = defaultdict(lambda: {"calls": 0, "cached": 0, "input_tokens": 0, "output_tokens": 0,
                                          "cached_input_tokens": 0, "cached_output_tokens": 0})

    def record(self, provider, model, usage, cached):
        row = self.calls[(provider, model)]
        prefix = "cached_" if cached else ""
        row["cached" if cached else "calls"] += 1
        row[prefix + "input_tokens"] += usage.get("input_tokens", 0)
        row[prefix + "output_tokens"] += usage.get("output_tokens", 0)

    @staticmethod
    def cost(model, input_tokens, output_tokens):
        price_in, price_out = PRICES.get(model, (0.0, 0.0))
        return (input_tokens * price_in + output_tokens * price_out) / 1_000_000

    def summary(self):
        rows = []
        for (provider, model), row in sorted(self.calls.items()):
            total_in = row["input_tokens"] + row["cached_input_tokens"]
            total_out = row["output_tokens"] + row["cached_output_tokens"]
            rows.append({
                "provider": provider, "model": model, **row,
                "cost_usd": self.cost(model, row["input_tokens"], row["output_tokens"]),
                "total_calls": row["calls"] + row["cached"],
                "total_input_tokens": total_in, "total_output_tokens": total_out,
                "total_cost_usd": self.cost(model, total_in, total_out),
            })
        return rows


class LLM:
    """Primary provider first; on failure, the other provider when its key is configured."""

    RETRIES = 4

    def __init__(self, provider=None, cache_dir=CACHE_DIR):
        load_dotenv()
        self.provider = (provider or os.environ.get("AUDIT_LLM_PROVIDER") or "openai").lower()
        if self.provider not in DEFAULT_MODELS:
            raise ValueError(f"unsupported AUDIT_LLM_PROVIDER {self.provider!r}")
        self.models = {
            "text": os.environ.get("AUDIT_TEXT_MODEL") or DEFAULT_MODELS[self.provider]["text"],
            "vision": os.environ.get("AUDIT_VISION_MODEL") or DEFAULT_MODELS[self.provider]["vision"],
        }
        self.cache_dir = cache_dir
        self.meter = Meter()
        self.used_keys = set()
        self.events = []            # fallbacks taken and failures, for the usage report
        self._clients = {}

    def prune_cache(self):
        """Delete cache entries this process did not use (stale prompts). Call after a full run."""
        removed = 0
        for path in self.cache_dir.glob("*.json"):
            if path.stem not in self.used_keys:
                path.unlink()
                removed += 1
        return removed

    # -- public API --------------------------------------------------------

    def complete_json(self, system, prompt, kind="text", image_path=None, max_tokens=4000):
        """Return the model's JSON object answer for (system, prompt[, image]).

        Order: cache (primary model, then fallback model) -> primary provider -> fallback
        provider. Raises LLMError when nothing can answer; callers degrade gracefully.
        """
        image_bytes = image_path.read_bytes() if image_path else b""
        routes = [(self.provider, self.models[kind])]
        other = next(p for p in DEFAULT_MODELS if p != self.provider)
        if os.environ.get(API_KEY_ENV[other]):
            routes.append((other, DEFAULT_MODELS[other][kind]))
        keys = [self._key(provider, model, system, prompt, image_bytes) for provider, model in routes]

        for (provider, model), key in zip(routes, keys):
            cached = self._read_cache(key)
            if cached is not None:
                self.used_keys.add(key)
                self.meter.record(provider, model, cached["usage"], cached=True)
                return cached["response"]

        errors = []
        for (provider, model), key in zip(routes, keys):
            try:
                response, usage = self._call(provider, model, system, prompt, image_bytes, image_path, max_tokens)
            except Exception as exc:  # noqa: BLE001 - any provider failure moves on to the next route
                errors.append(f"{provider}/{model}: {type(exc).__name__}: {str(exc)[:160]}")
                continue
            if provider != self.provider:
                self.events.append(f"fallback provider {provider}/{model} answered after: {errors[-1]}")
            self.used_keys.add(key)
            self.meter.record(provider, model, usage, cached=False)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            (self.cache_dir / f"{key}.json").write_text(json.dumps({
                "provider": provider, "model": model, "usage": usage, "prompt_sha256": key, "response": response,
            }, ensure_ascii=False, indent=1), encoding="utf-8")
            return response
        self.events.append("no provider answered: " + " | ".join(errors))
        if all("LLMUnavailable" in e for e in errors):
            raise LLMUnavailable(" | ".join(errors))
        raise LLMError(" | ".join(errors))

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _key(provider, model, system, prompt, image_bytes):
        return hashlib.sha256(json.dumps([provider, model, system, prompt]).encode("utf-8") + image_bytes).hexdigest()

    def _read_cache(self, key):
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            entry = None
        if isinstance(entry, dict) and "response" in entry and "usage" in entry:
            return entry
        self.events.append(f"ignored unreadable cache entry {path.name}")
        return None

    def _client_for(self, provider):
        if provider in self._clients:
            return self._clients[provider]
        if not os.environ.get(API_KEY_ENV[provider]):
            raise LLMUnavailable(f"{API_KEY_ENV[provider]} is not set and the answer is not cached")
        if provider == "openai":
            from openai import OpenAI
            client = OpenAI()
        else:
            import anthropic
            client = anthropic.Anthropic()
        self._clients[provider] = client
        return client

    def _call(self, provider, model, system, prompt, image_bytes, image_path, max_tokens):
        client = self._client_for(provider)
        media_type = "image/png" if image_path and image_path.suffix.lower() == ".png" else "image/jpeg"
        encoded = base64.b64encode(image_bytes).decode("ascii") if image_bytes else None
        for attempt in range(self.RETRIES):
            try:
                if provider == "openai":
                    content = [{"type": "text", "text": prompt}]
                    if encoded:
                        content.append({"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{encoded}", "detail": "high"}})
                    result = client.chat.completions.create(
                        model=model, temperature=0, seed=7, max_tokens=max_tokens,
                        response_format={"type": "json_object"},
                        messages=[{"role": "system", "content": system}, {"role": "user", "content": content}],
                    )
                    text = result.choices[0].message.content
                    usage = {"input_tokens": result.usage.prompt_tokens, "output_tokens": result.usage.completion_tokens}
                else:
                    content = [{"type": "text", "text": prompt + "\n\nRespond with a single JSON object only."}]
                    if encoded:
                        content.insert(0, {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": encoded}})
                    result = client.messages.create(
                        model=model, temperature=0, max_tokens=max_tokens, system=system,
                        messages=[{"role": "user", "content": content}],
                    )
                    text = "".join(block.text for block in result.content if block.type == "text")
                    text = text[text.find("{"): text.rfind("}") + 1]
                    usage = {"input_tokens": result.usage.input_tokens, "output_tokens": result.usage.output_tokens}
                answer = json.loads(text)
                if not isinstance(answer, dict):
                    raise ValueError("model answer is not a JSON object")
                return answer, usage
            except LLMUnavailable:
                raise
            except Exception:  # transient API / JSON errors: retry with backoff, then let the caller fall back
                if attempt == self.RETRIES - 1:
                    raise
                time.sleep(2 ** attempt)
