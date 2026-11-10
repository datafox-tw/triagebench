"""LLM verifier over retrieved candidates.

Three things here exist because of how this fails in production rather than in
a notebook:

**A disk cache keyed by the exact prompt.** Re-running the benchmark must not
re-bill the model, and a published number must be reproducible by someone who
cannot afford to rerun 2,000 calls. Cache hits also make prompt-profile
comparisons honest, since unchanged decisions are not re-sampled.

**A fallback on failure.** Every call can fail -- rate limit, timeout, a reply
that parses to nothing. A verifier that raises is a pipeline that stops. This
one falls back to NEW, records `failed=True`, and the failure rate is reported
alongside accuracy rather than hidden.

**Validation of the returned id.** Models cheerfully return ids that were not
in the candidate list. An unvalidated id is a merge into an unrelated report,
which is the single most expensive error this system can make.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .base import Candidate, Decision, Verifier
from .prompts import build_prompt

CACHE_DIR = Path("data/cache")
_ANSWER = re.compile(r"\b(?:DUPLICATE|DUP)\b[^0-9]{0,12}(\d+)", re.IGNORECASE)
_NEW = re.compile(r"\bNEW\b", re.IGNORECASE)


class PromptCache:
    """Append-only JSONL cache of prompt -> completion."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._mem: dict[str, str] = {}
        if path.exists():
            with path.open() as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self._mem[rec["key"]] = rec["response"]

    @staticmethod
    def key(model: str, prompt: str) -> str:
        h = hashlib.sha256()
        h.update(model.encode())
        h.update(b"\0")
        h.update(prompt.encode("utf-8", "ignore"))
        return h.hexdigest()

    def get(self, key: str) -> str | None:
        return self._mem.get(key)

    def put(self, key: str, response: str) -> None:
        with self._lock:
            if key in self._mem:
                return
            self._mem[key] = response
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as fh:
                fh.write(json.dumps({"key": key, "response": response}) + "\n")

    def __len__(self) -> int:
        return len(self._mem)


class LLMVerifier(Verifier):
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        profile: str = "balanced",
        *,
        top_n: int = 5,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str | None = None,
        max_retries: int = 3,
        timeout: float = 60.0,
        temperature: float = 0.0,
        cache_dir: Path | str = CACHE_DIR,
    ) -> None:
        self.model = model
        self.profile = profile
        self.top_n = top_n
        self.max_retries = max_retries
        self.timeout = timeout
        self.temperature = temperature
        self.name = f"llm-{profile}-top{top_n}"

        self._api_key_env = api_key_env
        self._base_url = base_url
        self._client = None
        self.cache = PromptCache(Path(cache_dir) / f"verify-{_slug(model)}.jsonl")

        # Model families disagree about parameter names and defaults: the
        # gpt-5 line rejects `max_tokens` in favour of `max_completion_tokens`
        # and refuses a non-default temperature. Rather than maintain a table
        # of model capabilities that goes stale, drop the offending parameter
        # once on first rejection and remember the choice.
        self._param_style: str | None = None
        self._send_temperature = True

        self.n_calls = 0
        self.n_cache_hits = 0
        self.n_failures = 0
        self._counter_lock = threading.Lock()

    # -- client -------------------------------------------------------------

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            key = os.environ.get(self._api_key_env)
            if not key:
                raise RuntimeError(
                    f"{self._api_key_env} is not set; export it or pass api_key_env"
                )
            self._client = OpenAI(api_key=key, base_url=self._base_url,
                                  timeout=self.timeout)
        return self._client

    def _complete(self, prompt: str) -> tuple[str, int, int, bool]:
        """Returns (text, prompt_tokens, completion_tokens, failed)."""
        key = PromptCache.key(self.model, prompt)
        cached = self.cache.get(key)
        if cached is not None:
            with self._counter_lock:
                self.n_cache_hits += 1
            return cached, 0, 0, False

        delay = 1.0
        for attempt in range(self.max_retries):
            try:
                resp = self._get_client().chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    **self._call_kwargs(),
                )
                text = (resp.choices[0].message.content or "").strip()
                usage = resp.usage
                with self._counter_lock:
                    self.n_calls += 1
                self.cache.put(key, text)
                return (text,
                        getattr(usage, "prompt_tokens", 0) or 0,
                        getattr(usage, "completion_tokens", 0) or 0,
                        False)
            except Exception as exc:  # noqa: BLE001 - retried, then counted
                if self._adapt_params(exc):
                    continue  # parameter problem, not a transient one
                if attempt == self.max_retries - 1:
                    with self._counter_lock:
                        self.n_failures += 1
                    return "", 0, 0, True
                time.sleep(delay)
                delay *= 2
        return "", 0, 0, True

    def _call_kwargs(self) -> dict:
        # Reasoning-capable models spend completion tokens on hidden reasoning
        # before emitting the answer, so a 24-token cap starves them.
        if self._param_style == "max_completion_tokens":
            kwargs: dict = {"max_completion_tokens": 2048}
        else:
            kwargs = {"max_tokens": 24}
        if self._send_temperature:
            kwargs["temperature"] = self.temperature
        return kwargs

    def _adapt_params(self, exc: Exception) -> bool:
        """Drop a rejected parameter. Returns True if something was changed."""
        msg = str(exc).lower()
        with self._counter_lock:
            if "max_tokens" in msg and self._param_style != "max_completion_tokens":
                self._param_style = "max_completion_tokens"
                return True
            if "temperature" in msg and self._send_temperature:
                self._send_temperature = False
                return True
        return False

    # -- verification -------------------------------------------------------

    def verify(self, query_id: int, query_text: str,
               candidates: list[Candidate]) -> Decision:
        shortlist = candidates[: self.top_n]
        if not shortlist:
            return Decision(query_id, None, reason="no candidates")

        prompt = build_prompt(
            self.profile, query_text,
            [(c.doc_id, c.text) for c in shortlist],
        )

        start = time.perf_counter()
        text, p_tok, c_tok, failed = self._complete(prompt)
        latency = time.perf_counter() - start

        if failed:
            # Fail closed: an unavailable verifier must not invent merges.
            return Decision(query_id, None, reason="call failed; fell back to NEW",
                            latency_s=latency, failed=True)

        master, reason = self._parse(text, {c.doc_id for c in shortlist})
        return Decision(query_id, master, reason=reason, latency_s=latency,
                        prompt_tokens=p_tok, completion_tokens=c_tok)

    @staticmethod
    def _parse(text: str, allowed: set[int]) -> tuple[int | None, str]:
        m = _ANSWER.search(text)
        if m:
            doc_id = int(m.group(1))
            if doc_id in allowed:
                return doc_id, "llm:duplicate"
            # A confident answer naming a report that was never offered is a
            # hallucinated merge target, not a near miss.
            return None, f"llm:id {doc_id} not in candidates; treated as NEW"
        if _NEW.search(text):
            return None, "llm:new"
        return None, f"llm:unparseable {text[:40]!r}; treated as NEW"

    # -- batch --------------------------------------------------------------

    def verify_many(self, items: list[tuple[int, str, list[Candidate]]],
                    workers: int = 8,
                    progress_every: int = 0) -> list[Decision]:
        """Verify a batch concurrently, preserving input order."""
        out: list[Decision | None] = [None] * len(items)
        start = time.perf_counter()
        done = 0
        lock = threading.Lock()

        def run(i: int):
            nonlocal done
            qid, qtext, cands = items[i]
            out[i] = self.verify(qid, qtext, cands)
            if progress_every:
                with lock:
                    done += 1
                    if done % progress_every == 0:
                        rate = done / (time.perf_counter() - start)
                        print(f"      {done:,}/{len(items):,} verified "
                              f"({rate:.1f}/s, {self.n_cache_hits:,} cached)",
                              flush=True)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(run, range(len(items))))

        return [d for d in out if d is not None]


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", s).strip("-")
