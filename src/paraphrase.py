"""S2 paraphrase pass: rewrite rendered vignettes for surface diversity.

The paraphraser MUST be a different model family from the subject models (Gemma-2,
Llama-3.1) so its style does not self-correlate with the interp experiments (design-doc
§5.3). Default backend = Gemini Flash; drop-in = GPT-4o-mini.

Both tiers are paraphrased by the SAME backend so the only systematic difference between
explicit and implicit vignettes is the banned-lexicon constraint (the RQ1 confound
control). For implicit vignettes the result is re-scanned for banned terms and retried;
if it still leaks after `max_retries`, we fall back to the clean rendered template
(tagged paraphrase_model="template-only").

Results are cached to disk keyed by content, so reruns are reproducible and never re-bill
the API despite the API being non-deterministic.
"""

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from templates import find_banned

# The "add nothing" half of this prompt is load-bearing. The pilot told the model to
# KEEP every fact but never told it to INVENT none, so it elaborated qualitative
# statements into quantitative ones: "I've been through several market cycles" became
# "I've been investing for 20 years", which QC then flagged on 56 clients under 30.
# Preservation and non-invention are separate instructions and both have to be given.
_SYSTEM = (
    "You rewrite a first-person description a person gives of themselves and their "
    "finances. Keep every concrete fact (ages, numbers, timelines, family, job, money "
    "amounts, how they reacted to past market moves). Change only wording and sentence "
    "flow so it reads naturally and a little differently.\n"
    "Add NOTHING. Do not introduce any number, duration, date, age, count or money "
    "amount that is not already stated in the text. In particular, never convert a "
    "vague description of experience into a specific span of years: if the text says "
    "someone has been through market cycles, do not say how many years they have been "
    "investing. Do not add opinions, motives, job details or family details that are "
    "not there. Keep one sentence per fact and do not merge or split them.\n"
    "Return ONLY the rewritten paragraph, first person, no preamble."
)
_IMPLICIT_RULE = (
    " Hard rule: do NOT use any of these words or variants of them: {banned}. Convey the "
    "same meaning through plain life narrative instead."
)


class ParaphraseClient:
    """Backend-agnostic paraphraser with a disk cache and banned-term retries."""

    def __init__(self, backend, model, banned_words, banned_regex, cache_dir,
                 temperature=0.7, max_retries=4, max_workers=8, enabled=True,
                 reasoning_effort=None):
        self.backend = backend
        self.model = model
        # Reasoning models (the GPT-5 family, o-series) bill reasoning tokens as
        # output. Paraphrasing is style transfer and needs none, so this is passed as
        # "none" rather than left at the model default.
        self.reasoning_effort = reasoning_effort
        # Params some models reject outright (reasoning models often refuse a custom
        # temperature). On the first rejection the offending param is dropped for the
        # rest of the run instead of failing every one of ~12,000 calls.
        self._unsupported = set()
        self.banned_words = banned_words
        self.banned_regex = banned_regex
        self.temperature = temperature
        self.max_retries = max_retries
        self.max_workers = max_workers
        self.enabled = enabled
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = None  # lazy

    # -- backends ----------------------------------------------------------------
    def _ensure_client(self):
        if self._client is not None:
            return
        if self.backend == "gemini":
            import google.generativeai as genai
            key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not key:
                raise RuntimeError("GEMINI_API_KEY not set")
            genai.configure(api_key=key)
            self._client = genai.GenerativeModel(self.model, system_instruction=_SYSTEM)
        elif self.backend == "openai":
            from openai import OpenAI
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY not set")
            self._client = OpenAI()
        else:
            raise ValueError(f"unknown backend {self.backend!r}")

    def _openai_kwargs(self):
        """Optional params, minus any this model has already rejected."""
        kw = {}
        if "temperature" not in self._unsupported:
            kw["temperature"] = self.temperature
        if self.reasoning_effort and "reasoning_effort" not in self._unsupported:
            kw["reasoning_effort"] = self.reasoning_effort
        return kw

    def _note_unsupported(self, err):
        """If the API rejected an optional param, drop it for the rest of the run.

        Returns True when something was dropped, i.e. the call is worth retrying.
        """
        msg = str(err).lower()
        dropped = False
        for param in ("temperature", "reasoning_effort"):
            if param not in self._unsupported and param in msg and (
                    "unsupported" in msg or "not supported" in msg
                    or "unknown" in msg or "does not support" in msg
                    or "invalid" in msg):
                self._unsupported.add(param)
                print(f"    note: {self.model} rejected '{param}'; continuing without it")
                dropped = True
        return dropped

    def _call(self, prompt):
        self._ensure_client()
        for attempt in range(4):
            try:
                if self.backend == "gemini":
                    import google.generativeai as genai
                    r = self._client.generate_content(
                        prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=self.temperature),
                    )
                    return r.text.strip()
                else:  # openai
                    r = self._client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "system", "content": _SYSTEM},
                                  {"role": "user", "content": prompt}],
                        **self._openai_kwargs(),
                    )
                    return r.choices[0].message.content.strip()
            except Exception as e:
                if self.backend != "gemini" and self._note_unsupported(e):
                    continue        # retry immediately without the rejected param
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)

    # -- caching -----------------------------------------------------------------
    def _cache_id(self, mode, text):
        """Stable id for (prompt, mode, source text) -- also the Batch API custom_id.

        The prompt digest is part of the key. Without it, editing _SYSTEM leaves every
        existing cache entry looking valid, so a re-run silently serves text generated
        under the OLD instructions -- the same stale-cache trap the activation cache
        had. Changing the prompt now simply misses the cache, which is correct.
        """
        prompt_digest = hashlib.sha1(
            (_SYSTEM + "|" + _IMPLICIT_RULE + "|" + ",".join(self.banned_words)).encode()
        ).hexdigest()[:10]
        return hashlib.sha1(
            f"{self.backend}|{self.model}|{prompt_digest}|{mode}|{text}".encode()).hexdigest()

    def _cache_path(self, mode, text):
        return self.cache_dir / f"{self._cache_id(mode, text)}.json"

    def is_cached(self, mode, text):
        return self._cache_path(mode, text).exists()

    def _build_prompt(self, mode, text, reinforce=False):
        if mode != "implicit":
            return text
        p = text + _IMPLICIT_RULE.format(banned=", ".join(self.banned_words))
        if reinforce:
            p += " You used a forbidden word last time. Try again, avoiding ALL of them."
        return p

    def _store(self, mode, text, out, model):
        self._cache_path(mode, text).write_text(json.dumps({"text": out, "model": model}))
        return out, model

    # -- public ------------------------------------------------------------------
    def paraphrase_one(self, text, mode):
        """Return (final_text, paraphrase_model). `mode` in {explicit, implicit}."""
        if not self.enabled:
            return text, "template-only"

        cp = self._cache_path(mode, text)
        if cp.exists():
            d = json.loads(cp.read_text())
            return d["text"], d["model"]

        out, model = None, self.model
        for attempt in range(self.max_retries):
            cand = self._call(self._build_prompt(mode, text, reinforce=attempt > 0))
            if mode == "implicit" and find_banned(cand, self.banned_regex):
                continue
            out = cand
            break
        if out is None:  # never produced a clean implicit rewrite -> use the clean template
            out, model = text, "template-only"
        return self._store(mode, text, out, model)

    def paraphrase_batch(self, jobs):
        """jobs: list of (text, mode). Returns list of (final_text, model), order-preserved."""
        if not self.enabled:
            return [(t, "template-only") for t, _ in jobs]
        results = [None] * len(jobs)
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futs = {ex.submit(self.paraphrase_one, t, m): i for i, (t, m) in enumerate(jobs)}
            for fut in futs:
                results[futs[fut]] = fut.result()
        return results

    # -- OpenAI Batch API (async, 50% cheaper, separate rate limits -- no RPD wall) ----
    def build_batch(self, jobs):
        """jobs: list of (text, mode). Returns (request_lines, sidecar) for UNCACHED jobs.

        request_lines: dicts ready to write as the Batch API input JSONL.
        sidecar: {custom_id -> {mode, source}} so the collector can map results back,
                 re-run the banned-word check, and fall back to template-only on leaks.
        """
        if self.backend != "openai":
            raise ValueError("Batch mode requires backend 'openai'")
        seen, lines, sidecar = set(), [], {}
        for text, mode in jobs:
            if self.is_cached(mode, text):
                continue
            cid = self._cache_id(mode, text)
            if cid in seen:          # identical (mode, text) appears twice -> one request
                continue
            seen.add(cid)
            lines.append({
                "custom_id": cid, "method": "POST", "url": "/v1/chat/completions",
                "body": {"model": self.model,
                         "messages": [{"role": "system", "content": _SYSTEM},
                                      {"role": "user", "content": self._build_prompt(mode, text)}],
                         **self._openai_kwargs()},
            })
            sidecar[cid] = {"mode": mode, "source": text}
        return lines, sidecar

    def submit_batch(self, input_path):
        """Upload the input JSONL and create a batch. Returns batch_id."""
        self._ensure_client()
        with open(input_path, "rb") as f:
            up = self._client.files.create(file=f, purpose="batch")
        batch = self._client.batches.create(
            input_file_id=up.id, endpoint="/v1/chat/completions", completion_window="24h")
        return batch.id

    def batch_status(self, batch_id):
        """Return the raw batch object (.status, .request_counts, .output_file_id, ...)."""
        self._ensure_client()
        return self._client.batches.retrieve(batch_id)

    def collect_batch(self, batch_id, sidecar):
        """Download a completed batch's output and populate the cache. Returns a summary."""
        self._ensure_client()
        batch = self._client.batches.retrieve(batch_id)
        if batch.status != "completed" and not batch.output_file_id:
            return {"status": batch.status, "cached": 0, "fallbacks": 0, "errors": 0}
        text = self._client.files.content(batch.output_file_id).text
        cached = fallbacks = errors = 0
        for line in text.splitlines():
            rec = json.loads(line)
            meta = sidecar.get(rec["custom_id"])
            if meta is None:
                continue
            mode, source = meta["mode"], meta["source"]
            content = None
            resp = rec.get("response")
            if resp and resp.get("status_code") == 200:
                content = resp["body"]["choices"][0]["message"]["content"].strip()
            if content and not (mode == "implicit" and find_banned(content, self.banned_regex)):
                self._store(mode, source, content, self.model)
            else:                                    # error, empty, or leaked -> clean template
                self._store(mode, source, source, "template-only")
                fallbacks += 1
            cached += 1
            if content is None:
                errors += 1
        return {"status": batch.status, "cached": cached,
                "fallbacks": fallbacks, "errors": errors}
