"""Streaming client for any OpenAI-compatible /v1/chat/completions endpoint.

Works unchanged against llama.cpp's llama-server, Ollama, mlx_lm.server and
vLLM, so the same harness covers the Mac, the GPU workstation, the cloud T4
and an Android phone running llama-server under Termux.

All timing is client-side with a monotonic clock, the way a load test would
see it. Token counts come from the server's `usage` block because only the
server knows the real tokenization.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import httpx


@dataclass
class Completion:
    ok: bool
    error: str | None = None
    text: str = ""
    reasoning_chars: int = 0
    t_start: float = 0.0          # time.monotonic() at request send
    t_first: float | None = None  # first generated content (or reasoning) chunk
    t_end: float = 0.0
    chunk_times: list[float] = field(default_factory=list)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    server_timings: dict | None = None  # llama.cpp reports its own prefill/decode rates

    @property
    def ttft_s(self) -> float | None:
        """Time to first *visible text*. Engines hold a chunk back until it is
        valid UTF-8, so when a script is tokenized into byte fragments the first
        character needs several decode steps. That delay is real for the user
        and is language-dependent; it is not pure prefill. Use the server's own
        prefill timing (server_timings) when you need the engine-level number."""
        return None if self.t_first is None else self.t_first - self.t_start

    @property
    def total_s(self) -> float:
        return self.t_end - self.t_start

    @property
    def decode_tok_s(self) -> float | None:
        """Client-side decode rate: tokens after the first chunk / time after it.
        Biased slightly upward when the first chunk carries several tokens (see
        ttft_s). The report prefers the server's own decode rate when available."""
        if self.t_first is None or not self.completion_tokens or self.completion_tokens < 2:
            return None
        span = self.t_end - self.t_first
        return (self.completion_tokens - 1) / span if span > 0 else None

    def inter_chunk_gaps(self) -> list[float]:
        t = self.chunk_times
        return [b - a for a, b in zip(t, t[1:])]


class ChatClient:
    def __init__(self, base_url: str, api_key: str | None = None, timeout_s: float = 600.0):
        self.base_url = base_url.rstrip("/")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.Client(headers=headers, timeout=httpx.Timeout(timeout_s, connect=10.0))

    def close(self) -> None:
        self._http.close()

    def ping(self) -> tuple[bool, str]:
        try:
            r = self._http.get(f"{self.base_url}/models", timeout=5.0)
            r.raise_for_status()
            ids = [m.get("id", "?") for m in r.json().get("data", [])]
            return True, ", ".join(ids) or "(no models listed)"
        except Exception as e:  # noqa: BLE001 - doctor wants the message, whatever it is
            return False, f"{type(e).__name__}: {e}"

    def complete(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int,
        temperature: float = 0.0,
        seed: int | None = 0,
        extra_body: dict | None = None,
    ) -> Completion:
        body = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if seed is not None:
            body["seed"] = seed
        if extra_body:
            body.update(extra_body)

        out = Completion(ok=False)
        parts: list[str] = []
        out.t_start = time.monotonic()
        try:
            with self._http.stream("POST", f"{self.base_url}/chat/completions", json=body) as r:
                if r.status_code >= 400:
                    r.read()
                    out.error = f"HTTP {r.status_code}: {r.text[:300]}"
                    out.t_end = time.monotonic()
                    return out
                for line in r.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    now = time.monotonic()
                    chunk = json.loads(payload)
                    if chunk.get("error"):
                        # Engines report some failures (context overflow, OOM) inside a 200 stream.
                        err = chunk["error"]
                        out.error = f"server error in stream: {err.get('message', err) if isinstance(err, dict) else err}"
                        break
                    if chunk.get("usage"):
                        out.prompt_tokens = chunk["usage"].get("prompt_tokens")
                        out.completion_tokens = chunk["usage"].get("completion_tokens")
                    if chunk.get("timings"):
                        out.server_timings = chunk["timings"]
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        content = delta.get("content") or ""
                        reasoning = delta.get("reasoning_content") or ""
                        if content or reasoning:
                            if out.t_first is None:
                                out.t_first = now
                            out.chunk_times.append(now)
                            parts.append(content)
                            out.reasoning_chars += len(reasoning)
                        if choice.get("finish_reason"):
                            out.finish_reason = choice["finish_reason"]
            if out.error is None and out.finish_reason is None:
                # The connection closed without a finish_reason: a truncated stream is not a result.
                out.error = "stream ended without finish_reason (server dropped the connection?)"
            out.ok = out.error is None
        except Exception as e:  # noqa: BLE001 - a failed request is a data point, not a crash
            out.error = f"{type(e).__name__}: {e}"
        out.t_end = time.monotonic()
        out.text = "".join(parts)
        return out
