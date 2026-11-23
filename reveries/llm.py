"""Minimal OpenAI-compatible chat client for the SGLang servers.

Every call returns the text, the reasoning text (Gemma thinking), token usage and
the exact request parameters, so each run can record what was sent.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import httpx

# Fixed sampling for every probe and arm (00_OVERVIEW.md §7: one fixed temperature).
# Values are Google's recommended Gemma 4 sampling settings.
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 64}
MODEL = "gemma-4-31b-it"


@dataclass
class Reply:
    content: str
    reasoning: str
    usage: dict
    finish_reason: str
    request: dict
    seconds: float


@dataclass
class LLM:
    base_url: str = "http://localhost:8100"
    model: str = MODEL
    thinking: bool = True
    max_tokens: int = 8192
    timeout: float = 1800.0
    sampling: dict = field(default_factory=lambda: dict(SAMPLING))

    def chat(self, messages: list[dict], seed: int | None = None, max_tokens: int | None = None,
             thinking: bool | None = None, stop: list[str] | None = None) -> Reply:
        req = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            **self.sampling,
            "chat_template_kwargs": {"enable_thinking": self.thinking if thinking is None else thinking},
        }
        if seed is not None:
            req["seed"] = seed
        if stop:
            req["stop"] = stop
        t0 = time.time()
        for attempt in range(5):
            try:
                r = httpx.post(f"{self.base_url}/v1/chat/completions", json=req, timeout=self.timeout)
                r.raise_for_status()
                break
            except (httpx.TransportError, httpx.HTTPStatusError):
                if attempt == 4:
                    raise
                time.sleep(10 * (attempt + 1))
        d = r.json()
        ch = d["choices"][0]
        msg = ch["message"]
        return Reply(
            content=msg.get("content") or "",
            reasoning=msg.get("reasoning_content") or "",
            usage=d.get("usage") or {},
            finish_reason=ch.get("finish_reason") or "",
            request={k: v for k, v in req.items() if k != "messages"},
            seconds=time.time() - t0,
        )


def server_info(base_url: str) -> dict:
    """Server config for provenance (00_OVERVIEW.md §7: record server config per run)."""
    try:
        return httpx.get(f"{base_url}/get_server_info", timeout=30).json()
    except Exception as e:  # recorded, not fatal
        return {"error": repr(e)}


# P20: a Claude model as the guide (the agent stays on Gemma). Same chat() interface and Reply shape as LLM.
CLAUDE_PRICE = {  # USD per million tokens (input, output), Anthropic first-party list prices, 2026-10
    "claude-opus-5-5": (4.0, 20.0), "claude-fable-5-1": (10.0, 50.0),
    "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50),
}


class BudgetExceeded(RuntimeError):
    pass


class ClaudeSpend:
    """Process-wide spend ledger with a hard cap, checked before every call."""

    def __init__(self, cap_usd: float, ledger: str | None = None):
        import threading
        self.cap, self.spent, self.ledger, self.lock = cap_usd, 0.0, ledger, threading.Lock()

    def check(self):
        if self.spent >= self.cap:
            raise BudgetExceeded(f"spent ${self.spent:.2f} of the ${self.cap:.2f} cap")

    def add(self, model: str, usage: dict):
        pin, pout = CLAUDE_PRICE[model]
        cost = usage["prompt_tokens"] * pin / 1e6 + usage["completion_tokens"] * pout / 1e6
        with self.lock:
            self.spent += cost
            if self.ledger:
                import json as _j
                with open(self.ledger, "a") as f:
                    f.write(_j.dumps({"t": time.time(), "model": model, **usage, "usd": round(cost, 5),
                                      "total_usd": round(self.spent, 4)}) + "\n")
        return cost


@dataclass
class ClaudeLLM:
    """Anthropic Messages API. Adaptive thinking is on by default on these models; effort is set explicitly.
    No server-side refusal fallback: a fallback would swap the guide model mid-experiment. A refusal is returned as
    finish_reason "refusal" with empty content, which the guide parses as no judgement."""
    model: str = "claude-opus-5-5"
    effort: str = "high"
    max_tokens: int = 16000
    spend: ClaudeSpend | None = None

    def __post_init__(self):
        import anthropic
        self.client = anthropic.Anthropic(max_retries=6)

    def chat(self, messages: list[dict], seed: int | None = None, max_tokens: int | None = None,
             thinking: bool | None = None, stop: list[str] | None = None) -> Reply:
        if self.spend:
            self.spend.check()
        req = {"model": self.model, "max_tokens": max(max_tokens or 0, self.max_tokens),
               "output_config": {"effort": self.effort}}
        t0 = time.time()
        r = self.client.messages.create(messages=messages, **req)
        text = "".join(b.text for b in r.content if b.type == "text")
        usage = {"prompt_tokens": r.usage.input_tokens, "completion_tokens": r.usage.output_tokens,
                 "total_tokens": r.usage.input_tokens + r.usage.output_tokens}
        if self.spend:
            usage["usd"] = round(self.spend.add(self.model, usage), 5)
        return Reply(content=text, reasoning="", usage=usage, finish_reason=r.stop_reason or "",
                     request={**req, "request_id": r._request_id}, seconds=time.time() - t0)


# P23: an OpenAI-compatible endpoint as the guide (e.g. DeepSeek on the SPIKE cluster). Same Reply shape as LLM.
# Primarily for token counting: the usage block records prompt and completion tokens per call.
@dataclass
class OpenAILLM:
    """OpenAI chat-completions API with a bearer key. base_url must include the version suffix (e.g. .../v1).
    Token use is recorded per call; `spend` (a ClaudeSpend without pricing) sums prompt/completion tokens."""
    base_url: str
    model: str
    api_key: str
    effort: str | None = None
    max_tokens: int = 16000
    timeout: float = 1800.0
    spend: "ClaudeSpend | None" = None
    sampling: dict = field(default_factory=dict)  # for logging/compat with the agent harness
    thinking: bool = True

    def chat(self, messages: list[dict], seed: int | None = None, max_tokens: int | None = None,
             thinking: bool | None = None, stop: list[str] | None = None) -> Reply:
        req = {"model": self.model, "messages": messages, "max_tokens": max_tokens or self.max_tokens, **self.sampling}
        if self.effort:
            req["reasoning_effort"] = self.effort
        if stop:
            req["stop"] = stop
        t0 = time.time()
        for attempt in range(5):
            try:
                r = httpx.post(f"{self.base_url}/chat/completions", json=req, timeout=self.timeout,
                               headers={"Authorization": f"Bearer {self.api_key}"})
                r.raise_for_status()
                break
            except (httpx.TransportError, httpx.HTTPStatusError):
                if attempt == 4:
                    raise
                time.sleep(10 * (attempt + 1))
        d = r.json()
        ch = d["choices"][0]; msg = ch["message"]
        u = d.get("usage") or {}
        usage = {"prompt_tokens": u.get("prompt_tokens", 0), "completion_tokens": u.get("completion_tokens", 0),
                 "total_tokens": u.get("total_tokens", 0)}
        if self.spend:  # token-only ledger (cost 0)
            with self.spend.lock:
                self.spend.spent += 0.0
                if self.spend.ledger:
                    import json as _j
                    with open(self.spend.ledger, "a") as f:
                        f.write(_j.dumps({"t": time.time(), "model": self.model, **usage}) + "\n")
        return Reply(content=msg.get("content") or "", reasoning=msg.get("reasoning_content") or "",
                     usage=usage, finish_reason=ch.get("finish_reason") or "",
                     request={k: v for k, v in req.items() if k != "messages"}, seconds=time.time() - t0)
