"""One small LLM interface shared by all three projects.

Every node in every graph asks for a *typed* answer (a Pydantic model), so the
rest of the code never parses free text with regexes. Three backends:

- ClaudeLLM  - Anthropic SDK, structured outputs via `messages.parse`
- OllamaLLM  - local or Ollama-cloud models, JSON-schema constrained output
- FakeLLM    - scripted responses for offline tests

Pick one with AGENT_LLM_PROVIDER=claude|ollama|fake. With no setting, Claude is
used when ANTHROPIC_API_KEY is set, otherwise Ollama.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5-5")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:120b-cloud")

# USD per million tokens (input, output); used only for the run summary.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}


class LLMRefusal(RuntimeError):
    """The model declined the request (stop_reason == "refusal")."""


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, input_tokens: int, output_tokens: int, seconds: float) -> None:
        with self._lock:
            self.calls += 1
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.seconds += seconds

    def cost_usd(self, model: str) -> float | None:
        price = PRICES.get(model)
        if not price:
            return None
        return (self.input_tokens * price[0] + self.output_tokens * price[1]) / 1_000_000

    def summary(self, model: str) -> str:
        cost = self.cost_usd(model)
        cost_text = f", ~${cost:.4f}" if cost is not None else ""
        return (
            f"{self.calls} LLM calls, {self.input_tokens:,} in / {self.output_tokens:,} out tokens, "
            f"{self.seconds:.1f}s model time{cost_text}"
        )


class LLM(Protocol):
    model: str
    usage: Usage

    def structured(self, system: str, prompt: str, schema: type[T], *, effort: str = "medium") -> T: ...

    def text(self, system: str, prompt: str, *, effort: str = "medium") -> str: ...


class ClaudeLLM:
    """Claude through the official SDK.

    Uses server-side refusal fallbacks ("default" routing) so a declined request
    is retried on a suitable model inside the same call.
    """

    BETAS = ["server-side-fallback-2026-07-01"]

    def __init__(self, model: str = CLAUDE_MODEL):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model
        self.usage = Usage()

    def _record(self, response, started: float) -> None:
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            raise LLMRefusal(f"Model declined: {getattr(details, 'category', None)}")
        self.usage.add(response.usage.input_tokens, response.usage.output_tokens, time.time() - started)

    def structured(self, system: str, prompt: str, schema: type[T], *, effort: str = "medium") -> T:
        started = time.time()
        response = self.client.beta.messages.parse(
            model=self.model,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_format=schema,
            output_config={"effort": effort},
            betas=self.BETAS,
            fallbacks="default",
        )
        self._record(response, started)
        return response.parsed_output

    def text(self, system: str, prompt: str, *, effort: str = "medium") -> str:
        started = time.time()
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": effort},
            betas=self.BETAS,
            fallbacks="default",
        )
        self._record(response, started)
        return "".join(block.text for block in response.content if block.type == "text")


class OllamaLLM:
    """Any Ollama model. JSON output is constrained to the Pydantic schema."""

    def __init__(self, model: str = OLLAMA_MODEL, host: str | None = None):
        import ollama

        self.client = ollama.Client(host=host or os.getenv("OLLAMA_HOST", "http://localhost:11434"))
        self.model = model
        self.usage = Usage()
        # Ollama cloud models reject too many parallel requests (HTTP 429).
        self.slots = threading.Semaphore(int(os.getenv("OLLAMA_MAX_CONCURRENCY", "2")))

    def _chat(self, system: str, prompt: str, fmt=None) -> str:
        import ollama

        for attempt in range(6):
            with self.slots:
                started = time.time()
                try:
                    response = self.client.chat(
                        model=self.model,
                        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                        format=fmt,
                        options={"temperature": 0},
                    )
                except ollama.ResponseError as err:
                    if err.status_code != 429 or attempt == 5:
                        raise
                    retry = True
                else:
                    retry = False
            if retry:
                time.sleep(2 ** attempt)
                continue
            self.usage.add(response.prompt_eval_count or 0, response.eval_count or 0, time.time() - started)
            return response.message.content
        raise RuntimeError("unreachable")

    def structured(self, system: str, prompt: str, schema: type[T], *, effort: str = "medium") -> T:
        # Ollama-cloud models do not always honour `format`, so the schema also goes in the prompt.
        json_schema = schema.model_json_schema()
        system = f"{system}\n\nReply with only a JSON object that matches this JSON schema:\n{json.dumps(json_schema)}"
        content = self._chat(system, prompt, json_schema)
        try:
            return schema.model_validate_json(content)
        except ValidationError as err:
            # One repair attempt: show the model its own output and the error.
            repair = f"{prompt}\n\nYour previous reply was invalid:\n{content}\n\nError:\n{err}\nReply again with valid JSON."
            return schema.model_validate_json(self._chat(system, repair, json_schema))

    def text(self, system: str, prompt: str, *, effort: str = "medium") -> str:
        return self._chat(system, prompt)


Responder = Callable[[str, str], BaseModel | str]


class FakeLLM:
    """Scripted model for tests.

    Register a responder per schema name (or "text"). A responder receives
    (system, prompt) and returns the object to hand back, so tests can branch on
    what the graph actually asked.
    """

    def __init__(self, responders: dict[str, Responder] | None = None):
        self.model = "fake"
        self.usage = Usage()
        self.responders: dict[str, Responder] = dict(responders or {})
        self.calls: list[tuple[str, str]] = []

    def on(self, name: str, responder: Responder) -> "FakeLLM":
        self.responders[name] = responder
        return self

    def _respond(self, name: str, system: str, prompt: str):
        self.calls.append((name, prompt))
        self.usage.add(len(prompt) // 4, 50, 0.0)
        if name not in self.responders:
            raise KeyError(f"FakeLLM has no responder for {name!r}")
        return self.responders[name](system, prompt)

    def structured(self, system: str, prompt: str, schema: type[T], *, effort: str = "medium") -> T:
        result = self._respond(schema.__name__, system, prompt)
        return result if isinstance(result, schema) else schema.model_validate(result)

    def text(self, system: str, prompt: str, *, effort: str = "medium") -> str:
        return str(self._respond("text", system, prompt))


def get_llm(provider: str | None = None) -> LLM:
    provider = (provider or os.getenv("AGENT_LLM_PROVIDER") or "").lower()
    if not provider:
        provider = "claude" if os.getenv("ANTHROPIC_API_KEY") else "ollama"
    if provider == "claude":
        return ClaudeLLM()
    if provider == "ollama":
        return OllamaLLM()
    if provider == "fake":
        return FakeLLM()
    raise ValueError(f"Unknown AGENT_LLM_PROVIDER {provider!r}")


def dumps(obj) -> str:
    """Compact JSON for prompts, accepting Pydantic models and plain data."""
    if isinstance(obj, BaseModel):
        return obj.model_dump_json(indent=1)
    return json.dumps(obj, indent=1, default=str)
