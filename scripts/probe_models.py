#!/usr/bin/env python3
"""Check which proxy models answer a LOTUS-style filter prompt at temperature 0.

For each model: whether temperature 0 is accepted, whether log-probabilities
come back, the fingerprint, token usage (including reasoning tokens), and the
cost the proxy reports, if any. A model that rejects log-probabilities is
retried without them.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sembench_repeats import ENDPOINTS, _env  # noqa: E402

MODELS = (
    "openai/managed-gpt-oss-120b",
    "openai/vllm-gpt-oss-120b",
    "openai/managed-llama-3-3-70b-instruct",
    "openai/vllm-llama-3-3-70b-instruct",
    "openai/vllm-qwen2-5-72b-instruct",
    "openai/managed-mistral-small-3-1-24b-2503",
    "openai/vllm-granite-3-3-8b-instruct",
    "openai/managed-gpt-oss-20b",
    "openai/vllm-qwen3-vl-235b-a22b-instruct",
    "openai/vllm-qwen2-vl-72b-instruct",
)
PROMPT = (
    "Context:\n[Reviewtext]: «A tense, well-made thriller with a weak ending.»\n\n\n"
    'Claim: Determine if the following movie review is clearly positive. Review: "Reviewtext".\n'
    "Answer with True or False. Format: Answer: <True|False>"
)


def probe(litellm, model: str, logprobs: bool, extra: dict, token_arg: str = "max_tokens", prompt: str = PROMPT) -> dict:
    kwargs = {"temperature": 0.0, token_arg: 2048, "timeout": 120, **extra}
    if logprobs:
        kwargs.update(logprobs=True, top_logprobs=5)
    started = time.time()
    response = litellm.completion(model=model, messages=[{"role": "user", "content": prompt}], **kwargs)
    choice = response.choices[0]
    usage = getattr(response, "usage", None)
    details = getattr(usage, "completion_tokens_details", None)
    reasoning = getattr(choice.message, "reasoning_content", None)
    hidden = getattr(response, "_hidden_params", {}) or {}
    headers = hidden.get("additional_headers") or {}
    return {
        "model": model,
        "ok": True,
        "seconds": round(time.time() - started, 1),
        "output": (choice.message.content or "")[:40],
        "logprobs": bool(getattr(getattr(choice, "logprobs", None), "content", None)),
        "fingerprint": getattr(response, "system_fingerprint", None),
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "reasoning_tokens": getattr(details, "reasoning_tokens", None),
        "reasoning_chars": len(reasoning) if reasoning else 0,
        "cost": hidden.get("response_cost") or headers.get("llm_provider-x-litellm-response-cost")
        or headers.get("x-litellm-response-cost"),
        **({"extra": extra} if extra else {}),
        **({"prompt": prompt[:60]} if prompt != PROMPT else {}),
    }


def probe_raw(model: str, token_arg: str, temperature: float = 0.0) -> dict:
    """The same request through the OpenAI client, so only the proxy and provider can refuse it."""
    import os

    from openai import OpenAI

    client = OpenAI(base_url=os.environ["OPENAI_API_BASE"], api_key=os.environ["OPENAI_API_KEY"], timeout=120)
    name = model.split("/", 1)[1] if model.startswith("openai/") else model
    started = time.time()
    try:
        response = client.chat.completions.create(
            model=name, messages=[{"role": "user", "content": PROMPT}], temperature=temperature, **{token_arg: 2048}
        )
    except Exception as exc:
        return {"model": model, "client": "openai", "temperature": temperature, "ok": False, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
    usage = response.usage
    details = getattr(usage, "completion_tokens_details", None)
    return {
        "model": model,
        "client": "openai",
        "temperature": temperature,
        "ok": True,
        "seconds": round(time.time() - started, 1),
        "output": (response.choices[0].message.content or "")[:40],
        "fingerprint": response.system_fingerprint,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "reasoning_tokens": getattr(details, "reasoning_tokens", None),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="local", choices=sorted(ENDPOINTS))
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--extra", default=None, help="JSON of extra completion arguments, e.g. a thinking setting")
    parser.add_argument("--token-arg", default="max_tokens", help="name of the output-token limit (newer OpenAI models take max_completion_tokens)")
    parser.add_argument("--prompt-file", default=None, help="file whose text replaces the default filter prompt")
    parser.add_argument("--raw", action="store_true", help="send through the OpenAI client instead of LiteLLM")
    parser.add_argument("--temperature", type=float, default=0.0, help="temperature for --raw probes")
    args = parser.parse_args(argv)
    _env(args.endpoint)
    if args.raw:
        for model in args.models:
            print(json.dumps(probe_raw(model, args.token_arg, args.temperature)), flush=True)
        return
    import litellm

    extra = json.loads(args.extra) if args.extra else {}
    prompt = Path(args.prompt_file).read_text() if args.prompt_file else PROMPT
    for model in args.models:
        record = None
        for logprobs in (True, False):
            try:
                record = probe(litellm, model, logprobs, extra, args.token_arg, prompt)
                if args.token_arg != "max_tokens":
                    record["token_arg"] = args.token_arg
                break
            except Exception as exc:
                record = {"model": model, "ok": False, "logprobs_requested": logprobs,
                          "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
