#!/usr/bin/env python3
"""Print the model ids a LiteLLM proxy serves, and their prices where the proxy reports them."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sembench_repeats import ENDPOINTS, _env  # noqa: E402


def get(path: str):
    base = os.environ["OPENAI_API_BASE"].rstrip("/")
    request = urllib.request.Request(base + path, headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="local", choices=sorted(ENDPOINTS))
    args = parser.parse_args(argv)
    _env(args.endpoint)
    ids = sorted(model["id"] for model in get("/v1/models").get("data", []))
    prices = {}
    try:
        for entry in get("/model/info").get("data", []):
            info = entry.get("model_info") or {}
            prices[entry.get("model_name")] = {
                "input_per_mtok": (info.get("input_cost_per_token") or 0) * 1e6 or None,
                "output_per_mtok": (info.get("output_cost_per_token") or 0) * 1e6 or None,
                "logprobs": info.get("supports_logprobs") if "supports_logprobs" in info else None,
                "reasoning": info.get("supports_reasoning") if "supports_reasoning" in info else None,
            }
    except Exception as exc:
        print(json.dumps({"model_info": f"unavailable: {type(exc).__name__}"}))
    for model in ids:
        print(json.dumps({"model": model, **prices.get(model, {})}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
