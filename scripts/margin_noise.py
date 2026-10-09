#!/usr/bin/env python3
"""One mechanism for contested items, margins, and flips.

For every item of a deployment that returns log-probabilities, the first
execution's decision margin m (log-probability gap between the two most
likely tokens at the deciding token) and the number of later executions
whose output differs from the first. If every execution adds Gaussian noise
of scale sigma to the logit gap, a later execution differs from the first
with probability Phi(-m / sigma). A margin-independent floor lambda absorbs
changes that no small noise explains (for example a deciding token outside
the logged prefix): P = lambda + (1 - lambda) Phi(-m / sigma). Both are
fitted by maximum likelihood (binomial over later executions, grid search);
intervals are percentile bootstraps over items (10,000 resamples, seed 11).

Writes experiments/processed/sembench_repeats/margin_noise.json with the
observed share per margin bin and the fit, consumed by
report_sembench_repeats.py (Figure: sources, left) and paper_macros.py.
No model is called.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments" / "processed" / "sembench_repeats"
OUT = DATA / "margin_noise.json"
MAIN = {"w20", "w20_lp"}
SEED = 11
RESAMPLES = 10_000
BINS = [0, 0.25, 0.5, 1, 2, 4, 8, math.inf]
GRID = np.exp(np.linspace(math.log(0.02), math.log(20), 300))
FLOORS = np.concatenate([[0.0], np.exp(np.linspace(math.log(1e-5), math.log(3e-2), 40))])
LABELS = {
    "vllm-granite-3-3-8b-instruct": ("Granite-8B", "Granite"),
    "vllm-llama-3-3-70b-instruct": ("Llama-70B (V)", "LlamaR"),
    "vllm-qwen2-5-72b-instruct": ("Qwen-72B", "Qwen"),
    "Azure-gpt-4o": ("GPT-4o", "GptFour"),
}


def normal_cdf(x: np.ndarray) -> np.ndarray:
    from math import erf

    return 0.5 * (1 + np.vectorize(erf)(x / math.sqrt(2)))


def main() -> None:
    per_model: dict[str, list[tuple[float, int, int]]] = {}
    with (DATA / "items.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["setting"] not in MAIN or row["model"] not in LABELS or row["first_margin"] in ("", "inf"):
                continue
            values = [str(value) for value in json.loads(row["values"])]
            later = values[1:]
            per_model.setdefault(row["model"], []).append(
                (float(row["first_margin"]), sum(value != values[0] for value in later), len(later)))

    rng = np.random.default_rng(SEED)
    result = {}
    for model, rows in per_model.items():
        margins = np.array([row[0] for row in rows])
        differ = np.array([row[1] for row in rows])
        trials = np.array([row[2] for row in rows])
        # Items with equal (margin, changes, trials) share a likelihood; margins are quantized.
        unique, group = np.unique(np.stack([margins, differ, trials], axis=1), axis=0, return_inverse=True)
        group = group.ravel()
        um, ud, ut = unique[:, 0], unique[:, 1], unique[:, 2]
        phi = normal_cdf(-um[:, None] / GRID[None, :])
        p = FLOORS[None, None, :] + (1 - FLOORS[None, None, :]) * phi[:, :, None]
        p = np.clip(p, 1e-12, 1 - 1e-12)
        loglik = (ud[:, None, None] * np.log(p) + (ut - ud)[:, None, None] * np.log(1 - p)).reshape(len(unique), -1)
        weights = np.bincount(group, minlength=len(unique))
        flat = int(np.argmax(weights @ loglik))
        best, floor = GRID[flat // len(FLOORS)], FLOORS[flat % len(FLOORS)]
        # Without the floor, for comparison.
        no_floor = loglik.reshape(len(unique), len(GRID), len(FLOORS))[:, :, 0]
        best_no_floor = GRID[int(np.argmax(weights @ no_floor))]
        n = len(rows)
        sigmas = np.empty(RESAMPLES)
        floors = np.empty(RESAMPLES)
        for index in range(RESAMPLES):
            counts = np.bincount(group[rng.integers(0, n, n)], minlength=len(unique))
            pick = int(np.argmax(counts @ loglik))
            sigmas[index], floors[index] = GRID[pick // len(FLOORS)], FLOORS[pick % len(FLOORS)]
        bins = []
        for low, high in zip(BINS, BINS[1:]):
            mask = (margins >= low) & (margins < high)
            if mask.sum() == 0:
                continue
            observed = differ[mask].sum() / trials[mask].sum()
            centre = float(np.median(margins[mask]))
            bins.append({"low": low, "high": None if math.isinf(high) else high, "items": int(mask.sum()),
                         "median_margin": centre, "observed": float(observed),
                         "fitted": float(floor + (1 - floor) * 0.5 * math.erfc(centre / best / math.sqrt(2)))})
        result[model] = {
            "label": LABELS[model][0], "macro": LABELS[model][1], "items": n,
            "items_changing": int((differ > 0).sum()),
            "sigma": float(best),
            "sigma_ci": [float(np.percentile(sigmas, 2.5)), float(np.percentile(sigmas, 97.5))],
            "floor": float(floor),
            "floor_ci": [float(np.percentile(floors, 2.5)), float(np.percentile(floors, 97.5))],
            "sigma_without_floor": float(best_no_floor),
            "share_changes_small_margin": float(differ[margins < 1].sum() / max(differ.sum(), 1)),
            "share_items_small_margin": float((margins < 1).mean()),
            "bins": bins,
        }
    OUT.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    macros = {}
    for values in result.values():
        name = values["macro"]
        macros[f"sigma{name}"] = f"{values['sigma']:.2f}"
        macros[f"sigmaCi{name}"] = f"[{values['sigma_ci'][0]:.2f}, {values['sigma_ci'][1]:.2f}]"
        macros[f"floor{name}"] = f"{100 * values['floor']:.2f}\\%"
        macros[f"smallMarginChanges{name}"] = f"{100 * values['share_changes_small_margin']:.0f}\\%"
        macros[f"smallMarginItems{name}"] = f"{100 * values['share_items_small_margin']:.1f}\\%"
        macros[f"changingItems{name}"] = values["items_changing"]
        first, last = values["bins"][0], values["bins"][-1]
        macros[f"differLowMargin{name}"] = f"{100 * first['observed']:.0f}\\%"
        macros[f"differHighMargin{name}"] = f"{100 * last['observed']:.2f}\\%"
    (DATA / "margin_noise_macros.json").write_text(json.dumps({"macros": macros}, indent=1) + "\n", encoding="utf-8")
    for model, values in result.items():
        print(model, values["items"], values["items_changing"], round(values["sigma"], 3), [round(v, 3) for v in values["sigma_ci"]],
              "floor", values["floor"], values["floor_ci"], "no-floor sigma", round(values["sigma_without_floor"], 3),
              "changes at m<1", round(values["share_changes_small_margin"], 3), "items at m<1", round(values["share_items_small_margin"], 3))
        for row in values["bins"]:
            print("   ", row)


if __name__ == "__main__":
    main()
