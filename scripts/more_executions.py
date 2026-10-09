"""What more executions show on cheap queries.

Movie Q2, Q3, Q4, Q8, and Q9 were re-executed beyond the main matrix:
gpt-oss-120B and Granite-8B toward 40 executions, Llama-70B (M) to 20. The
"_all" settings of experiments/processed/sembench_repeats/items.csv hold
every execution in order. Items with a missing output are dropped. Per
deployment, pooled over the five queries:

- Flip rate at k executions, k = 5, 10, 20, and the largest k reached: the
  exact expectation over k-subsets of the recorded executions.
- Items unanimous in the first five executions, and the share of them that
  change in any later execution (Wilson interval).
- Bounds: for the Boolean queries, the certain answer (items qualifying in
  all of the first five executions) and the possible answer (in any); how
  many later executions return a set between them.
- Voting: disagreement between two disjoint groups of three executions under
  plurality vote (ties to the earlier execution's output), against single
  executions; and, for Boolean items, the measured reduction and the one predicted from each
  item's minority share by q3 = p^3 + 3p^2(1-p).
- Drift: pairwise disagreement within the first campaign's executions,
  within the later ones, and across the two; an item bootstrap of
  cross minus the mean of the two within-campaign rates.

Bootstrap: items, 10,000 resamples, seed 11. Writes
experiments/processed/more_executions.json with a "macros" map.
"""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_sembench_repeats as base  # noqa: E402

ITEMS = base.PROCESSED / "items.csv"
OUT = base.PROCESSED.parent / "more_executions.json"
QUERIES = (2, 3, 4, 8, 9)
BOOLEAN = (2, 3, 4)
DEPLOYMENTS = {
    "managed-gpt-oss-120b": ("w20_all", "Gptoss"),
    "vllm-granite-3-3-8b-instruct": ("w20_lp_all", "Granite"),
    "managed-llama-3-3-70b-instruct": ("w20_all", "LlamaW"),
}
FIT, GROUP, VOTE_DRAWS = 5, 3, 200
RESAMPLES, SEED = 10_000, 11


def load() -> dict[str, dict[int, dict[str, list]]]:
    data: dict[str, dict[int, dict[str, list]]] = {model: {} for model in DEPLOYMENTS}
    if not ITEMS.exists():
        return data
    with ITEMS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            spec = DEPLOYMENTS.get(row["model"])
            query = int(row["query_id"])
            if row["scenario"] != "movie" or not spec or row["setting"] != spec[0] or query not in QUERIES:
                continue
            values = json.loads(row["values"])
            if any(value is None for value in values):
                continue
            data[row["model"]].setdefault(query, {})[row["item"]] = [json.dumps(value) for value in values]
    return data


def plurality(values: list[str]) -> str:
    counts = Counter(values)
    top = max(counts.values())
    return next(value for value in values if counts[value] == top)


def disagreement(values: list[str], pairs: list[tuple[int, int]]) -> float:
    return mean(values[i] != values[j] for i, j in pairs) if pairs else float("nan")


def bootstrap(values: list[float]) -> list[float]:
    rng = random.Random(SEED)
    draws = sorted(mean(rng.choices(values, k=len(values))) for _ in range(RESAMPLES))
    return [draws[int(0.025 * RESAMPLES)], draws[int(0.975 * RESAMPLES) - 1]]


def pct(value: float) -> str:
    return f"{100 * value:.1f}\\%"


def analyse(model: str, queries: dict[int, dict[str, list]]) -> dict | None:
    if not queries:
        return None
    k = min(len(values) for items in queries.values() for values in items.values())
    if k <= FIT:
        return None
    items = [(query, key, values[:k]) for query, by_item in sorted(queries.items()) for key, values in by_item.items()]
    main = base.MAIN_EXECUTIONS.get(model, base.MAIN_DEFAULT)
    flips = {m: mean(base.subset_flip(values, m) for _, _, values in items) for m in sorted({5, 10, 20, k}) if m <= k}

    unanimous = [values for _, _, values in items if len(set(values[:FIT])) == 1]
    later_change = sum(len(set(values)) > 1 for values in unanimous)

    bracketed = later = 0
    for query in BOOLEAN:
        rows = list(queries.get(query, {}).values())
        if not rows:
            continue
        rows = [values[:k] for values in rows]
        certain = {i for i, values in enumerate(rows) if all(v == "true" for v in values[:FIT])}
        possible = {i for i, values in enumerate(rows) if any(v == "true" for v in values[:FIT])}
        for r in range(FIT, k):
            chosen = {i for i, values in enumerate(rows) if values[r] == "true"}
            later += 1
            bracketed += certain <= chosen <= possible

    rng = random.Random(SEED)
    single, voted, theory_single, theory_voted = [], [], [], []
    bool_single, bool_voted = [], []
    for query, _, values in items:
        single.append(base.pair_disagreement(values))
        total = 0
        for _ in range(VOTE_DRAWS):
            chosen = rng.sample(range(k), 2 * GROUP)
            a = plurality([values[i] for i in sorted(chosen[:GROUP])])
            b = plurality([values[i] for i in sorted(chosen[GROUP:])])
            total += a != b
        voted.append(total / VOTE_DRAWS)
        if query in BOOLEAN:
            bool_single.append(single[-1])
            bool_voted.append(voted[-1])
            p = sum(v == "true" for v in values) / k
            q = p ** 3 + 3 * p ** 2 * (1 - p)
            theory_single.append(2 * p * (1 - p))
            theory_voted.append(2 * q * (1 - q))

    within_main, within_later, cross, gaps = [], [], [], []
    for _, _, values in items:
        early, late = range(min(main, k)), range(min(main, k), k)
        a = disagreement(values, list(combinations(early, 2)))
        b = disagreement(values, list(combinations(late, 2)))
        c = disagreement(values, [(i, j) for i in early for j in late])
        if late and len(early) >= 2 and len(late) >= 2:
            within_main.append(a)
            within_later.append(b)
            cross.append(c)
            gaps.append(c - (a + b) / 2)

    low, high = base.wilson(later_change, len(unanimous)) if unanimous else (float("nan"),) * 2
    single_mean, voted_mean = mean(single), mean(voted)
    return {
        "executions": k,
        "items": len(items),
        "flip": {str(m): value for m, value in flips.items()},
        "unanimous": len(unanimous),
        "later_change": later_change,
        "later_change_ci": [low, high],
        "bracketed": bracketed,
        "later": later,
        "vote_single": single_mean,
        "vote_three": voted_mean,
        "vote_reduction": 1 - voted_mean / single_mean if single_mean else None,
        "bool_reduction": 1 - mean(bool_voted) / mean(bool_single) if bool_single and mean(bool_single) else None,
        "theory_reduction": 1 - mean(theory_voted) / mean(theory_single) if theory_single and mean(theory_single) else None,
        "drift_within_main": mean(within_main) if within_main else None,
        "drift_within_later": mean(within_later) if within_later else None,
        "drift_cross": mean(cross) if cross else None,
        "drift_gap_ci": bootstrap(gaps) if gaps else None,
        "main_executions": main,
    }


def main() -> None:
    data = load()
    results = {model: analyse(model, queries) for model, queries in data.items()}
    macros: dict[str, str] = {}
    words = {5: "Five", 10: "Ten", 20: "Twenty"}
    for model, (_, suffix) in DEPLOYMENTS.items():
        r = results[model]
        if not r:
            continue
        macros[f"moreExec{suffix}"] = str(r["executions"])
        macros[f"moreItems{suffix}"] = f"{r['items']:,}".replace(",", "{,}")
        for m, value in r["flip"].items():
            name = "Max" if int(m) == r["executions"] else words[int(m)]
            macros[f"moreFlip{name}{suffix}"] = pct(value)
        macros[f"moreUnanimous{suffix}"] = f"{r['unanimous']:,}".replace(",", "{,}")
        macros[f"moreLaterChange{suffix}"] = pct(r["later_change"] / r["unanimous"]) if r["unanimous"] else "--"
        macros[f"moreLaterChangeCi{suffix}"] = "[{:.1f}, {:.1f}]".format(*(100 * v for v in r["later_change_ci"]))
        macros[f"moreBracketed{suffix}"] = f"{r['bracketed']}/{r['later']}"
        macros[f"moreVoteSingle{suffix}"] = pct(r["vote_single"])
        macros[f"moreVoteThree{suffix}"] = pct(r["vote_three"])
        if r["vote_reduction"] is not None:
            macros[f"moreVoteReduction{suffix}"] = pct(r["vote_reduction"])
        if r["bool_reduction"] is not None:
            macros[f"moreVoteBool{suffix}"] = pct(r["bool_reduction"])
        if r["theory_reduction"] is not None:
            macros[f"moreVoteTheory{suffix}"] = pct(r["theory_reduction"])
        if r["drift_cross"] is not None:
            macros[f"moreDriftCross{suffix}"] = pct(r["drift_cross"])
            macros[f"moreDriftWithinMain{suffix}"] = pct(r["drift_within_main"])
            macros[f"moreDriftWithinLater{suffix}"] = pct(r["drift_within_later"])
            macros[f"moreDriftCi{suffix}"] = "[{:+.1f}, {:+.1f}]".format(*(100 * v for v in r["drift_gap_ci"]))
    OUT.write_text(json.dumps({"results": results, "macros": macros}, indent=1) + "\n")
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
