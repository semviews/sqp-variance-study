#!/usr/bin/env python3
"""Item-level and answer-level variance of repeated SemBench executions.

Inputs, per cell experiments/raw/sembench-repeats/<scenario>/<model>/<setting>/repeat-<r>/:
  Q<q>.calls.jsonl  every model call, in input order
  Q<q>.csv          the answer SemBench's runner returned
and experiments/processed/sembench_repeats/scores.csv from score_sembench_repeats.py.

Outputs in experiments/processed/sembench_repeats/:
  items.csv    one row per (model, setting, query, item): outputs across repeats
  queries.csv  one row per (model, setting, query): item flip rate, answer
               reproduction, metric spread, and the reproduction predicted
               when items flip independently
No model is called.
"""

from __future__ import annotations

import csv
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "experiments" / "raw" / "sembench-repeats"
PROCESSED = ROOT / "experiments" / "processed" / "sembench_repeats"
REVIEWS = ROOT / "SemBench" / "files" / "movie" / "data" / "sf_2000" / "Reviews.csv"
SEED = 11
SIMULATIONS = 2000

# Output kind of the semantic operator each Movie query calls, and the
# plan-level transformation that turns item outputs into the answer.
MOVIE = {
    1: ("bool", "limit5"),
    2: ("bool", "limit5"),
    3: ("bool", "count"),
    4: ("bool", "ratio"),
    5: ("bool", "pairs_limit10"),
    6: ("bool", "pairs_limit10"),
    7: ("bool", "pairs"),
    8: ("label", "label_counts"),
    9: ("score", "vector"),
    10: ("score", "group_mean_rank"),
}
MEDICAL = {
    1: ("bool", "set"),
    4: ("bool", "mean_of_attribute"),
    10: ("extract", "vector"),
}
KINDS = {"movie": MOVIE, "medical": MEDICAL}
MEDICAL_DATA = ROOT / "SemBench" / "files" / "medical" / "data"


def medical_q4_ages() -> list[float]:
    """Ages in the row order of patients.join(symptoms, on=patient_id, how=inner)."""
    with (MEDICAL_DATA / "text_symptoms_data.csv").open(newline="", encoding="utf-8") as handle:
        symptoms = defaultdict(list)
        for row in csv.DictReader(handle):
            symptoms[row["patient_id"]].append(row)
    ages = []
    with (MEDICAL_DATA / "patient_data.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for _ in symptoms.get(row["patient_id"], []):
                ages.append(float(row["age"]))
    return ages


def lotus_answer(output: str) -> str:
    """LOTUS's cot_postprocessor: the text after the first "Answer:", or from offset 6 if there is none."""
    return output[output.find("Answer:") + len("Answer:"):]


def bool_defaulted(output: str | None) -> bool:
    answer = lotus_answer(output or "")
    return "True" not in answer and "False" not in answer


def score_value(output: str) -> tuple[float, bool]:
    """SemBench Movie Q9/Q10: float() of the reply if it lies in [1, 5], else 3.0. Returns (score, defaulted)."""
    try:
        number = float(output)
    except (TypeError, ValueError):
        return 3.0, True
    return (number, False) if 1 <= number <= 5 else (3.0, True)


def parse(kind: str, output: str | None):
    if output is None:
        return None
    text = output.strip()
    if kind == "bool":
        # LOTUS's filter_postprocess, which sem_filter and sem_join apply with default=True.
        answer = lotus_answer(output)
        return True if "True" in answer else False if "False" in answer else True
    if kind == "score":
        return score_value(output)[0]
    if kind == "label":
        # SemBench Movie Q8 groups the raw replies with value_counts.
        return output
    if kind == "extract":
        # LOTUS's extract_postprocess: json.loads of the whole reply, {} if it fails, values as str.
        try:
            payload = json.loads(output)
        except json.JSONDecodeError:
            return None
        value = payload.get("text_diagnosis") if isinstance(payload, dict) else None
        return None if value is None else str(value)
    return " ".join(text.casefold().split())


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (math.nan, math.nan)
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


COMMON_K = 5
MAIN_SETTINGS = {"w20", "w20_lp"}
ALL_SUFFIX = "_all"
MAIN_EXECUTIONS = {"managed-gpt-oss-120b": 10}
MAIN_DEFAULT = 5
MAX_SUBSETS = 3000


def seed_name(model: str) -> str:
    """Deployment name with the artifact's neutral service names, so simulations match in both trees."""
    return re.sub(r"(?<![a-z])vllm(?![a-z])", "vllm", model.replace("managed", "managed"))


def subset_flip(values: list, m: int = COMMON_K) -> float:
    """Probability that m runs drawn without replacement from the k recorded runs are not unanimous."""
    k = len(values)
    if k < m:
        return math.nan
    counts = Counter(str(value) for value in values)
    return 1.0 - sum(math.comb(n, m) for n in counts.values()) / math.comb(k, m)


def pair_disagreement(values: list) -> float:
    """Probability that two distinct runs, drawn without replacement, differ."""
    k = len(values)
    if k < 2:
        return math.nan
    counts = Counter(values)
    same = sum(n * (n - 1) for n in counts.values()) / (k * (k - 1))
    return 1.0 - same


DECISION = re.compile(r"^(true|false|[1-5]|pos|neg)", flags=re.I)


def decision_margin(record: dict) -> float | None:
    """Log-probability gap between the top two candidates at the first decision token."""
    for item in record.get("logprobs") or []:
        token = (item.get("token") or "").strip()
        if DECISION.match(token):
            top = item.get("top") or []
            if len(top) >= 2:
                return float(top[0][1]) - float(top[1][1])
            return math.inf
    return None


def read_calls(path: Path, kind: str) -> list[tuple[str, object, dict]]:
    """Ordered (item key, parsed output, raw record). Duplicate prompts get an occurrence suffix."""
    seen: Counter = Counter()
    rows = []
    for line in path.read_text().splitlines():
        record = json.loads(line)
        seen[record["key"]] += 1
        key = f"{record['key']}#{seen[record['key']]}"
        value = None if "error" in record else parse(kind, record.get("output"))
        rows.append((key, value, record))
    return rows


def canonical_answer(path: Path) -> str:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    header, body = rows[0], rows[1:]

    def norm(cell: str) -> str:
        """Integers stay exact, since they include IDs; other numbers absorb float rounding."""
        try:
            number = float(cell)
        except ValueError:
            return cell.strip()
        return str(int(number)) if number.is_integer() else f"{number:.6g}"

    return json.dumps([header, sorted([norm(cell) for cell in row] for row in body)])


def movie_of_items() -> dict[str, str]:
    """Map the first 120 characters of a review to its movie, for Q10."""
    mapping = {}
    with REVIEWS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            mapping[" ".join(row["reviewText"].split())[:120]] = row["id"]
    return mapping


def review_prefix(text: str) -> str:
    match = re.search(r"«(.*)", text, flags=re.S)
    body = match.group(1).split("»")[0] if match else text
    return " ".join(body.split())[:120]


def without_self_pairs(calls: list) -> list:
    """Drop the diagonal of an exact self-join, whose n*n calls run left-major."""
    n = math.isqrt(len(calls))
    if n * n != len(calls):
        raise ValueError(f"{len(calls)} join calls are not a square")
    return [call for index, call in enumerate(calls) if index // n != index % n]


def answer_from_items(plan: str, order: list[str], values: dict[str, object], groups: dict[str, str] | None):
    if plan in ("set", "pairs"):
        return frozenset(key for key in order if values.get(key) is True)
    if plan == "pairs_limit10":
        return tuple(key for key in order if values.get(key) is True)[:10]
    if plan == "mean_of_attribute":
        chosen = [groups[key] for key in order if values.get(key) is True and key in groups]
        return round(statistics.fmean(chosen), 9) if chosen else None
    if plan == "limit5":
        return tuple(key for key in order if values.get(key) is True)[:5]
    if plan == "count":
        return sum(values.get(key) is True for key in order)
    if plan == "ratio":
        return round(sum(values.get(key) is True for key in order) / len(order), 6)
    if plan == "label_counts":
        return tuple(sorted(Counter(values.get(key) for key in order).items(), key=str))
    if plan == "vector":
        return tuple(values.get(key) for key in order)
    if plan == "group_mean_rank":
        sums: dict[str, list] = defaultdict(list)
        for key in order:
            value = values.get(key)
            if value is not None and groups and key in groups:
                sums[groups[key]].append(value)
        means = {group: statistics.fmean(scores) for group, scores in sums.items()}
        return tuple(sorted(means, key=lambda group: (-round(means[group], 9), group)))
    raise ValueError(plan)


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    scores = defaultdict(list)
    score_path = PROCESSED / "scores.csv"
    if score_path.exists():
        with score_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["quality"] not in ("", "None"):
                    scores[(row["scenario"], row["model"], row["setting"], int(row["query_id"]))].append(
                        (int(row["repeat"]), float(row["quality"]))
                    )

    prefixes = None
    item_rows, query_rows = [], []
    cells = defaultdict(list)
    for metrics in sorted(RAW.glob("*/*/*/repeat-*/Q*.metrics.json")):
        query = int(re.match(r"Q(\d+)", metrics.name).group(1))
        repeat_dir = metrics.parent
        setting_dir = repeat_dir.parent
        model_dir = setting_dir.parent
        scenario = model_dir.parent.name
        setting = setting_dir.name
        if setting in MAIN_SETTINGS:
            cells[(scenario, model_dir.name, setting + ALL_SUFFIX, query)].append(repeat_dir)
        else:
            cells[(scenario, model_dir.name, setting, query)].append(repeat_dir)
    for key in [key for key in cells if key[2].endswith(ALL_SUFFIX)]:
        # The main matrix keeps the first clean executions in repeat order; later
        # ones enter only the "_all" view with every execution.
        # An execution that failed all its attempts is replaced by the next one.
        main_key = (key[0], key[1], key[2][: -len(ALL_SUFFIX)], key[3])
        ordered = sorted(cells[key], key=lambda path: int(path.name.split("-")[1]))
        clean = [path for path in ordered
                 if not json.loads((path / f"Q{key[3]}.metrics.json").read_text()).get("call_errors")]
        cells[main_key] = clean[: MAIN_EXECUTIONS.get(key[1], MAIN_DEFAULT)]
        if len(clean) <= len(cells[main_key]):
            del cells[key]

    for (scenario, model, setting, query), repeat_dirs in sorted(cells.items()):
        if scenario not in KINDS or query not in KINDS[scenario]:
            continue
        kind, plan = KINDS[scenario][query]
        runs = []
        for repeat_dir in sorted(repeat_dirs, key=lambda path: int(path.name.split("-")[1])):
            calls_path = repeat_dir / f"Q{query}.calls.jsonl"
            answer_path = repeat_dir / f"Q{query}.csv"
            meta = json.loads((repeat_dir / f"Q{query}.metrics.json").read_text())
            if meta.get("call_errors"):
                continue
            if calls_path.exists() and answer_path.exists():
                runs.append((repeat_dir, read_calls(calls_path, kind), canonical_answer(answer_path)))
        if len(runs) < 2:
            continue
        if plan.startswith("pairs"):
            # A cascade calls a data-dependent subset of pairs, so items exist only for exact joins.
            if "_exact" not in setting:
                continue
            runs = [(path, without_self_pairs(calls), answer) for path, calls, answer in runs]

        order = [key for key, _value, _record in runs[0][1]]
        groups = None
        if plan == "group_mean_rank":
            if prefixes is None:
                prefixes = movie_of_items()
            groups = {
                key: prefixes.get(review_prefix(record.get("text", "")))
                for key, _value, record in runs[0][1]
            }
            groups = {key: value for key, value in groups.items() if value}
        if plan == "mean_of_attribute":
            ages = medical_q4_ages()
            if len(ages) != len(order):
                raise ValueError(f"Q{query}: {len(ages)} joined rows but {len(order)} calls")
            groups = dict(zip(order, ages))
        per_item: dict[str, list] = defaultdict(list)
        unparsed = errors = 0
        total_calls = 0
        reasoning = defaultdict(list)
        margins = defaultdict(list)
        for _dir, calls, _answer in runs:
            for key, value, record in calls:
                per_item[key].append(value)
                total_calls += 1
                unparsed += "error" not in record and (
                    bool_defaulted(record.get("output")) if kind == "bool"
                    else score_value(record.get("output"))[1] if kind == "score"
                    else value is None)
                errors += "error" in record
                reasoning[key].append(record.get("reasoning_chars") or 0)
                margin = decision_margin(record)
                if margin is not None:
                    margins[key].append(margin)
        complete = {key: values for key, values in per_item.items() if len(values) == len(runs)}
        flips = 0
        disagreements = []
        beyond_one = []
        flips_common_k = []
        for key, values in complete.items():
            flipped = len(set(values)) > 1
            flips += flipped
            disagreements.append(pair_disagreement(values))
            flips_common_k.append(subset_flip(values))
            if kind == "score":
                numeric = [value for value in values if isinstance(value, (int, float))]
                pairs = list(combinations(numeric, 2))
                beyond_one.append(sum(abs(a - b) > 1 for a, b in pairs) / len(pairs) if pairs else math.nan)
            item_rows.append(
                {
                    "scenario": scenario,
                    "model": model,
                    "setting": setting,
                    "query_id": query,
                    "item": key,
                    "values": json.dumps(values, default=str),
                    "flipped": int(flipped),
                    "flip_k5": subset_flip(values),
                    "pair_disagreement": pair_disagreement(values),
                    "mean_reasoning_chars": statistics.fmean(reasoning[key]) if reasoning[key] else 0,
                    "first_reasoning_chars": reasoning[key][0] if reasoning[key] else 0,
                    "min_margin": min(margins[key]) if margins[key] else "",
                    "first_margin": margins[key][0] if margins[key] else "",
                }
            )

        answers = [answer for _dir, _calls, answer in runs]
        answer_pairs = list(combinations(answers, 2))
        reproduced = sum(a == b for a, b in answer_pairs) / len(answer_pairs)

        # Independence model: draw each item's output from its own empirical
        # distribution across repeats, independently, and ask how often two
        # simulated answers coincide.
        empirical = {key: values for key, values in complete.items()}
        rng = random.Random(f"{SEED}:{scenario}:{seed_name(model)}:{setting}:{query}")
        simulated_same = 0
        for _ in range(SIMULATIONS):
            left = {key: rng.choice(values) for key, values in empirical.items()}
            right = {key: rng.choice(values) for key, values in empirical.items()}
            simulated_same += answer_from_items(plan, order, left, groups) == answer_from_items(plan, order, right, groups)
        # Held-out check of the same model: fit item distributions on the first
        # half of the runs and predict reproduction among the second half.
        heldout_predicted = heldout_measured = ""
        if len(runs) >= 2 * COMMON_K:
            fit = {key: values[:COMMON_K] for key, values in complete.items()}
            same = 0
            for _ in range(SIMULATIONS):
                left = {key: rng.choice(values) for key, values in fit.items()}
                right = {key: rng.choice(values) for key, values in fit.items()}
                same += answer_from_items(plan, order, left, groups) == answer_from_items(plan, order, right, groups)
            heldout_predicted = same / SIMULATIONS
            later = list(combinations(answers[COMMON_K:], 2))
            heldout_measured = sum(a == b for a, b in later) / len(later)

        # The without-replacement analogue of the measured rate.
        reconstructed = [
            answer_from_items(plan, order, {key: per_item[key][index] for key in complete}, groups)
            for index in range(len(runs))
        ]
        reconstructed_pairs = list(combinations(reconstructed, 2))

        quality_by_repeat = dict(scores.get((scenario, model, setting, query), []))
        run_quality = [quality_by_repeat.get(int(path.name.split("-")[1])) for path, _calls, _answer in runs]
        quality = [value for value in run_quality if value is not None]

        # Share of COMMON_K-subsets of the executions that agree at each level:
        # every item output, the final answer, and the benchmark's quality.
        levels = {"outputs": 0, "answer": 0, "quality": 0}
        size = min(COMMON_K, len(runs))
        if math.comb(len(runs), size) <= MAX_SUBSETS:
            subsets = list(combinations(range(len(runs)), size))
        else:
            subset_rng = random.Random(f"{SEED}:subsets:{scenario}:{seed_name(model)}:{setting}:{query}")
            subsets = [tuple(sorted(subset_rng.sample(range(len(runs)), size))) for _ in range(MAX_SUBSETS)]
        for subset in subsets:
            levels["outputs"] += all(len({values[index] for index in subset}) == 1 for values in complete.values())
            levels["answer"] += len({answers[index] for index in subset}) == 1
            levels["quality"] += len({round(run_quality[index], 9) for index in subset
                                      if run_quality[index] is not None}) <= 1
        low, high = wilson(flips, len(complete))
        query_rows.append(
            {
                "scenario": scenario,
                "model": model,
                "setting": setting,
                "query_id": query,
                "kind": kind,
                "plan": plan,
                "repeats": len(runs),
                "items": len(complete),
                "calls": total_calls,
                "call_errors": errors,
                "unparsed": unparsed,
                "item_flip_rate": flips / len(complete) if complete else math.nan,
                "item_flip_low": low,
                "item_flip_high": high,
                "item_flip_rate_k5": statistics.fmean(flips_common_k) if flips_common_k else math.nan,
                "item_pair_disagreement": statistics.fmean(disagreements) if disagreements else math.nan,
                "item_beyond_one": statistics.fmean([v for v in beyond_one if v == v]) if beyond_one else "",
                "answer_reproduction": reproduced,
                "answer_reproduction_from_items": sum(a == b for a, b in reconstructed_pairs) / len(reconstructed_pairs),
                "answer_reproduction_independent": simulated_same / SIMULATIONS,
                "heldout_predicted": heldout_predicted,
                "heldout_measured": heldout_measured,
                "distinct_answers": len(set(answers)),
                "outputs_identical_k5": levels["outputs"] / len(subsets),
                "answer_identical_k5": levels["answer"] / len(subsets),
                "quality_identical_k5": levels["quality"] / len(subsets) if quality else "",
                "quality_values": json.dumps([round(value, 4) for value in quality]),
                "quality_mean": statistics.fmean(quality) if quality else "",
                "quality_min": min(quality) if quality else "",
                "quality_max": max(quality) if quality else "",
                "quality_range": (max(quality) - min(quality)) if quality else "",
            }
        )

    for name, rows in (("items", item_rows), ("queries", query_rows)):
        if not rows:
            continue
        with (PROCESSED / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for row in query_rows:
        print(
            f"{row['model'][:28]:28s} {row['setting']:8s} Q{row['query_id']:<2d} k={row['repeats']:2d} n={row['items']:4d} "
            f"flip={row['item_flip_rate']:.3f} repro={row['answer_reproduction']:.2f} "
            f"indep={row['answer_reproduction_independent']:.2f} distinct={row['distinct_answers']} "
            f"q={row['quality_values']}"
        )


if __name__ == "__main__":
    main()
