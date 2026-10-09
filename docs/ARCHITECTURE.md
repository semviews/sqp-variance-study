# How the code works

The code has two halves. The **drivers** execute a benchmark query through
the benchmark's own harness, once per execution, and log every model call.
The **analysis** reads those logs and computes everything the paper reports.
No analysis script calls a model.

```
benchmark harness (SemBench / LRO-Bench / UDA-Bench, unmodified)
        │  wrapped by a driver in scripts/ that logs each call
        ▼
experiments/raw/<suite>/<deployment>/<setting>/repeat-<r>/     one directory per execution
        │  analyze_*.py, score_*.py
        ▼
experiments/processed/                                          items.csv, queries.csv, *.json
        │  report_*.py, plot_*.py, and the analyses with a "macros" map
        ▼
outputs/tables/*.tex, outputs/figures/*.pdf, outputs/tables/macros.tex
        │  scripts/compare_reference.py
        ▼
reference/tables/   (the tables and macros as submitted)
```

Related documents: [DATA.md](DATA.md) gives the schema of every file named
here, and [REPRODUCE.md](REPRODUCE.md) gives the commands.

## 1. Principles

- **Fidelity.** Prompts, query plans, output parsers, and evaluators are the
  benchmarks'. Behaviour is changed only by subclassing or wrapping benchmark
  code from this repository. The benchmark checkouts are never edited.
- **One directory per execution.** Each execution of a query is a *cell*,
  identified by suite, deployment, setting, repeat number, and query. A cell
  is written once. Every driver skips finished cells, so a launcher can be
  re-run to resume or fill gaps.
- **Failures are recorded, never imputed.** An execution with any failed
  model call is a failure, not an answer. It is appended to a failure record
  and retried at most three times in total. Analyses use only clean
  executions.
- **Numbers come from code.** Every number in the paper's prose is a LaTeX
  macro written by `scripts/paper_macros.py`, and every table is a generated
  file. A value that is missing renders as a red `[pending]`.

## 2. Drivers

### SemBench: `scripts/sembench_repeats.py`

This is the main driver. It runs SemBench's Movie and Medical text queries
through SemBench's own LOTUS runner.

1. It loads SemBench's `LotusRunner` class for the scenario from the
   SemBench checkout and subclasses it. The query code and prompts are
   SemBench's.
2. `_configure_lm` builds the LOTUS `LM` at temperature 0 with the requested
   number of concurrent workers. It replaces the LM's
   `_process_uncached_messages` with a wrapper. The wrapper passes every
   batch to the original method and then writes one record per request with
   `_record`. A record holds the request key, a prompt excerpt, the start
   time, the output, the reasoning text if the deployment returns one, token
   counts, the system fingerprint, the cost the gateway reports, and
   optionally the first four output tokens with their top-5 log-probabilities.
3. LOTUS's cache is disabled, so duplicate input rows are sent as separate
   requests. SemBench's scenario setup is disabled, because it regenerates
   input CSV files and concurrent drivers would race on them.
4. A request that the gateway refused for rate limiting got no model output.
   It is sent again (up to five rounds), because a refusal is not a sample of
   the model. The record notes the number of resends.
5. `run_cell` executes one query. If the runner succeeds and no call failed,
   it writes `Q<q>.calls.jsonl`, the answer `Q<q>.csv`, and
   `Q<q>.metrics.json`. Otherwise it appends a record to
   `Q<q>.failures.jsonl`. A cell with three failure records is not retried.
   If a provider's content policy rejected all three attempts of one
   execution, the driver skips the query's other executions on that
   deployment.

Options select the deployment (`--model`), the gateway (`--endpoint`), the
number of executions (`--repeats`, `--first-repeat`), and the factor a probe
varies: `--workers`, `--shuffle` (input order, seeded by the repeat number),
`--reasoning` (reasoning effort), and `--logprobs`. `--policy exact` and
`--data-dir` run the semantic joins exactly on a reduced input (below).
`--tag` keeps probes apart from the main matrix. `--max-cost` stops a driver
before the next cell once the spend logged on that gateway reaches a cap;
`logged_spend` sums the cost of every logged call, including failed attempts.

The setting name encodes these options, for example `w20_lp` (20 workers,
log-probabilities) or `w8_lp_probe`. `setting_tag` builds it.

### Movie joins: `scripts/make_movie_join_subset.py`

SemBench's sampler keeps all 256 reviews of the film that Movie Q5 to Q7
self-join at every scale factor, so the quadratic join cannot be made
smaller through the scale factor. This script copies SemBench's `sf_2000`
inputs into `experiments/data/movie-join40/` and keeps a seeded sample of
40 distinct reviews of that film. SemBench's gold SQL is then evaluated on
the same files. The joins run with `--policy exact`, which sends every pair
to the model. LOTUS's sampled join cascade is not used.

### LRO-Bench: `scripts/lrobench_repeats.py`

It imports and runs LRO-Bench's own evaluation pipelines (`eval/*.py`)
unchanged, once per execution. It patches LRO-Bench's model caller to log
every call and to tag it as an operator call or an LLM-judge call. It also
captures the prediction that each metric receives, so two executions can be
compared on the answer and not only on the score. The settings follow
LRO-Bench's evaluation scripts. One change: LRO-Bench's impute metric uses
the model under test as its judge, and here the judge is fixed to one
deployment (gpt-oss-120B (M)) for every run. Otherwise a change in the judge
could change the score of an unchanged prediction. A failed execution (the
pipeline could not parse a reply) is written as `<query>.failed.json` and
counts as its own outcome.

### UDA-Bench: `scripts/uda_repeats.py` and `scripts/score_uda_repeats.py`

UDA-Bench's LOTUS adapter answers a query in two steps. It extracts each
attribute the query needs from every document with one `sem_map` call per
document and attribute. It then runs the query as SQL over the extracted
tables. `uda_repeats.py` performs the extraction for every attribute that
the Player dataset's queries use, with the adapter's instruction,
descriptions, and examples, once per execution. Each value is cut to its
first non-empty line, coerced to the type of the matching ground-truth
column, and the adapter's "empty" marker becomes NULL. Raw outputs are kept
next to the coerced tables.

`score_uda_repeats.py` runs in UDA-Bench's own environment. It executes each
query's SQL with UDA-Bench's `GtRunner` over the extracted tables and scores
the result with UDA-Bench's `run_eval` in lexical mode, so no model is
involved in scoring. Queries that fail on the ground truth itself are
excluded and recorded in `gt_check.json`.

### Temperature control: `scripts/temperature_control.py`

This control rebuilds the exact LOTUS filter requests of Movie Q1 and Q3
with LOTUS's own formatter. It keeps only requests whose hash equals a key
recorded in the main campaign, so they are the requests the benchmark sent.
A manifest fixes the items (seed 11): half changed in the main campaign and
half did not. Each item is sent 20 times at concurrency 1 in each of three
settings: temperature 0, temperature 0 with a fixed seed, and temperature 1.
Every response is logged with its full reasoning text.

### Product reviews: `scripts/run_product_reviews.py`

Thirteen filter and map tasks over 100 synthetic product reviews
(`experiments/data/product-reviews/`), five executions each, one LOTUS
`sem_filter` or `sem_map` call per task and execution with the cache
disabled.

### Endpoint checks: `scripts/probe_models.py` and `scripts/list_models.py`

These scripts list the models a gateway serves and check whether a model
accepts temperature 0 and returns log-probabilities. They are not needed to
reproduce any result.

### Launchers: `scripts/run_*.sh`

Each launcher starts one campaign: one driver process per deployment, in
parallel, with logs under `experiments/logs/`. The launchers are listed with
their call counts in [REPRODUCE.md](REPRODUCE.md#5-level-c-re-execute-the-benchmarks).

## 3. Analysis pipeline

`scripts/rebuild.sh` runs every step in order. Each step reads the raw
logs or earlier processed files and writes processed files, tables, or
figures. The order matters only where a step reads another's output.

| Step | Script | Reads | Writes |
|---|---|---|---|
| Published results | `analyze_published_repeats.py` | SemBench's published per-run metrics | `processed/published_repeats/*.csv` |
| | `report_published_repeats.py` | the above | `tables/published.tex`, `figures/published_ranges.pdf` |
| | `published_engines.py` | the above | `processed/published_engines.json` |
| Score SemBench | `score_sembench_repeats.py` | raw SemBench answers, SemBench's evaluator | `processed/sembench_repeats/scores.csv` |
| Items and answers | `analyze_sembench_repeats.py` | raw call logs, answers, scores | `processed/sembench_repeats/{items,queries}.csv` |
| Margins | `margin_noise.py` | `items.csv` | `margin_noise.json` |
| SemBench report | `report_sembench_repeats.py` | `items.csv`, `queries.csv` | `tables/{repeats,queries}.tex`, `figures/repeats_*.pdf`, `facts.json` |
| Supplementary analyses | `boost_analyses.py` | processed files and raw cells | `boost.json`, `tables/distances.tex` |
| More executions | `more_executions.py` | `items.csv` (all executions) | `processed/more_executions.json` |
| Commercial models | `commercial_report.py` | raw commercial cells | `processed/commercial.json`, `tables/commercial.tex` |
| Campaigns | `campaigns.py` | every call log | `processed/campaigns.json` |
| Remedies | `remedies_sembench_repeats.py` | raw cells | `remedies.csv`, `tables/remedies.tex` |
| | `selective_vote.py`, `answer_bounds.py` | raw cells, `items.csv` | `selective_vote.json`, `answer_bounds.json` |
| Cause probes | `report_probes.py` | probe settings in `items.csv`, `queries.csv` | `tables/probes.tex`, `probes.json` |
| Temperature control | `temperature_analysis.py`, `reasoning_divergence.py` | `raw/controls/temperature/` | `processed/controls/*.json` |
| LRO-Bench | `analyze_lrobench_repeats.py`, `lrobench_failures.py`, `claim_audit.py` | `raw/lrobench/`, LRO-Bench metadata | `processed/lrobench_repeats/`, `lrobench_failures.json`, `claims.json`, `tables/claims.tex` |
| UDA-Bench | `score_uda_repeats.py`, `analyze_uda_repeats.py`, `uda_lineage.py` | `raw/uda-repeats/`, UDA-Bench ground truth | `processed/uda_repeats/`, `uda_lineage.json` |
| Exposure rules | `exposure_rules.py` | `queries.csv`, `uda_lineage.json` | `exposure.json`, `tables/exposure.tex` |
| Other suites | `report_text_suites.py` | processed joins, LRO-Bench, UDA-Bench, product reviews | `tables/{joins,lro_repeats,uda,product}.tex`, `text_suites_facts.json` |
| Examples | `report_item_examples.py` | raw call logs | `duplicates.json`, `tables/examples.tex` |
| Three levels | `report_ladder.py` | processed files of all suites | `ladder.json`, `tables/ladder.tex`, `figures/ladder.pdf` |
| Cascades | `cascade_replay.py` | `items.csv` | `cascade.json`, `tables/cascade.tex` |
| Executions needed | `runs_needed.py` | published cells, `queries.csv` | `runs_needed.json` |
| Figures | `plot_published.py`, `plot_teaser.py`, `plot_amplification.py` | processed files, raw Movie Q3 answers | `figures/{published,teaser_q3,amplification}.pdf` |
| Macros | `paper_macros.py` | every facts file and every `"macros"` map | `tables/macros.tex` |

Paths in the table are relative to `experiments/` or `outputs/`. Some steps
write tables that the paper does not include (for example `distances.tex`
and `product.tex`); their numbers reach the paper through macros.

`paper_macros.py` defines macros from the facts files directly, and it also
merges every processed JSON file that carries a `"macros"` map. An analysis
added later only needs to write such a map to make its numbers available to
the paper.

## 4. Measures

Definitions follow Section 3 of the paper. The functions named here are in
`scripts/analyze_sembench_repeats.py` unless stated otherwise.

- **Item.** One model call of an operator: one row of a filter or map, one
  pair of a join, one document and attribute of an extraction. Items are
  keyed by the hash of the request (`key` in the call log) plus an
  occurrence number, because inputs contain duplicate rows.
- **Output.** The call's reply after the benchmark's own parser:
  `parse` mirrors LOTUS's filter rule (text after the first `Answer:`, a row
  is kept unless the reply says False), SemBench's score rule (a number in
  1 to 5, otherwise 3), and LOTUS's JSON extraction rule.
- **Item flip rate over k executions** (`subset_flip`). The probability
  that k executions drawn without replacement from the recorded ones do not
  all give the item the same output. It is computed exactly from the counts
  of each output. Flip rates grow with k, so deployments are compared at
  k = 5; with more recorded executions, the value is the exact expectation
  over all subsets of five.
- **Pairwise disagreement** (`pair_disagreement`). The probability that two
  distinct executions differ on the item. It does not depend on k.
- **Answer reproduction.** The share of pairs of executions whose final
  answers are identical. An answer is the result table the benchmark's
  runner returned, compared as a sorted list of rows with numbers
  normalized (`canonical_answer`).
- **Possible-worlds prediction.** If items change independently and the
  answer exposes every item, reproduction is the product of each item's
  agreement, approximately `exp(-x)` with `x` the sum of the items'
  pairwise disagreements. For answers under `LIMIT`, aggregates, and
  rankings, the prediction is obtained by simulating independent item
  outcomes through the query's plan (`answer_from_items`, 2,000
  simulations).
- **Decision margin** (`decision_margin`). For deployments that return
  log-probabilities, the gap between the two most likely tokens at the
  first token that decides the output.

## 5. Statistics

- Intervals are percentile bootstraps with 10,000 resamples and seed 11.
  Item claims resample items; cross-query claims resample queries. Single
  proportions use Wilson intervals (`wilson`).
- Where one script runs several bootstraps or simulations, each is seeded
  separately from a stable name (`seed_name`), so results do not depend on
  the order in which cells are visited.
- The main matrix of a deployment uses its first clean executions in repeat
  order: ten for gpt-oss-120B (M), five for every other deployment. Later
  executions of the same cells enter only the `_all` view in `items.csv`,
  which `more_executions.py` reads.

## 6. Module map

| File | Role |
|---|---|
| `scripts/sembench_repeats.py` | SemBench driver; also provides `_env` (gateway settings from `.env`) and `logged_spend` to other drivers |
| `scripts/lrobench_repeats.py` | LRO-Bench driver |
| `scripts/uda_repeats.py` | UDA-Bench extraction driver |
| `scripts/temperature_control.py` | Temperature control driver |
| `scripts/run_product_reviews.py` | Product-review driver |
| `scripts/make_movie_join_subset.py` | Builds the 40-review join input |
| `scripts/analyze_sembench_repeats.py` | Item and answer tables for SemBench; the measures above |
| `scripts/analyze_lrobench_repeats.py`, `analyze_uda_repeats.py` | The same for LRO-Bench and UDA-Bench |
| `scripts/score_*.py` | Scoring with each benchmark's own evaluator |
| `scripts/report_*.py`, `plot_*.py` | Tables and figures |
| `scripts/paper_macros.py` | All number macros |
| `scripts/compare_reference.py` | `make check`: compares regenerated tables and macros with `reference/` |
| `tests/` | Unit tests of the measures, parsers, and driver bookkeeping |
