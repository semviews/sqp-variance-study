# Data

This document describes the benchmarks and deployments, the layout of the
raw logs, and the schema of every log and processed file. Paths are relative
to the repository root.

## 1. Benchmarks and inputs

| Suite | What runs | Inputs | Where they come from |
|---|---|---|---|
| SemBench, published results | No execution: the five runs per configuration that SemBench publishes | SemBench's metrics folders | SemBench repository |
| SemBench Movie | Q1–Q4, Q8–Q10 (filters and maps over reviews), scale factor 2000 | `SemBench/files/movie/data/sf_2000/` | SemBench repository |
| SemBench Movie joins | Q5–Q7 (semantic self-joins), exact, on 40 reviews of the joined film | `experiments/data/movie-join40/` | Built by `scripts/make_movie_join_subset.py` from SemBench's inputs (seeded) |
| SemBench Medical | Q1, Q4, Q10 (text filters and an extraction) | `SemBench/files/medical/data/` | SemBench repository |
| LRO-Bench | Every scored query of the select, match, impute, cluster, order, and multi-operator suites, every LLM implementation | LRO-Bench's databases and metadata | LRO-Bench repository |
| UDA-Bench | The Player dataset: attribute extraction from documents, then SQL | Documents and ground truth from UDA-Bench's download links; query manifests in `experiments/data/uda/queries/` | UDA-Bench |
| Product reviews | Thirteen filter and map tasks | `experiments/data/product-reviews/product_reviews.csv` (100 synthetic reviews) | Generated with a knowledge graph (Wikidata) and LLMs (see [AI_USE.md](AI_USE.md)); shipped here |
| Temperature control | Movie Q1 and Q3 requests replayed one at a time | Recorded request keys of the main campaign | This repository |

`experiments/data/uda/queries/Player/<task>/<group>/<n>/sql.json` holds
one file per UDA-Bench Player query, prepared from UDA-Bench's query files
and attribute descriptions. Each file has the query's SQL and, per table,
the attributes the SQL reads with their types and descriptions. The drivers
extract exactly these attributes, and `uda_lineage.py` uses the same lists
to compute each query's exposure.

## 2. Deployments

| Directory name | Paper name | Service | Log-probabilities |
|---|---|---|---|
| `managed-gpt-oss-120b` | gpt-oss-120B (M) | managed inference service | no |
| `managed-llama-3-3-70b-instruct` | Llama-70B (M) | managed | no |
| `managed-mistral-small-3-1-24b-2503` | Mistral-24B (M) | managed | no |
| `vllm-llama-3-3-70b-instruct` | Llama-70B (V) | shared vLLM service | yes |
| `vllm-qwen2-5-72b-instruct` | Qwen-72B (V) | vLLM | yes |
| `vllm-granite-3-3-8b-instruct` | Granite-8B (V) | vLLM | yes |
| `Azure-gpt-4o` | GPT-4o (C) | commercial API | yes |
| `aws-claude-sonnet-5` | Claude Sonnet 5 (C) | commercial API | no |

Every execution uses temperature 0. gpt-oss-120B is a reasoning model and
returns its reasoning text.

## 3. Raw logs: `experiments/raw/`

### SemBench: `sembench-repeats/<scenario>/<deployment>/<setting>/repeat-<r>/`

`<scenario>` is `movie` or `medical`. `<setting>` encodes how the driver ran
(see `setting_tag` in `scripts/sembench_repeats.py`):

| Setting | Meaning |
|---|---|
| `w20` | 20 concurrent requests, SemBench's default; the main matrix |
| `w20_lp` | the same, requesting log-probabilities (deployments that return them) |
| `*_exact_join40` | Movie Q5–Q7 run exactly on the 40-review input |
| `w1_lp_probe`, `w8_lp_probe`, `w64_lp_probe` | cause probe: concurrency 1, 8, 64 |
| `w20_lp_probe` | cause probe: the reference setting, interleaved with the others |
| `w20_shuffle_lp_probe` | cause probe: input rows shuffled, seeded by the repeat number |
| `w20_r-{low,medium,high}_probe` | cause probe: reasoning effort of gpt-oss-120B |

Files per query `Q<q>`:

| File | Contents |
|---|---|
| `Q<q>.calls.jsonl` | One JSON object per model call, in the order LOTUS issued them |
| `Q<q>.csv` | The answer SemBench's runner returned |
| `Q<q>.metrics.json` | Execution metadata |
| `Q<q>.failures.jsonl` | One record per failed attempt (only if an attempt failed) |

Fields of a call record:

| Field | Meaning |
|---|---|
| `key` | First 16 hex digits of the SHA-256 of the request messages; equal requests have equal keys |
| `text` | The first 300 characters of the last user message |
| `t` | Start time of the batch the call belonged to (Unix seconds) |
| `output` | The reply text |
| `reasoning_chars`, `reasoning_text` | Length and text of the reasoning, if the deployment returns it |
| `finish_reason` | As returned |
| `prompt_tokens`, `completion_tokens` | As returned |
| `fingerprint` | `system_fingerprint`, if returned |
| `cost` | USD reported by the gateway, if any |
| `logprobs` | First four output tokens, each with its log-probability and the top 5 alternatives (`--logprobs`) |
| `rate_limit_retries` | How often a rate-limited request was resent before it was answered |
| `error` | Present only in failed calls (failed calls appear only in failure records) |

`Q<q>.metrics.json` holds `scenario`, `model`, `setting`, `policy`,
`repeat`, `query_id`, `status`, `execution_time`, `token_usage`,
`model_calls`, `call_errors`, `scale_factor`, `temperature`, `endpoint`,
`workers`, `reasoning`, `shuffle_seed`, `attempt`, `data_dir`, and `data`
(a SHA-256 prefix of every input CSV, which shows that all executions read
identical inputs). Older cells lack the last four fields. A failure record
has the same fields plus `error`, up to three `call_error_samples`, and
`spend`. URLs inside error text are replaced with `<url>`.

### LRO-Bench: `lrobench/<deployment>/<operator>_<implementation>/repeat-<r>/`

Implementations follow LRO-Bench: `llm_all` (all rows in one prompt),
`llm_one` (one row per prompt), `llm_semi`, and for ordering
`llm_one-simple` and `llm_one-heap`. Per query `<id>`:

| File | Contents |
|---|---|
| `<id>.json` | Result: settings, attempt, seconds, the values each metric returned, token counts, and per metric the prediction it received (`metrics`) |
| `<id>.calls.jsonl` | Every model call: `key`, `role` (`operator` or `judge`), `t`, `seconds`, `output` |
| `<id>.failed.json` | Written instead of `<id>.json` when all attempts failed: the error and every failed attempt |

The impute judge is fixed to `managed-gpt-oss-120b` (`judge_model`).

### UDA-Bench: `uda-repeats/Player/<deployment>/w20/repeat-<r>/`

| Path | Contents |
|---|---|
| `extract/<table>.csv` | The extracted table after coercion, one row per document |
| `extract/raw/<table>.csv` | The raw replies before coercion |
| `extract/<table>.calls.jsonl` | Every call, with `table`, `attribute`, `doc`, and `attempt` added to the SemBench call fields |
| `extract/<table>.metrics.json` | Calls, documents, retries, failed calls, time, tokens |
| `extract/parts/<table>/<attribute>.json` | Per-attribute results, which let an interrupted extraction resume |
| `queries/<task>/<group>/<n>/result.csv` | The query's result over the extracted tables |
| `queries/<task>/<group>/<n>/score.json`, `acc_result/` | UDA-Bench's lexical scoring of the result |

### Temperature control: `controls/temperature/`

`manifest-Q<q>.json` lists the items (request keys), whether each changed
in the main campaign, and the request settings. `Q<q>-<setting>.jsonl`
holds one record per response, for settings `t0`, `t0-seed`, and `t1`:
`setting`, `repeat`, `key`, `attempt`, `t`, `seconds`, the response `id`,
`model_returned`, `output`, `reasoning_text`, `finish_reason`, token counts,
and `fingerprint`. Provider response headers were recorded and are removed
from this release.

### Product reviews: `product-reviews/sf10/<task>/repeat-<r>/labels.json`

The task, operator, model, and the output for every review id (`labels`),
with call and token counts.

## 4. Processed files: `experiments/processed/`

The main tables:

| File | One row per | Key columns |
|---|---|---|
| `sembench_repeats/scores.csv` | execution and query | `quality` (SemBench's metric), `metric`, `model_calls`, `execution_time` |
| `sembench_repeats/items.csv` | deployment, setting, query, item | `values` (outputs in execution order), `flipped`, `flip_k5`, `pair_disagreement`, reasoning length, `first_margin` |
| `sembench_repeats/queries.csv` | deployment, setting, query | `item_flip_rate_k5`, `item_pair_disagreement`, `answer_reproduction`, `answer_reproduction_independent` (prediction), the quality range, and the three-level identity shares |
| `lrobench_repeats/queries.csv` | deployment, implementation, query | `failed`, `distinct_answers`, `answer_reproduction`, quality range |
| `uda_repeats/items.csv` | deployment, table, document, attribute | `flipped`, `distinct`, `values` |
| `uda_repeats/queries.csv` | deployment, query | `answer_reproduction`, `gold_empty`, F1 range |
| `published_repeats/cells.csv` | SemBench configuration, system, query | the published runs' quality `values`, `range`, `sd` |

In `sembench_repeats/`, a setting with the suffix `_all` holds every
recorded execution of a main-matrix cell. The setting without the suffix
holds the main matrix: the first ten clean executions of gpt-oss-120B (M)
and the first five of every other deployment.

The JSON files (`facts.json`, `boost.json`, `campaigns.json`, and others)
hold the values that tables and macros quote. Files with a `"macros"` key
map macro names directly to the strings that appear in the paper.

## 5. What was removed from the logs

To keep the review anonymous, this release renames the two hosted services
to `managed` and `vllm` in every path and file, removes provider response
headers, replaces URLs inside error text with `<url>`, and replaces home
directory paths in tracebacks with `/home/user`. No model output was edited.
