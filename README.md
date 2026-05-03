# emolia-bench

Benchmark tooling for two emotion-audio annotation subsets and CLAP-style
model evaluation against them.

## Subsets

* **emolia-emo** — 3-level ordinal rating (`not_present` / `weakly_present` /
  `strongly_present`) per `(file, queried_emotion, task_type)`. Used to score
  whether a clip *expresses* a queried emotion. Variables are self-explanatory
  emotion names.
* **emolia-dim** — binary `yes` / `no` rating per
  `(file, dimension, level, polarity)` against a written rubric in
  `dataset/emolia-dim/variables.json`. Used to score whether a clip matches a
  specific level of a perceptual dimension (e.g. `TEMP` level 5 = "fast").

## Repository layout

```
annotations_raw/          # gitignored; real usernames live here
  emolia-emo/{annotations.csv, users.csv}
  emolia-dim/{annotations.csv, users.csv}
annotations/              # committed; usernames anonymized to user_0, user_1, …
  emolia-emo/{annotations.csv, users.csv}
  emolia-dim/{annotations.csv, users.csv}
dataset/
  emolia-emo/data/<Emotion>_best/<stem>.{mp3,json}
  emolia-dim/data/<DIM>/<level>/<polarity>/sample_NN.{mp3,json}
  emolia-dim/variables.json   # rubric for prompts
analysis_outputs/<subset>/
  benchmark_labels.csv per_*_summary.csv summary.json incomplete_items.csv
analysis_outputs/report.md   # combined paper-ready summary
benchmark_outputs/<subset>/
  predictions.csv metrics_by_*.csv summary.json report.md
```

## Environment

```bash
uv venv --python 3.13       # only needed once
uv run anonymize.py         # then any other entry point
```

All scripts use `uv run`.

## Pipeline

### 1. Refresh anonymized annotations

`annotations_raw/` is gitignored because usernames are still in there. Run
the anonymizer any time the raw CSVs are replaced:

```bash
uv run anonymize.py
```

This rewrites `annotations/<subset>/annotations.csv` and
`annotations/<subset>/users.csv` with usernames replaced by `user_0`, `user_1`,
… (sorted username order; mapping saved to `annotations_raw/<subset>/_anon_map.csv`).

### 2. Build benchmark labels and the agreement summary

```bash
uv run analysis.py
```

For each subset this writes to `analysis_outputs/<subset>/`:

* `benchmark_labels.csv` — one row per item with majority-vote target
  (`majority_present`), per-rating vote counts, `n_raters`, `all_agree_binary`,
  and `benchmark_bucket` (`unanimous_*`, `majority_*`, `single_rater_*`).
* `per_*_summary.csv` — slice tables (task type / emotion / dimension / polarity).
* `incomplete_items.csv` — items lacking 3 raters.
* `summary.json` — machine-readable summary including
  `human_upper_bound_binary` (mean pairwise exact agreement) and rater-coverage
  histogram.

It also writes a single combined paper-ready report to
`analysis_outputs/report.md`. Numbers there are formatted for direct use in a
methods section: total annotations, annotators / demographics, kappa /
Fleiss kappa, per-task and per-emotion / per-dimension breakdowns.

### 3. Run the model benchmark

Sham mode (deterministic fake similarity, no audio is read):

```bash
uv run benchmark.py
```

Remote endpoint mode:

```bash
uv run benchmark.py --endpoint http://127.0.0.1:8765/v1/similarity
```

Useful flags:

* `--subset emolia-emo|emolia-dim|both` (default `both`).
* `--require-3-raters` — restrict to items with full 3-rater coverage.
* `--limit N` — quick smoke test on first N rows.
* `--threshold 0.0` — predict positive if similarity ≥ threshold.
* `--no-audio-send` — with `--endpoint`, send only stem + text in the JSON
  payload (server reads files itself).

For each subset, `benchmark.py` writes to `benchmark_outputs/<subset>/`:

* `predictions.csv`
* `metrics_by_<task_type|polarity>.csv`
* `metrics_by_benchmark_bucket.csv`
* `summary.json`
* `report.md`

The report **starts and ends** with a score rubric:

| Band | Balanced accuracy | Notes |
|---|---|---|
| Bad | < 0.55 | At or below random; model isn't learning |
| Weak | 0.55 – 0.65 | Some signal, far from human |
| Medium | 0.65 – 0.75 | Useful but lossy; decent training target |
| Good | 0.75 – 0.85 | Strong CLAP-style performance |
| Excellent | ≥ 0.85 | Approaches human inter-rater agreement |

Headline metric: **balanced accuracy on `majority_present`**. The unanimous
subset (`benchmark_bucket` starting with `unanimous_`) is the cleanest target
to optimize against.

### Endpoint contract

`benchmark.py` posts JSON to your endpoint:

```json
{
  "text": "Speech audio in which the speaker expresses or conveys Sadness.",
  "audio_filename": "EN_B00025_S06526_W000000.mp3",
  "audio_base64": "<base64 mp3 bytes>"
}
```

The server should return one of `{"similarity": …}`, `{"score": …}`, or
`{"logit": …}`. The local stub server `sham_clap_server.py` implements this
contract for testing.

```bash
uv run sham_clap_server.py --port 8765
uv run benchmark.py --endpoint http://127.0.0.1:8765/v1/similarity
```

## End-to-end refresh after new annotations

```bash
# Drop new raw CSVs into annotations_raw/<subset>/, then:
uv run anonymize.py
uv run analysis.py
uv run benchmark.py
```
