# emolia-bench

Benchmark tooling for two emotion-audio annotation subsets and CLAP-style
model evaluation against them.

> ### 🎧 Interactive companion pages (for reviewers)
>
> **[▶ Open the "Giving a Coding Agent Ears" companion site](docs/agent-ears/index.html)** &nbsp;·&nbsp;
> [Demo 1 — Understanding (in-the-wild radar profiles)](docs/agent-ears/understanding.html) &nbsp;·&nbsp;
> [Demo 2 — Guiding generation (designed character voices)](docs/agent-ears/voices.html)
>
> These three self-contained HTML pages (a landing page plus two sub-pages, audio embedded inline —
> just open them in a browser) let you **hear** what this benchmark measures and what it enables.
>
> **A note on naming:** the perceptual-dimension subset called **`emolia-dim`** in this repository is
> the **VoiceNet** taxonomy (the 57 fine-grained voice-performance dimensions); **`emolia-emo`** is the
> 40-emotion set. The companion pages use the **VoiceNet** name throughout.
>
> **What the pages show — beyond contrastive retrieval:**
> * **Understanding.** Real *in-the-wild* clips from the EmoLia dataset are scored along the VoiceNet
>   dimensions and drawn as **radar / spider-web profiles**, so you can see (and hear) how the taxonomy
>   describes natural, uncontrolled speech — not just curated benchmark items.
> * **Guiding generation.** The same VoiceNet dimensions (as fast predictor heads) are used as a
>   **fitness signal to steer text-to-speech / voice-acting generation** via evolutionary search over
>   natural-language voice instructions — designing out-of-distribution voices (ASMR, dragons, goblins,
>   grieving voices …) **with no reference recordings and no copyrighted training voices**.
>
> In other words: the benchmark's dimensions are not only a representation-level yardstick — they are a
> reusable *perception* signal that can **drive** expressive TTS and copyright-friendly voice design.
> (The pages are anonymized for double-blind review.)

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

In-process CLAP mode (loads a SentenceTransformer audio-text model on GPU and
scores in-process — no HTTP round-trip):

```bash
uv run benchmark.py \
  --clap-model /path/to/snapshot_or_hf_repo \
  --clap-device cuda --clap-dtype bfloat16
```

Published VoiceCLAP HF models have a `--model` shortcut that picks the right
in-process backend (SentenceTransformer for `voiceclap-large`, AutoModel for
`voiceclap-small`):

```bash
uv run benchmark.py --model voiceclap-small
uv run benchmark.py --model voiceclap-large
```

In-process modes need `torch`, `sentence-transformers`, `transformers`, and
`librosa`, declared as the `clap` extra:

```bash
uv sync --extra clap
```

CLAP-mode tuning flags (only active with `--clap-model` / `--model`):

* `--clap-device cuda|cpu` (default `cuda`).
* `--clap-dtype bfloat16|float16|float32` (default `bfloat16`).
* `--clap-audio-batch-size N` (default 8).
* `--clap-text-batch-size N` (default 8).
* `--clap-max-seconds S` — truncate audio before encoding (default 30s).

Useful flags:

* `--subset emolia-emo|emolia-dim|both` (default `both`).
* `--limit N` — quick smoke test on first N rows.
* `--threshold 0.0` — predict positive if similarity ≥ threshold.
* `--no-audio-send` — with `--endpoint`, send only stem + text in the JSON
  payload (server reads files itself).
* Filter flags (default keeps every row in `benchmark_labels.csv`):
    * `--min-raters N` — drop items with fewer than N human raters.
    * `--exclude-flagged` — drop items annotators flagged as broken audio.
    * `--unanimous-only` — keep only items where every rater agreed.

### The three standard use modes

`benchmark_labels.csv` keeps every annotated item, including ones with only
one rater and ones an annotator flagged. The columns `n_raters`,
`benchmark_bucket`, and `flagged` describe each row so you can pick the
subset that matches your evaluation goal:

#### 1. All samples (default — including single-rater items)

The most permissive view. Every annotated item counts, the label
`majority_present` is the majority of whatever votes are available (a single
rater's vote when only one human rated the item).

```bash
uv run benchmark.py
# or, equivalently in pandas:
labels = pd.read_csv("analysis_outputs/emolia-dim/benchmark_labels.csv")
```

Use this when you want maximum data volume and don't mind that ~25% of
emolia-dim labels come from a single rater.

#### 2. Stronger-evidence cases (≥2 raters, no flagged audio)

The recommended evaluation subset for headline numbers in a paper. Every
included item has at least two humans agreeing or disagreeing on the same
clip, and broken-audio items are excluded.

```bash
uv run benchmark.py --min-raters 2 --exclude-flagged
# or, in pandas:
labels = pd.read_csv("analysis_outputs/emolia-dim/benchmark_labels.csv")
labels = labels[(labels["n_raters"] >= 2) & (~labels["flagged"])]
```

This matches the "2 humans + Gemini preselection = 3-rater panel" that the
human upper bound is computed on, so the model and the human ceiling are
comparable.

#### 3. Auditing the broken-audio items (flagged-only or flag-aware)

If you want to see how a model performs *only* on the items annotators
flagged, or audit them yourself:

```bash
# Just the flagged items, in pandas:
labels = pd.read_csv("analysis_outputs/emolia-dim/benchmark_labels.csv")
flagged = labels[labels["flagged"]]
# Or skim the raw flag reasons:
flags = pd.read_csv("annotations/emolia-dim/flags.csv")
```

The 37 flagged items in emolia-dim are 0.2% of the corpus and the headline
metrics move by < 0.001 with or without them, but it's the right thing to
inspect when training a model that should robustly skip clips with no
speech.

#### Bonus: unanimous-only (the cleanest possible labels)

Add `--unanimous-only` to either of the modes above to keep only items
where *every* rater on the panel agreed. Smallest sample but highest
label confidence.

```bash
uv run benchmark.py --min-raters 2 --exclude-flagged --unanimous-only
```

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
