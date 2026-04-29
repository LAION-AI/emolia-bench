# Benchmark (CLAP vs human majority)

- Mode: **`remote`** (`http://127.0.0.1:18765/v1/similarity`)
- Labels: `/Users/robert/Developer/python_code/emolia-bench/analysis_outputs/benchmark_labels.csv`
- Rows in run: **100**
- Evaluated OK: **100**
- Missing `dataset/**/*.json`: **0**
- Request/score failures: **0**
- Threshold: **0.0** (predict present if similarity ≥ threshold)

## Overall vs `majority_present`
- Accuracy: **`0.5100`**
- Balanced accuracy: **`0.5500`**
- F1 (positive class): **`0.6475`**
- Precision: **`0.9184`** · Recall: **`0.5000`**
- Similarity distribution (OK rows): min=-0.9902, median=-0.0215, max=0.9949

## Interpretation
- **Sham** scores are deterministic from `(audio stem, prompt)` — they do **not** read audio.
- Point `--endpoint` at your CLAP HTTP service using the JSON contract in `benchmark.py`.

## Outputs
- `predictions.csv` — all rows (ok + errors); filter `ok==True` for metrics rows
- `metrics_by_task_type.csv`, `metrics_by_benchmark_bucket.csv` — slice metrics (ok rows)
- `summary.json` — machine-readable summary
