# Benchmark (CLAP vs human majority)

- Mode: **`remote`** (`http://127.0.0.1:8765/v1/similarity`)
- Labels: `/Users/robert/Developer/python_code/emolia-bench/analysis_outputs/benchmark_labels.csv`
- Rows in run: **7984**
- Evaluated OK: **7984**
- Missing `dataset/**/*.json`: **0**
- Request/score failures: **0**
- Threshold: **0.0** (predict present if similarity ≥ threshold)

## Overall vs `majority_present`
- Accuracy: **`0.5014`**
- Balanced accuracy: **`0.5004`**
- F1 (positive class): **`0.5402`**
- Precision: **`0.5784`** · Recall: **`0.5068`**
- Similarity distribution (OK rows): min=-0.9992, median=0.0150, max=0.9995

## Interpretation
- **Sham** scores are deterministic from `(audio stem, prompt)` — they do **not** read audio.
- Point `--endpoint` at your CLAP HTTP service using the JSON contract in `benchmark.py`.

## Outputs
- `predictions.csv` — all rows (ok + errors); filter `ok==True` for metrics rows
- `metrics_by_task_type.csv`, `metrics_by_benchmark_bucket.csv` — slice metrics (ok rows)
- `summary.json` — machine-readable summary
