#!/usr/bin/env python3
"""
Evaluate a CLAP-like audio-text model against human benchmark labels.

- Loads `analysis_outputs/benchmark_labels.csv` (run `analysis.py` first unless you pass --labels).
- Resolves each row to `dataset/**/{stem}.json` (metadata; same stem as annotated `.mp3` names).
- Calls either a sham local scorer (--no endpoint) or a remote HTTP POST (--endpoint).

Remote endpoint contract (sham-compatible; customize your server):

  POST {endpoint URL}
  Content-Type: application/json

  Body:
    {
      "text": "<prompt about the queried emotion>",
      "audio_base64": "<base64-encoded raw bytes>",
      "audio_filename": "<original filename for logging>"
    }

  Response JSON must include ONE of:
    { "similarity": <float> }   # higher = stronger match / presence signal
    { "score": <float> }
    { "logit": <float> }       # interpreted like similarity when no sigmoid requested

Classification: predict present if similarity >= threshold (default 0.0).

Example (sham):

  ./.venv/bin/python benchmark.py --output-dir benchmark_outputs

Example (dry-run with fake scores only, limit 500):

  ./.venv/bin/python benchmark.py --limit 500

Example (stub server that matches sham scores):

  ./.venv/bin/python sham_clap_server.py --port 8765
  ./.venv/bin/python benchmark.py --endpoint http://127.0.0.1:8765/v1/similarity

Example (your real CLAP HTTP server):

  ./.venv/bin/python benchmark.py --endpoint https://gpu.example.invalid/v1/embed
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_LABELS = Path("analysis_outputs/benchmark_labels.csv")
DEFAULT_DATASET_ROOT = Path("dataset")
DEFAULT_OUTPUT_DIR = Path("benchmark_outputs")


def stem_from_annotation_filename(file_name: str) -> str:
    return Path(file_name).stem


def build_dataset_json_index(dataset_root: Path) -> dict[str, Path]:
    """Map basename stem → first matching JSON path (deterministic ordering)."""
    index: dict[str, Path] = {}
    paths = sorted(dataset_root.rglob("*.json"), key=lambda p: str(p))
    for p in paths:
        stem = p.stem
        if stem not in index:
            index[stem] = p
    return index


def resolve_dataset_json(file_name: str, index: dict[str, Path]) -> Path | None:
    stem = stem_from_annotation_filename(file_name)
    return index.get(stem)


def emotion_to_prompt(queried_emotion: str) -> str:
    label = queried_emotion.replace("___", " / ").replace("_", " ").strip()
    return f"Speech audio in which the speaker expresses or conveys {label}."


def sham_similarity(audio_stem: str, text: str, seed_byte: bytes = b"clap-sham-v1") -> float:
    """
    Deterministic fake score in [-1, 1] for reproducible runs without a server.
    Uses human labels only indirectly via differing text embeddings would in real CLAP —
    here it is unrelated to audio content (sham baseline).
    """
    h = hashlib.sha256(seed_byte + audio_stem.encode("utf-8") + b"|" + text.encode("utf-8")).digest()
    x = int.from_bytes(h[:8], "big") / (2**64)
    return float(2.0 * x - 1.0)


def post_similarity(endpoint: str, text: str, audio_path: Path, timeout_s: float) -> float:
    payload: dict[str, Any] = {
        "text": text,
        "audio_base64": base64.standard_b64encode(audio_path.read_bytes()).decode("ascii"),
        "audio_filename": audio_path.name,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"HTTP {e.code} from {endpoint}: {detail}") from e

    if "similarity" in body:
        return float(body["similarity"])
    if "score" in body:
        return float(body["score"])
    if "logit" in body:
        return float(body["logit"])
    raise KeyError(f"Response missing similarity/score/logit keys: {list(body.keys())}")


def classify_from_score(score: float, threshold: float) -> int:
    return 1 if score >= threshold else 0


def binary_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    assert len(y_true) == len(y_pred)
    n = len(y_true)
    if n == 0:
        return {
            "n": 0.0,
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
        }

    tp = fp = tn = fn = 0
    for yt, yp in zip(y_true, y_pred, strict=True):
        if yt == 1 and yp == 1:
            tp += 1
        elif yt == 0 and yp == 1:
            fp += 1
        elif yt == 0 and yp == 0:
            tn += 1
        elif yt == 1 and yp == 0:
            fn += 1

    acc = (tp + tn) / n
    sen = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spe = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    balanced = (sen + spe) / 2.0 if not (math.isnan(sen) or math.isnan(spe)) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec = sen
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else float("nan")

    return {
        "n": float(n),
        "accuracy": acc,
        "balanced_accuracy": balanced,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


def summarize_by_task(rows: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for task, g in rows.groupby("task_type", sort=False):
        m = binary_metrics(g["majority_present"].astype(int).tolist(), g["pred_present"].astype(int).tolist())
        parts.append({"task_type": task, **{k: m[k] for k in ["n", "accuracy", "balanced_accuracy", "f1"]}})
    return pd.DataFrame(parts).sort_values("task_type")


def summarize_by_bucket(rows: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for bucket, g in rows.groupby("benchmark_bucket", sort=False):
        m = binary_metrics(g["majority_present"].astype(int).tolist(), g["pred_present"].astype(int).tolist())
        parts.append({"benchmark_bucket": bucket, **{k: m[k] for k in ["n", "accuracy", "balanced_accuracy", "f1"]}})
    return pd.DataFrame(parts)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark CLAP-like model vs benchmark_labels.csv")
    p.add_argument("--labels", type=Path, default=DEFAULT_LABELS, help="Path to benchmark_labels.csv")
    p.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT, help="Root containing **/*.json shards")
    p.add_argument(
        "--endpoint",
        type=str,
        default=None,
        help="Full URL for POST (JSON audio_base64 + text). Omit to use deterministic sham scorer.",
    )
    p.add_argument("--threshold", type=float, default=0.0, help="similarity threshold for predicting present")
    p.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout seconds")
    p.add_argument("--limit", type=int, default=None, help="Max rows after filtering (sanity subset)")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for CSV/JSON/MD reports")
    p.add_argument(
        "--no-audio-send",
        action="store_true",
        help="With endpoint: only send stem + text in JSON (paths only; sham testing). Sham mode ignores.",
    )
    return p.parse_args()


def remote_score_minimal(endpoint: str, stem: str, prompt: str, timeout_s: float) -> float:
    """Optional path-only ping; server might read files by stem from disk."""
    payload = {"audio_stem": stem, "text": prompt}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    if "similarity" in body:
        return float(body["similarity"])
    if "score" in body:
        return float(body["score"])
    if "logit" in body:
        return float(body["logit"])
    raise KeyError(f"Response missing similarity/score/logit keys: {list(body.keys())}")


def relative_or_abs(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd()))
    except ValueError:
        return str(path.resolve())


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    labels_df = pd.read_csv(args.labels)
    if args.limit is not None:
        labels_df = labels_df.iloc[: args.limit].copy()

    dataset_index = build_dataset_json_index(args.dataset_root)

    records: list[dict[str, Any]] = []
    missing_files = 0
    errors = 0

    for row in labels_df.itertuples(index=False):
        fname = getattr(row, "file_name")
        queried = getattr(row, "queried_emotion")
        task_type = getattr(row, "task_type")
        benchmark_bucket = getattr(row, "benchmark_bucket")
        gt = int(getattr(row, "majority_present"))

        ds_path = resolve_dataset_json(fname, dataset_index)
        prompt = emotion_to_prompt(queried)

        if ds_path is None:
            missing_files += 1
            records.append({
                "file_name": fname,
                "dataset_path": "",
                "queried_emotion": queried,
                "task_type": task_type,
                "benchmark_bucket": benchmark_bucket,
                "majority_present": gt,
                "similarity": float("nan"),
                "pred_present": "",
                "ok": False,
                "error": "dataset_json_missing",
            })
            continue

        try:
            if args.endpoint is None:
                sim = sham_similarity(stem_from_annotation_filename(fname), prompt)
            elif args.no_audio_send:
                sim = remote_score_minimal(args.endpoint, stem_from_annotation_filename(fname), prompt, args.timeout)
            else:
                sim = post_similarity(args.endpoint, prompt, ds_path, args.timeout)

            pred = classify_from_score(sim, args.threshold)
            records.append({
                "file_name": fname,
                "dataset_path": relative_or_abs(ds_path),
                "queried_emotion": queried,
                "task_type": task_type,
                "benchmark_bucket": benchmark_bucket,
                "majority_present": gt,
                "similarity": sim,
                "pred_present": pred,
                "ok": True,
                "error": "",
            })
        except Exception as e:
            errors += 1
            records.append({
                "file_name": fname,
                "dataset_path": relative_or_abs(ds_path),
                "queried_emotion": queried,
                "task_type": task_type,
                "benchmark_bucket": benchmark_bucket,
                "majority_present": gt,
                "similarity": float("nan"),
                "pred_present": "",
                "ok": False,
                "error": repr(e),
            })

    out = pd.DataFrame.from_records(records)
    ok = out[out["ok"]].copy()
    if len(ok):
        ok["pred_present"] = ok["pred_present"].astype(int)
    preds = ok[["majority_present", "pred_present"]]

    metrics_overall = (
        binary_metrics(preds["majority_present"].tolist(), preds["pred_present"].tolist())
        if len(ok)
        else {"n": 0.0, "accuracy": float("nan"), "balanced_accuracy": float("nan"), "f1": float("nan")}
    )
    summary = {
        "mode": "remote" if args.endpoint else "sham_hash",
        "endpoint": args.endpoint,
        "threshold": args.threshold,
        "labels_path": str(args.labels.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "rows_total": int(len(labels_df)),
        "rows_evaluated_ok": int(len(ok)),
        "missing_dataset_json": int(missing_files),
        "evaluation_errors": int(errors),
        "overall_binary": metrics_overall,
    }

    out.to_csv(args.output_dir / "predictions.csv", index=False)

    if len(ok):
        by_task = summarize_by_task(ok)
        by_bucket = summarize_by_bucket(ok)
        summary["by_task_type"] = by_task.to_dict(orient="records")
        summary["by_benchmark_bucket"] = by_bucket.to_dict(orient="records")
        by_task.to_csv(args.output_dir / "metrics_by_task_type.csv", index=False)
        by_bucket.to_csv(args.output_dir / "metrics_by_benchmark_bucket.csv", index=False)
    else:
        summary["by_task_type"] = []
        summary["by_benchmark_bucket"] = []

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    scores = ok["similarity"].to_numpy(dtype=float) if len(ok) else np.array([])

    def fmt4(x: object) -> str:
        if x is None:
            return "n/a"
        try:
            xf = float(x)
        except (TypeError, ValueError):
            return "n/a"
        if math.isnan(xf):
            return "n/a"
        return f"{xf:.4f}"

    sim_line = "- Similarity distribution (OK rows): "
    if scores.size:
        sim_line += (
            f"min={float(scores.min()):.4f}, median={float(np.median(scores)):.4f}, max={float(scores.max()):.4f}"
        )
    else:
        sim_line += "no successful predictions"

    md_lines = [
        "# Benchmark (CLAP vs human majority)",
        "",
        f"- Mode: **`{summary['mode']}`**"
        + (f" (`{summary['endpoint']}`)" if summary["endpoint"] else ""),
        f"- Labels: `{summary['labels_path']}`",
        f"- Rows in run: **{summary['rows_total']}**",
        f"- Evaluated OK: **{summary['rows_evaluated_ok']}**",
        f"- Missing `dataset/**/*.json`: **{summary['missing_dataset_json']}**",
        f"- Request/score failures: **{summary['evaluation_errors']}**",
        f"- Threshold: **{summary['threshold']}** (predict present if similarity ≥ threshold)",
        "",
        "## Overall vs `majority_present`",
        f"- Accuracy: **`{fmt4(metrics_overall.get('accuracy'))}`**",
        f"- Balanced accuracy: **`{fmt4(metrics_overall.get('balanced_accuracy'))}`**",
        f"- F1 (positive class): **`{fmt4(metrics_overall.get('f1'))}`**",
        f"- Precision: **`{fmt4(metrics_overall.get('precision'))}`** · Recall: **`{fmt4(metrics_overall.get('recall'))}`**",
        sim_line,
        "",
        "## Interpretation",
        "- **Sham** scores are deterministic from `(audio stem, prompt)` — they do **not** read audio.",
        "- Point `--endpoint` at your CLAP HTTP service using the JSON contract in `benchmark.py`.",
        "",
        "## Outputs",
        "- `predictions.csv` — all rows (ok + errors); filter `ok==True` for metrics rows",
        "- `metrics_by_task_type.csv`, `metrics_by_benchmark_bucket.csv` — slice metrics (ok rows)",
        "- `summary.json` — machine-readable summary",
        "",
    ]

    md_text = "\n".join(md_lines)
    (args.output_dir / "report.md").write_text(md_text)
    print(md_text)


if __name__ == "__main__":
    main()
