"""Evaluate a CLAP-like audio-text model against EmoLia human benchmark labels.

Two subsets are supported:

* ``emolia-emo`` — prompt is built from the queried emotion. The model receives
  the audio plus a short English sentence and returns a similarity score.
  Ground truth is the human ``majority_present`` (binary) label.

* ``emolia-dim`` — prompt is the rubric description for the queried
  ``(dimension, level)`` pair from ``dataset/emolia-dim/variables.json``.
  Ground truth is the human ``majority_present`` (yes/no) label, optionally
  filtered to ``polarity == positive`` items where the rubric *should* match.

Audio resolution:

* emo: stem matched against ``dataset/emolia-emo/data/**/*.json|*.mp3``.
* dim: ``dataset/emolia-dim/data/<dimension>/<level>/<polarity>/<file_name>``.

Run prerequisites::

    uv run anonymize.py
    uv run analysis.py
    uv run benchmark.py                           # both subsets, sham scorer
    uv run benchmark.py --subset emolia-emo       # one subset
    uv run benchmark.py --endpoint http://...     # remote CLAP server

A score rubric is printed at the top *and* bottom of the run output so the
person training the model immediately knows what counts as bad / medium / good.
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

ANALYSIS_ROOT = Path("analysis_outputs")
DATASET_ROOT = Path("dataset")
DEFAULT_OUTPUT_DIR = Path("benchmark_outputs")
SUBSETS = ("emolia-emo", "emolia-dim")


# ---------- prompt construction ----------

def emotion_to_prompt(queried_emotion: str) -> str:
    label = queried_emotion.replace("___", " / ").replace("_", " ").strip()
    return f"Speech audio in which the speaker expresses or conveys {label}."


_VARIABLES_CACHE: dict[Path, dict[str, Any]] = {}


def load_dim_variables(path: Path) -> dict[str, Any]:
    if path not in _VARIABLES_CACHE:
        _VARIABLES_CACHE[path] = json.loads(path.read_text())
    return _VARIABLES_CACHE[path]


def dim_to_prompt(dimension: str, level: str | int, variables: dict[str, Any]) -> str:
    spec = variables.get(dimension)
    if not spec:
        return f"Speech audio matching the description of {dimension} at level {level}."
    levels = spec.get("levels", {})
    description = levels.get(str(level))
    if not description:
        short = spec.get("short_description", dimension)
        return f"Speech audio expressing the dimension '{short}' at level {level}."
    return description


# ---------- audio path resolution ----------

def emo_audio_index(dataset_root: Path) -> dict[str, Path]:
    """Map ``stem`` -> first ``.mp3`` (else ``.json``) under emolia-emo."""
    index: dict[str, Path] = {}
    base = dataset_root / "emolia-emo"
    for ext in ("*.mp3", "*.json"):
        for p in sorted(base.rglob(ext), key=lambda x: str(x)):
            index.setdefault(p.stem, p)
    return index


def dim_audio_path(
    dataset_root: Path, dimension: str, level: str | int, polarity: str, file_name: str
) -> Path:
    return dataset_root / "emolia-dim" / "data" / str(dimension) / str(level) / str(polarity) / file_name


# ---------- scoring backends ----------

def sham_similarity(audio_stem: str, text: str) -> float:
    """Deterministic [-1, 1] sham score; ignores audio content."""
    h = hashlib.sha256(b"clap-sham-v1" + audio_stem.encode("utf-8") + b"|" + text.encode("utf-8")).digest()
    return float(2.0 * (int.from_bytes(h[:8], "big") / (2**64)) - 1.0)


def post_similarity(
    endpoint: str, text: str, audio_path: Path, timeout_s: float, send_audio: bool
) -> float:
    payload: dict[str, Any] = {"text": text, "audio_filename": audio_path.name}
    if send_audio and audio_path.suffix.lower() == ".mp3" and audio_path.exists():
        payload["audio_base64"] = base64.standard_b64encode(audio_path.read_bytes()).decode("ascii")
    else:
        payload["audio_stem"] = audio_path.stem
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

    for key in ("similarity", "score", "logit"):
        if key in body:
            return float(body[key])
    raise KeyError(f"Response missing similarity/score/logit keys: {list(body.keys())}")


# ---------- metrics ----------

def binary_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    n = len(y_true)
    if n == 0:
        return {k: float("nan") for k in
                ("n", "accuracy", "balanced_accuracy", "precision", "recall", "f1")}
    tp = fp = tn = fn = 0
    for yt, yp in zip(y_true, y_pred, strict=True):
        if yt and yp:
            tp += 1
        elif yp and not yt:
            fp += 1
        elif not yp and not yt:
            tn += 1
        else:
            fn += 1
    acc = (tp + tn) / n
    sen = tp / (tp + fn) if (tp + fn) else float("nan")
    spe = tn / (tn + fp) if (tn + fp) else float("nan")
    bal = (sen + spe) / 2.0 if not (math.isnan(sen) or math.isnan(spe)) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = 2 * prec * sen / (prec + sen) if not math.isnan(prec) and not math.isnan(sen) and (prec + sen) else float("nan")
    return {
        "n": float(n),
        "accuracy": acc,
        "balanced_accuracy": bal,
        "precision": prec,
        "recall": sen,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


def fmt(x: Any, digits: int = 4) -> str:
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if math.isnan(xf):
        return "n/a"
    return f"{xf:.{digits}f}"


# ---------- score rubric ----------

def score_rubric(majority_rate: float, human_numbers: dict[str, Any]) -> list[str]:
    """Return a list of markdown lines describing the score rubric.

    The rubric prints both **accuracy** and **balanced accuracy** for humans on
    the same binary task (computed in ``analysis.py``), so a model's score has
    a directly comparable human ceiling — this is what reviewers and coauthors
    will ask for first.
    """
    majority_rate = float(majority_rate)
    majority_baseline = max(majority_rate, 1.0 - majority_rate)
    pairwise_acc = human_numbers.get("human_accuracy_binary")
    pairwise_bal = human_numbers.get("human_balanced_accuracy_binary")
    loo_acc = human_numbers.get("human_loo_accuracy_binary")
    loo_bal = human_numbers.get("human_loo_balanced_accuracy_binary")
    loo_n = int(human_numbers.get("human_loo_n_annotations") or 0)
    return [
        "## Score rubric (read me first)",
        "",
        "Primary metric: **balanced accuracy** vs `majority_present` (binary).",
        "Secondary metric: **F1** of the positive class.",
        "",
        f"- Random / chance baseline: **0.500** balanced accuracy",
        f"- Always-majority baseline: **{majority_baseline:.3f}** accuracy "
        f"(0.500 balanced)",
        "",
        "**Human upper bound on the exact same task** (computed in",
        "`analysis_outputs/<subset>/summary.json`):",
        "",
        "| Human metric | Accuracy | Balanced accuracy |",
        "|---|---:|---:|",
        f"| Pairwise (rater A vs rater B on shared items) | "
        f"{fmt(pairwise_acc, 3)} | {fmt(pairwise_bal, 3)} |",
        f"| Leave-one-out (rater vs majority of others, n={loo_n}) | "
        f"{fmt(loo_acc, 3)} | {fmt(loo_bal, 3)} |",
        "",
        "These are the numbers to compare your model's score against. The LOO",
        "balanced accuracy is the most apples-to-apples — it answers \"if a",
        "single human took this benchmark, what would they score against the",
        "consensus?\"",
        "",
        "Quality bands (balanced accuracy on `majority_present`):",
        "",
        "| Band | Balanced accuracy | Notes |",
        "|---|---|---|",
        "| Bad | < 0.55 | At or below random; model isn't learning |",
        "| Weak | 0.55 – 0.65 | Some signal, around single-human level |",
        "| Medium | 0.65 – 0.75 | Above single-human; useful training target |",
        "| Good | 0.75 – 0.85 | Strong CLAP-style performance |",
        "| Excellent | ≥ 0.85 | Approaches the consensus of multiple humans |",
        "",
        "If you train a model, **maximize balanced accuracy on the unanimous",
        "subset** (`benchmark_bucket` in {`unanimous_*`}) — those items are the",
        "cleanest part of the benchmark. The headline number we care about is",
        "balanced accuracy on all 3-rater items.",
        "",
    ]


# ---------- per-subset evaluation ----------

def emo_iter(labels: pd.DataFrame, audio_index: dict[str, Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in labels.itertuples(index=False):
        stem = Path(row.file_name).stem
        path = audio_index.get(stem)
        rows.append({
            "file_name": row.file_name,
            "queried_emotion": row.queried_emotion,
            "task_type": row.task_type,
            "benchmark_bucket": row.benchmark_bucket,
            "n_raters": int(getattr(row, "n_raters", 1)),
            "majority_present": int(row.majority_present),
            "audio_path": path,
            "stem": stem,
            "prompt": emotion_to_prompt(row.queried_emotion),
        })
    return rows


def dim_iter(
    labels: pd.DataFrame, dataset_root: Path, variables: dict[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in labels.itertuples(index=False):
        path = dim_audio_path(dataset_root, row.dimension, row.level, row.polarity, row.file_name)
        rows.append({
            "file_name": row.file_name,
            "dimension": row.dimension,
            "level": row.level,
            "polarity": row.polarity,
            "benchmark_bucket": row.benchmark_bucket,
            "n_raters": int(getattr(row, "n_raters", 1)),
            "majority_present": int(row.majority_present),
            "audio_path": path,
            "stem": Path(row.file_name).stem + f"__{row.dimension}_{row.level}_{row.polarity}",
            "prompt": dim_to_prompt(row.dimension, row.level, variables),
        })
    return rows


def evaluate_rows(
    rows: list[dict[str, Any]],
    *,
    endpoint: str | None,
    threshold: float,
    timeout: float,
    send_audio: bool,
) -> tuple[pd.DataFrame, int, int]:
    records: list[dict[str, Any]] = []
    missing = errors = 0
    for r in rows:
        path: Path | None = r["audio_path"]
        record = {k: v for k, v in r.items() if k not in ("audio_path", "stem", "prompt")}
        record["audio_path"] = "" if path is None else str(path)
        record["prompt"] = r["prompt"]
        if path is None or not path.exists():
            missing += 1
            record.update({"similarity": float("nan"), "pred_present": "", "ok": False,
                           "error": "audio_missing"})
            records.append(record)
            continue
        try:
            if endpoint is None:
                sim = sham_similarity(r["stem"], r["prompt"])
            else:
                sim = post_similarity(endpoint, r["prompt"], path, timeout, send_audio)
            pred = 1 if sim >= threshold else 0
            record.update({"similarity": sim, "pred_present": pred, "ok": True, "error": ""})
        except Exception as exc:
            errors += 1
            record.update({"similarity": float("nan"), "pred_present": "", "ok": False,
                           "error": repr(exc)})
        records.append(record)
    return pd.DataFrame.from_records(records), missing, errors


def metrics_by(rows: pd.DataFrame, key: str) -> pd.DataFrame:
    parts = []
    for value, g in rows.groupby(key, sort=True):
        m = binary_metrics(
            g["majority_present"].astype(int).tolist(),
            g["pred_present"].astype(int).tolist(),
        )
        parts.append({key: value, **{k: m[k] for k in ("n", "accuracy", "balanced_accuracy", "f1")}})
    return pd.DataFrame(parts)


# ---------- per-subset reporting ----------

def quality_band(balanced_accuracy: float) -> str:
    if math.isnan(balanced_accuracy):
        return "n/a"
    if balanced_accuracy < 0.55:
        return "Bad"
    if balanced_accuracy < 0.65:
        return "Weak"
    if balanced_accuracy < 0.75:
        return "Medium"
    if balanced_accuracy < 0.85:
        return "Good"
    return "Excellent"


def write_subset_report(
    subset: str,
    out_dir: Path,
    predictions: pd.DataFrame,
    summary: dict[str, Any],
    rubric_lines: list[str],
) -> str:
    ok = predictions[predictions["ok"]].copy()
    if len(ok):
        ok["pred_present"] = ok["pred_present"].astype(int)

    overall = (
        binary_metrics(ok["majority_present"].tolist(), ok["pred_present"].tolist())
        if len(ok)
        else binary_metrics([], [])
    )
    by_bucket = metrics_by(ok, "benchmark_bucket") if len(ok) else pd.DataFrame()
    if subset == "emolia-emo":
        by_slice = metrics_by(ok, "task_type") if len(ok) else pd.DataFrame()
        slice_name = "task_type"
    else:
        by_slice = metrics_by(ok, "polarity") if len(ok) else pd.DataFrame()
        slice_name = "polarity"

    if len(ok):
        unanimous_mask = ok["benchmark_bucket"].astype(str).str.startswith("unanimous_")
        unanimous_ok = ok[unanimous_mask]
        unanimous_metrics = (
            binary_metrics(
                unanimous_ok["majority_present"].tolist(),
                unanimous_ok["pred_present"].tolist(),
            )
            if len(unanimous_ok)
            else binary_metrics([], [])
        )
    else:
        unanimous_metrics = binary_metrics([], [])

    summary["overall"] = overall
    summary["unanimous_subset"] = unanimous_metrics
    summary[f"by_{slice_name}"] = by_slice.to_dict(orient="records") if len(by_slice) else []
    summary["by_benchmark_bucket"] = by_bucket.to_dict(orient="records") if len(by_bucket) else []

    predictions.to_csv(out_dir / "predictions.csv", index=False)
    if len(by_slice):
        by_slice.to_csv(out_dir / f"metrics_by_{slice_name}.csv", index=False)
    if len(by_bucket):
        by_bucket.to_csv(out_dir / "metrics_by_benchmark_bucket.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    sims = ok["similarity"].to_numpy(dtype=float) if len(ok) else np.array([])
    sim_line = (
        f"min={float(sims.min()):.4f}, median={float(np.median(sims)):.4f}, max={float(sims.max()):.4f}"
        if sims.size
        else "no successful predictions"
    )

    md = [
        f"# Benchmark report — `{subset}`",
        "",
        *rubric_lines,
        f"- Mode: **{summary['mode']}**"
        + (f" (`{summary.get('endpoint')}`)" if summary.get("endpoint") else ""),
        f"- Items considered: **{summary['rows_total']}** "
        f"(evaluated OK: **{summary['rows_evaluated_ok']}**, missing audio: "
        f"**{summary['missing_audio']}**, errors: **{summary['evaluation_errors']}**)",
        f"- Threshold: **{summary['threshold']}** (predict positive if similarity ≥ threshold)",
        f"- Similarity distribution (OK rows): {sim_line}",
        "",
        "## Headline metrics (all 3-rater items)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Items evaluated | {int(overall['n']) if not math.isnan(overall['n']) else 0} |",
        f"| Accuracy | {fmt(overall['accuracy'])} |",
        f"| **Balanced accuracy** | **{fmt(overall['balanced_accuracy'])}** |",
        f"| F1 (positive class) | {fmt(overall['f1'])} |",
        f"| Precision / Recall | {fmt(overall['precision'])} / {fmt(overall['recall'])} |",
        f"| Quality band | **{quality_band(overall['balanced_accuracy'])}** |",
        "",
        "## Unanimous-only subset (cleanest items)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Items evaluated | {int(unanimous_metrics['n']) if not math.isnan(unanimous_metrics['n']) else 0} |",
        f"| Balanced accuracy | {fmt(unanimous_metrics['balanced_accuracy'])} |",
        f"| F1 | {fmt(unanimous_metrics['f1'])} |",
        f"| Quality band | **{quality_band(unanimous_metrics['balanced_accuracy'])}** |",
        "",
    ]
    if len(by_slice):
        md += [f"## By {slice_name}", "", f"| {slice_name} | n | accuracy | bal. acc | F1 |",
               "|---|---:|---:|---:|---:|"]
        for row in by_slice.itertuples(index=False):
            md.append(
                f"| {getattr(row, slice_name)} | {int(row.n)} | "
                f"{fmt(row.accuracy)} | {fmt(row.balanced_accuracy)} | {fmt(row.f1)} |"
            )
        md.append("")
    if len(by_bucket):
        md += ["## By `benchmark_bucket`", "",
               "| bucket | n | accuracy | bal. acc | F1 |", "|---|---:|---:|---:|---:|"]
        for row in by_bucket.itertuples(index=False):
            md.append(
                f"| {row.benchmark_bucket} | {int(row.n)} | "
                f"{fmt(row.accuracy)} | {fmt(row.balanced_accuracy)} | {fmt(row.f1)} |"
            )
        md.append("")
    md += [
        "## Outputs",
        "",
        "- `predictions.csv` — all rows; filter `ok==True` for metric rows",
        "- `metrics_by_*.csv` — slice metrics",
        "- `summary.json` — machine-readable summary",
        "",
        "## Score rubric (recap — share these numbers with reviewers)",
        "",
        f"Model balanced accuracy = **{fmt(overall['balanced_accuracy'])}** "
        f"-> **{quality_band(overall['balanced_accuracy'])}** band.",
        "",
        "Comparable numbers on the *same binary task*:",
        "",
        f"- Random baseline: **0.500**",
        f"- Always-majority baseline: **{summary['majority_baseline']:.3f}** "
        "accuracy (0.500 balanced)",
        f"- Single-human pairwise: accuracy **{fmt(summary.get('human_accuracy_binary'), 3)}**, "
        f"balanced **{fmt(summary.get('human_balanced_accuracy_binary'), 3)}**",
        f"- Single-human leave-one-out (vs. majority of others, n="
        f"{int(summary.get('human_loo_n_annotations') or 0)}): "
        f"accuracy **{fmt(summary.get('human_loo_accuracy_binary'), 3)}**, "
        f"balanced **{fmt(summary.get('human_loo_balanced_accuracy_binary'), 3)}**",
        "",
    ]
    text = "\n".join(md)
    (out_dir / "report.md").write_text(text)
    return text


# ---------- subset orchestration ----------

def run_subset(
    subset: str,
    *,
    output_root: Path,
    dataset_root: Path,
    analysis_root: Path,
    endpoint: str | None,
    threshold: float,
    timeout: float,
    limit: int | None,
    require_3_raters: bool,
    send_audio: bool,
) -> str:
    out_dir = output_root / subset
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_path = analysis_root / subset / "benchmark_labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Missing {labels_path}. Run `uv run analysis.py` first."
        )
    labels = pd.read_csv(labels_path)
    if "n_raters" not in labels.columns:
        labels["n_raters"] = 3
    if require_3_raters:
        labels = labels[labels["n_raters"] >= 3].reset_index(drop=True)
    if limit is not None:
        labels = labels.iloc[:limit].copy()

    # Pull human upper bound + majority rate from analysis summary.json
    analysis_summary_path = analysis_root / subset / "summary.json"
    analysis_summary: dict[str, Any] = (
        json.loads(analysis_summary_path.read_text()) if analysis_summary_path.exists() else {}
    )
    majority_rate = float(labels["majority_present"].mean()) if len(labels) else 0.5
    majority_baseline = max(majority_rate, 1.0 - majority_rate)
    human_upper = analysis_summary.get("human_upper_bound_binary")
    human_numbers = {
        "human_accuracy_binary": analysis_summary.get("human_accuracy_binary"),
        "human_balanced_accuracy_binary": analysis_summary.get("human_balanced_accuracy_binary"),
        "human_loo_accuracy_binary": analysis_summary.get("human_loo_accuracy_binary"),
        "human_loo_balanced_accuracy_binary": analysis_summary.get("human_loo_balanced_accuracy_binary"),
        "human_loo_n_annotations": analysis_summary.get("human_loo_n_annotations"),
    }

    if subset == "emolia-emo":
        rows = emo_iter(labels, emo_audio_index(dataset_root))
    else:
        variables = load_dim_variables(dataset_root / "emolia-dim" / "variables.json")
        rows = dim_iter(labels, dataset_root, variables)

    predictions, missing, errors = evaluate_rows(
        rows,
        endpoint=endpoint,
        threshold=threshold,
        timeout=timeout,
        send_audio=send_audio,
    )

    rubric_lines = score_rubric(majority_rate, human_numbers)
    summary: dict[str, Any] = {
        "subset": subset,
        "mode": "remote" if endpoint else "sham_hash",
        "endpoint": endpoint,
        "threshold": threshold,
        "rows_total": int(len(labels)),
        "rows_evaluated_ok": int(predictions["ok"].sum()) if len(predictions) else 0,
        "missing_audio": int(missing),
        "evaluation_errors": int(errors),
        "majority_baseline": majority_baseline,
        "human_upper_bound_binary": human_upper,
        **human_numbers,
        "labels_path": str(labels_path.resolve()),
    }
    return write_subset_report(subset, out_dir, predictions, summary, rubric_lines)


# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subset", choices=("emolia-emo", "emolia-dim", "both"), default="both")
    p.add_argument("--analysis-root", type=Path, default=ANALYSIS_ROOT)
    p.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--endpoint", type=str, default=None,
                   help="Full URL for POST. Omit for the deterministic sham scorer.")
    p.add_argument("--threshold", type=float, default=0.0,
                   help="Predict positive if similarity >= threshold.")
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--limit", type=int, default=None,
                   help="Max rows after filtering, for quick smoke tests.")
    p.add_argument("--no-audio-send", action="store_true",
                   help="With --endpoint: send only stem + text in JSON.")
    p.add_argument("--require-3-raters", action="store_true",
                   help="Restrict labels to items with all 3 raters present.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subsets = SUBSETS if args.subset == "both" else (args.subset,)
    for subset in subsets:
        text = run_subset(
            subset,
            output_root=args.output_dir,
            dataset_root=args.dataset_root,
            analysis_root=args.analysis_root,
            endpoint=args.endpoint,
            threshold=args.threshold,
            timeout=args.timeout,
            limit=args.limit,
            require_3_raters=args.require_3_raters,
            send_audio=not args.no_audio_send,
        )
        print(text)
        print()


if __name__ == "__main__":
    main()
