"""Compute benchmark labels and inter-annotator agreement for both subsets.

Two annotation subsets share this pipeline:

* ``emolia-emo`` — 3-level ordinal rating per (file, queried_emotion, task_type)
  with ratings in {not_present, weakly_present, strongly_present}.
* ``emolia-dim`` — binary yes/no rating per (file, dimension, level, polarity)
  asking whether the audio matches the level rubric in
  ``dataset/emolia-dim/variables.json``.

For each subset we write to ``analysis_outputs/<subset>/``:

* ``benchmark_labels.csv`` — one row per item, with majority-vote targets
  ready for evaluation (``majority_present``, ``benchmark_bucket``, ...).
* ``per_*_summary.csv``    — slice tables.
* ``incomplete_items.csv`` — items lacking the configured 3-rater coverage.
* ``summary.json``         — machine-readable summary.
* ``report.md``            — human-readable summary suitable for paper drafts.

A combined paper-ready ``analysis_outputs/report.md`` is also written, with
sample counts, demographics, and inter-rater agreement for both subsets.

Run with: ``uv run analysis.py``
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ANNOTATIONS_ROOT = Path("annotations")
DATASET_ROOT = Path("dataset")
OUTPUT_ROOT = Path("analysis_outputs")

EMO_RATING_ORDER = ["not_present", "weakly_present", "strongly_present"]
EMO_RATING_TO_INT = {r: i for i, r in enumerate(EMO_RATING_ORDER)}
EMO_INT_TO_RATING = {i: r for r, i in EMO_RATING_TO_INT.items()}

DIM_RATING_ORDER = ["no", "yes"]
DIM_RATING_TO_INT = {r: i for i, r in enumerate(DIM_RATING_ORDER)}

EMO_ITEM_KEYS = ["file_name", "queried_emotion", "task_type"]
DIM_ITEM_KEYS = ["file_name", "dimension", "level", "polarity"]

# Sampling-strategy semantics (preselection bias before annotation):
#   emolia-emo: `affirmative` = preselected as likely-present (model expects yes);
#               the other four task types (`contrastive_1/2`, `ultimate`,
#               `penultimate`) are sampling strategies that preselect items
#               likely to be absent.
#   emolia-dim: `positive` polarity = preselected as likely-yes; `negative`
#               polarity = preselected as likely-no.
# Annotators are free to disagree with the preselection on any item — these
# columns just expose which way each item was sampled, so we can quantify how
# often raters confirm or overturn the prior.
#
# The preselection prior was produced by gemini-3-flash judging each clip
# before any human saw it. We treat that prior as a synthetic 4th annotator
# (`PRESELECTION_RATER_NAME`) when the user wants to see the human+model
# panel, so kappa / pairwise / LOO numbers can be recomputed with the prior
# included.
EMO_AFFIRMATIVE_TASK = "affirmative"
PRESELECTION_RATER_NAME = "gemini3flash_preselect"


def augmented_agreement(
    cleaned: pd.DataFrame,
    item_keys: list[str],
    *,
    subset: str,
    target_col: str,
    min_humans: int = 2,
) -> dict[str, Any]:
    """Recompute pairwise + LOO + Fleiss kappa with the preselection prior added.

    The synthetic preselection rater is concatenated onto ``cleaned`` and the
    same metrics are recomputed. By default we restrict to items with at least
    ``min_humans`` real human raters — combined with the Gemini preselection
    that gives a 3+ rater panel which is what we treat as the standard
    annotation configuration in the paper. Returns ``nan`` when the augmented
    panel still has too few comparisons.
    """
    human_counts = cleaned.groupby(item_keys).size().rename("n_humans").reset_index()
    eligible_items = human_counts.loc[human_counts["n_humans"] >= min_humans, item_keys]
    if len(eligible_items):
        cleaned_eligible = cleaned.merge(eligible_items, on=item_keys, how="inner")
    else:
        cleaned_eligible = cleaned.iloc[0:0]
    synth = synthesize_preselection_annotations(cleaned_eligible, item_keys, subset=subset)
    if not len(synth):
        return {
            "preselection_min_humans": min_humans,
            "preselection_n_eligible_items": 0,
            "preselection_pairwise_accuracy": float("nan"),
            "preselection_pairwise_balanced_accuracy": float("nan"),
            "preselection_loo_accuracy": float("nan"),
            "preselection_loo_balanced_accuracy": float("nan"),
            "preselection_loo_n_annotations": 0.0,
            "preselection_fleiss_kappa": float("nan"),
        }
    augmented = pd.concat(
        [cleaned_eligible, synth[cleaned_eligible.columns.intersection(synth.columns)]],
        ignore_index=True,
    )
    pairwise = pairwise_human_metrics(augmented, item_keys, target_col)
    loo = leave_one_out_human_metrics(augmented, item_keys, target_col)
    pivot = augmented.pivot_table(
        index=item_keys, columns="username", values=target_col, aggfunc="first"
    )
    counts = counts_for_fleiss(pivot, [0, 1])
    # Fleiss requires equal coverage; restrict to items where every column is filled.
    mask = pivot.notna().all(axis=1)
    fleiss = float(fleiss_kappa(counts_for_fleiss(pivot[mask], [0, 1]))) if mask.any() else float("nan")
    # Gemini-vs-human-consensus: treat the synthetic rater as a "model" and
    # score it against the majority vote of the real human raters per item.
    # Computed on items where at least `min_humans` real humans rated.
    real_pivot = cleaned_eligible.pivot_table(
        index=item_keys, columns="username", values=target_col, aggfunc="first"
    )
    truths: list[int] = []
    preds: list[int] = []
    synth_lookup = synth.set_index(item_keys)[target_col].to_dict()
    for idx, row in real_pivot.iterrows():
        votes = row.dropna().astype(int).to_numpy()
        if len(votes) < min_humans:
            continue
        yes = int((votes == 1).sum())
        no = int((votes == 0).sum())
        if yes == no:
            continue
        consensus = 1 if yes > no else 0
        if idx in synth_lookup:
            truths.append(consensus)
            preds.append(int(synth_lookup[idx]))
    if truths:
        truths_arr = np.asarray(truths, dtype=int)
        preds_arr = np.asarray(preds, dtype=int)
        gemini_acc = float((truths_arr == preds_arr).mean())
        gemini_bal = _balanced_accuracy(truths_arr, preds_arr)
        gemini_n = len(truths_arr)
    else:
        gemini_acc = float("nan")
        gemini_bal = float("nan")
        gemini_n = 0

    return {
        "preselection_min_humans": min_humans,
        "preselection_n_eligible_items": int(len(eligible_items)),
        "preselection_pairwise_accuracy": pairwise["pairwise_accuracy"],
        "preselection_pairwise_balanced_accuracy": pairwise["pairwise_balanced_accuracy"],
        "preselection_loo_accuracy": loo["loo_accuracy"],
        "preselection_loo_balanced_accuracy": loo["loo_balanced_accuracy"],
        "preselection_loo_n_annotations": loo["loo_n_annotations"],
        "preselection_fleiss_kappa": fleiss,
        "preselection_n_full_coverage_items": int(mask.sum()),
        "gemini_vs_human_consensus_accuracy": gemini_acc,
        "gemini_vs_human_consensus_balanced_accuracy": gemini_bal,
        "gemini_vs_human_consensus_n_items": float(gemini_n),
    }


def synthesize_preselection_annotations(
    cleaned: pd.DataFrame,
    item_keys: list[str],
    *,
    subset: str,
) -> pd.DataFrame:
    """Return one synthetic annotation per item encoding the preselection prior.

    The synthetic rater (``PRESELECTION_RATER_NAME``) votes ``yes`` / ``present``
    on items the sampling pipeline judged likely-positive and ``no`` /
    ``not_present`` on items judged likely-negative. We use the *cleaned*
    annotation table (deduplicated, item-level keys) as the source of truth
    for which (item, prior) tuples exist — this guarantees we only synthesize
    a vote for items that real annotators have at least seen.
    """
    if not len(cleaned):
        return cleaned.iloc[0:0].copy()
    items = cleaned[item_keys].drop_duplicates().reset_index(drop=True)
    if subset == "emolia-emo":
        intended_positive = items["task_type"] == EMO_AFFIRMATIVE_TASK
        rating = np.where(intended_positive, "strongly_present", "not_present")
        rating_int = np.where(intended_positive, 2, 0)
        present = intended_positive.astype(int)
    elif subset == "emolia-dim":
        intended_positive = items["polarity"] == "positive"
        rating = np.where(intended_positive, "yes", "no")
        rating_int = intended_positive.astype(int)
        present = intended_positive.astype(int)
    else:
        raise ValueError(f"Unknown subset {subset!r}")
    synth = items.copy()
    synth["username"] = PRESELECTION_RATER_NAME
    synth["rating"] = rating
    synth["rating_int"] = rating_int
    synth["present"] = present
    return synth


# ---------- agreement primitives ----------

def cohen_kappa(a: np.ndarray, b: np.ndarray, labels: list[int]) -> float:
    observed = float((a == b).mean())
    a_probs = np.array([(a == lbl).mean() for lbl in labels], dtype=float)
    b_probs = np.array([(b == lbl).mean() for lbl in labels], dtype=float)
    expected = float((a_probs * b_probs).sum())
    if np.isclose(expected, 1.0):
        return float("nan")
    return (observed - expected) / (1.0 - expected)


def fleiss_kappa(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    n_items, _ = counts.shape
    raters_per_item = counts.sum(axis=1)
    if not np.all(raters_per_item == raters_per_item[0]):
        raise ValueError("Fleiss' kappa requires the same number of raters per item.")
    raters = raters_per_item[0]
    if raters < 2:
        return float("nan")
    category_marginals = counts.sum(axis=0) / (n_items * raters)
    per_item_agreement = ((counts**2).sum(axis=1) - raters) / (raters * (raters - 1))
    mean_observed = float(per_item_agreement.mean())
    mean_expected = float((category_marginals**2).sum())
    if np.isclose(mean_expected, 1.0):
        return float("nan")
    return (mean_observed - mean_expected) / (1.0 - mean_expected)


def pairwise_kappa(pivot: pd.DataFrame, labels: list[int]) -> dict[str, float]:
    """Mean Cohen's kappa over rater pairs, evaluated on items both raters rated."""
    out: dict[str, float] = {}
    for ua, ub in combinations(pivot.columns, 2):
        sub = pivot[[ua, ub]].dropna()
        if sub.empty:
            out[f"{ua}__{ub}"] = float("nan")
            continue
        out[f"{ua}__{ub}"] = cohen_kappa(
            sub[ua].to_numpy(dtype=int),
            sub[ub].to_numpy(dtype=int),
            labels,
        )
    pair_values = [v for k, v in out.items() if k != "mean"]
    out["mean"] = float(np.nanmean(pair_values)) if pair_values else float("nan")
    return out


def counts_for_fleiss(pivot: pd.DataFrame, labels: list[int]) -> np.ndarray:
    """Per-item label counts, skipping NaN cells (so unrated raters don't count)."""
    rows = []
    for _, row in pivot.iterrows():
        values = row.dropna().to_numpy(dtype=int)
        rows.append([int((values == lbl).sum()) for lbl in labels])
    return np.asarray(rows, dtype=int)


def majority_vote(values: pd.Series) -> tuple[int, int, bool]:
    counts = values.value_counts().sort_values(ascending=False)
    top = int(counts.iloc[0])
    is_tie = len(counts) > 1 and int(counts.iloc[1]) == top
    winner = int(counts.index.min()) if is_tie else int(counts.index[0])
    return winner, top, is_tie


def pairwise_exact_agreement(pivot: pd.DataFrame) -> float:
    if pivot.shape[1] < 2:
        return float("nan")
    matches = []
    for ua, ub in combinations(pivot.columns, 2):
        matches.append(float((pivot[ua].to_numpy() == pivot[ub].to_numpy()).mean()))
    return float(np.mean(matches))


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean of sensitivity and specificity; nan if either class is missing."""
    pos = y_true == 1
    neg = y_true == 0
    if not pos.any() or not neg.any():
        return float("nan")
    sens = float((y_pred[pos] == 1).mean())
    spec = float((y_pred[neg] == 0).mean())
    return 0.5 * (sens + spec)


def pairwise_human_metrics(
    annotations: pd.DataFrame, item_keys: list[str], target_col: str
) -> dict[str, float]:
    """Mean accuracy + balanced accuracy across all rater pairs.

    For every (rater_a, rater_b) pair, take items both rated, treat ``a`` as
    truth and ``b`` as prediction, and compute exact accuracy + balanced
    accuracy. Symmetric over direction (both ``a→b`` and ``b→a`` are added),
    so the resulting accuracy is also the mean pairwise exact agreement.
    """
    pivot = annotations.pivot_table(
        index=item_keys, columns="username", values=target_col, aggfunc="first"
    )
    raters = list(pivot.columns)
    accs: list[float] = []
    bals: list[float] = []
    for ua, ub in combinations(raters, 2):
        sub = pivot[[ua, ub]].dropna()
        if sub.empty:
            continue
        a = sub[ua].to_numpy(dtype=int)
        b = sub[ub].to_numpy(dtype=int)
        accs.append(float((a == b).mean()))
        for truth, pred in ((a, b), (b, a)):
            bal = _balanced_accuracy(truth, pred)
            if not np.isnan(bal):
                bals.append(bal)
    return {
        "pairwise_accuracy": float(np.mean(accs)) if accs else float("nan"),
        "pairwise_balanced_accuracy": float(np.mean(bals)) if bals else float("nan"),
        "n_rater_pairs": float(len(accs)),
    }


def leave_one_out_human_metrics(
    annotations: pd.DataFrame, item_keys: list[str], target_col: str
) -> dict[str, float]:
    """Each rater scored against the majority vote of the others on shared items.

    For every annotation (rater R on item I) where item I has at least two
    *other* raters with a clear majority, compare R's vote to that majority.
    Returns overall accuracy and balanced accuracy.

    This is the cleanest "how good is a single human?" number to compare
    directly to a model's balanced accuracy.
    """
    if not len(annotations):
        return {
            "loo_accuracy": float("nan"),
            "loo_balanced_accuracy": float("nan"),
            "loo_n_annotations": 0.0,
        }
    truths: list[int] = []
    preds: list[int] = []
    grouped = annotations.groupby(item_keys, sort=False)[target_col]
    item_arrays = {key: g.to_numpy(dtype=int) for key, g in grouped}
    item_indices = {key: g.index.to_numpy() for key, g in grouped}
    target = annotations[target_col].to_numpy(dtype=int)
    item_keys_tuple = annotations[item_keys].apply(tuple, axis=1).to_numpy()
    for key, indices in item_indices.items():
        votes = item_arrays[key]
        if len(votes) < 3:
            continue
        for local_idx, global_idx in enumerate(indices):
            others = np.delete(votes, local_idx)
            if len(others) < 2:
                continue
            yes = int((others == 1).sum())
            no = int((others == 0).sum())
            if yes == no:
                continue
            truths.append(1 if yes > no else 0)
            preds.append(int(target[global_idx]))
    if not truths:
        return {
            "loo_accuracy": float("nan"),
            "loo_balanced_accuracy": float("nan"),
            "loo_n_annotations": 0.0,
        }
    truths_arr = np.asarray(truths, dtype=int)
    preds_arr = np.asarray(preds, dtype=int)
    return {
        "loo_accuracy": float((truths_arr == preds_arr).mean()),
        "loo_balanced_accuracy": _balanced_accuracy(truths_arr, preds_arr),
        "loo_n_annotations": float(len(truths_arr)),
    }


# ---------- generic per-subset loader ----------

def load_subset(subset: str, item_keys: list[str], required_raters: int) -> dict[str, pd.DataFrame]:
    """Load and dedupe a subset; expose strict (3-rater) and partial (>=1) views.

    ``complete`` keeps only items with exactly ``required_raters`` raters and is
    used for kappa / Fleiss agreement statistics that require equal coverage.
    ``partial`` retains every cleaned annotation and is used to produce a
    permissive benchmark labels file (with an ``n_raters`` column so callers
    can filter to whichever coverage tier they want).
    """
    base = ANNOTATIONS_ROOT / subset
    annotations = pd.read_csv(base / "annotations.csv", parse_dates=["created_at"])
    users = pd.read_csv(base / "users.csv")
    user_item_keys = item_keys + ["username"]
    raw = annotations.copy()
    cleaned = (
        annotations.sort_values("created_at")
        .drop_duplicates(user_item_keys, keep="last")
        .reset_index(drop=True)
    )
    complete = (
        cleaned.groupby(item_keys, group_keys=False)
        .filter(lambda g: len(g) == required_raters)
        .reset_index(drop=True)
    )
    return {
        "raw": raw,
        "cleaned": cleaned,
        "partial": cleaned,
        "complete": complete,
        "users": users,
    }


def append_n_raters(labels: pd.DataFrame, partial: pd.DataFrame, item_keys: list[str]) -> pd.DataFrame:
    counts = (
        partial.groupby(item_keys).size().reset_index(name="n_raters")
    )
    return labels.merge(counts, on=item_keys, how="left")


# ---------- emolia-emo (ordinal) ----------

def build_emo_labels(complete: pd.DataFrame) -> pd.DataFrame:
    grouped = complete.groupby(EMO_ITEM_KEYS, sort=True)
    labels = grouped.agg(
        source_emotion=("source_emotion", lambda v: "|".join(sorted(set(v)))),
        usernames=("username", lambda v: "|".join(sorted(v))),
        reaction_time_ms_mean=("reaction_time_ms", "mean"),
        reaction_time_ms_median=("reaction_time_ms", "median"),
        sample_time_ms_mean=("sample_time_ms", "mean"),
    ).reset_index()

    ordinal = grouped["rating_int"].apply(majority_vote).reset_index(name="ordinal_vote")
    binary = grouped["present"].apply(majority_vote).reset_index(name="binary_vote")
    ord_parts = pd.DataFrame(
        ordinal["ordinal_vote"].tolist(),
        columns=["majority_rating_int", "majority_rating_votes", "majority_rating_tie"],
    )
    bin_parts = pd.DataFrame(
        binary["binary_vote"].tolist(),
        columns=["majority_present", "majority_present_votes", "majority_present_tie"],
    )
    labels = pd.concat([labels, ordinal[EMO_ITEM_KEYS], ord_parts, bin_parts], axis=1)
    labels = labels.loc[:, ~labels.columns.duplicated()].copy()

    score_sums = grouped.agg(
        strongly_present_votes=("rating", lambda v: int((v == "strongly_present").sum())),
        weakly_present_votes=("rating", lambda v: int((v == "weakly_present").sum())),
        not_present_votes=("rating", lambda v: int((v == "not_present").sum())),
    ).reset_index()
    labels = labels.merge(score_sums, on=EMO_ITEM_KEYS, how="left")

    labels["majority_rating"] = labels["majority_rating_int"].map(EMO_INT_TO_RATING)
    rater_count = (
        labels[["strongly_present_votes", "weakly_present_votes", "not_present_votes"]]
        .sum(axis=1)
    )
    multi_rater = rater_count >= 2
    labels["all_agree_binary"] = multi_rater & (labels["majority_present_votes"] == rater_count)
    labels["all_agree_ordinal"] = multi_rater & (labels["majority_rating_votes"] == rater_count)
    labels["intended_present"] = (labels["task_type"] == EMO_AFFIRMATIVE_TASK).astype(int)
    labels["matches_intended_label"] = labels["majority_present"] == labels["intended_present"]
    labels["benchmark_bucket"] = np.select(
        [
            labels["majority_present"].eq(1) & labels["all_agree_binary"],
            labels["majority_present"].eq(1) & multi_rater,
            labels["majority_present"].eq(0) & labels["all_agree_binary"],
            labels["majority_present"].eq(0) & multi_rater,
            labels["majority_present"].eq(1),
            labels["majority_present"].eq(0),
        ],
        [
            "unanimous_present",
            "majority_present",
            "unanimous_absent",
            "majority_absent",
            "single_rater_present",
            "single_rater_absent",
        ],
        default="mixed",
    )
    return labels.sort_values(["task_type", "queried_emotion", "file_name"]).reset_index(drop=True)


def emo_agreement(complete: pd.DataFrame, partial: pd.DataFrame) -> dict[str, Any]:
    ordinal_pivot = complete.pivot_table(
        index=EMO_ITEM_KEYS, columns="username", values="rating_int", aggfunc="first"
    ).sort_index(axis=1)
    binary_pivot = complete.pivot_table(
        index=EMO_ITEM_KEYS, columns="username", values="present", aggfunc="first"
    ).sort_index(axis=1)
    pairwise = pairwise_human_metrics(partial, EMO_ITEM_KEYS, "present")
    loo = leave_one_out_human_metrics(partial, EMO_ITEM_KEYS, "present")
    return {
        "exact_3way_ordinal": float((ordinal_pivot.nunique(axis=1) == 1).mean()),
        "exact_3way_binary": float((binary_pivot.nunique(axis=1) == 1).mean()),
        "pairwise_exact_binary": pairwise_exact_agreement(binary_pivot),
        "pairwise_exact_ordinal": pairwise_exact_agreement(ordinal_pivot),
        "pairwise_ordinal_kappa": pairwise_kappa(ordinal_pivot, [0, 1, 2]),
        "pairwise_binary_kappa": pairwise_kappa(binary_pivot, [0, 1]),
        "fleiss_ordinal_kappa": float(fleiss_kappa(counts_for_fleiss(ordinal_pivot, [0, 1, 2]))),
        "fleiss_binary_kappa": float(fleiss_kappa(counts_for_fleiss(binary_pivot, [0, 1]))),
        "human_pairwise_accuracy": pairwise["pairwise_accuracy"],
        "human_pairwise_balanced_accuracy": pairwise["pairwise_balanced_accuracy"],
        "human_loo_accuracy": loo["loo_accuracy"],
        "human_loo_balanced_accuracy": loo["loo_balanced_accuracy"],
        "human_loo_n_annotations": loo["loo_n_annotations"],
    }


def emo_per_task(labels: pd.DataFrame) -> pd.DataFrame:
    return (
        labels.groupby("task_type")
        .agg(
            items=("file_name", "size"),
            intended_present=("intended_present", "first"),
            majority_present_rate=("majority_present", "mean"),
            preselection_confirm_rate=("matches_intended_label", "mean"),
            unanimous_binary_rate=("all_agree_binary", "mean"),
            unanimous_ordinal_rate=("all_agree_ordinal", "mean"),
        )
        .sort_values("items", ascending=False)
        .reset_index()
    )


def emo_per_emotion(labels: pd.DataFrame) -> pd.DataFrame:
    return (
        labels.groupby("queried_emotion")
        .agg(
            items=("file_name", "size"),
            majority_present_rate=("majority_present", "mean"),
            unanimous_binary_rate=("all_agree_binary", "mean"),
            unanimous_ordinal_rate=("all_agree_ordinal", "mean"),
        )
        .sort_values(["majority_present_rate", "items"], ascending=[True, False])
        .reset_index()
    )


# ---------- emolia-dim (binary yes/no) ----------

def build_dim_labels(complete: pd.DataFrame) -> pd.DataFrame:
    grouped = complete.groupby(DIM_ITEM_KEYS, sort=True)
    labels = grouped.agg(
        usernames=("username", lambda v: "|".join(sorted(v))),
        reaction_time_ms_mean=("reaction_time_ms", "mean"),
        reaction_time_ms_median=("reaction_time_ms", "median"),
        sample_time_ms_mean=("sample_time_ms", "mean"),
    ).reset_index()

    binary = grouped["rating_int"].apply(majority_vote).reset_index(name="binary_vote")
    bin_parts = pd.DataFrame(
        binary["binary_vote"].tolist(),
        columns=["majority_present", "majority_present_votes", "majority_present_tie"],
    )
    labels = pd.concat([labels, binary[DIM_ITEM_KEYS], bin_parts], axis=1)
    labels = labels.loc[:, ~labels.columns.duplicated()].copy()

    yes_votes = grouped.agg(yes_votes=("rating", lambda v: int((v == "yes").sum()))).reset_index()
    no_votes = grouped.agg(no_votes=("rating", lambda v: int((v == "no").sum()))).reset_index()
    labels = labels.merge(yes_votes, on=DIM_ITEM_KEYS, how="left").merge(
        no_votes, on=DIM_ITEM_KEYS, how="left"
    )

    rater_count = labels["yes_votes"] + labels["no_votes"]
    multi_rater = rater_count >= 2
    labels["all_agree_binary"] = multi_rater & (labels["majority_present_votes"] == rater_count)
    labels["matches_intended_polarity"] = (
        (labels["polarity"] == "positive") & labels["majority_present"].eq(1)
    ) | ((labels["polarity"] == "negative") & labels["majority_present"].eq(0))
    labels["benchmark_bucket"] = np.select(
        [
            labels["majority_present"].eq(1) & labels["all_agree_binary"],
            labels["majority_present"].eq(1) & multi_rater,
            labels["majority_present"].eq(0) & labels["all_agree_binary"],
            labels["majority_present"].eq(0) & multi_rater,
            labels["majority_present"].eq(1),
            labels["majority_present"].eq(0),
        ],
        [
            "unanimous_yes",
            "majority_yes",
            "unanimous_no",
            "majority_no",
            "single_rater_yes",
            "single_rater_no",
        ],
        default="mixed",
    )
    return labels.sort_values(["dimension", "level", "polarity", "file_name"]).reset_index(drop=True)


def dim_agreement(complete: pd.DataFrame, partial: pd.DataFrame) -> dict[str, Any]:
    binary_pivot = complete.pivot_table(
        index=DIM_ITEM_KEYS, columns="username", values="rating_int", aggfunc="first"
    ).sort_index(axis=1)
    pairwise = pairwise_human_metrics(partial, DIM_ITEM_KEYS, "rating_int")
    loo = leave_one_out_human_metrics(partial, DIM_ITEM_KEYS, "rating_int")
    return {
        "exact_3way_binary": float((binary_pivot.nunique(axis=1) == 1).mean()),
        "pairwise_exact_binary": pairwise_exact_agreement(binary_pivot),
        "pairwise_binary_kappa": pairwise_kappa(binary_pivot, [0, 1]),
        "fleiss_binary_kappa": float(fleiss_kappa(counts_for_fleiss(binary_pivot, [0, 1]))),
        "human_pairwise_accuracy": pairwise["pairwise_accuracy"],
        "human_pairwise_balanced_accuracy": pairwise["pairwise_balanced_accuracy"],
        "human_loo_accuracy": loo["loo_accuracy"],
        "human_loo_balanced_accuracy": loo["loo_balanced_accuracy"],
        "human_loo_n_annotations": loo["loo_n_annotations"],
    }


def dim_per_dimension(labels: pd.DataFrame) -> pd.DataFrame:
    return (
        labels.groupby("dimension")
        .agg(
            items=("file_name", "size"),
            majority_yes_rate=("majority_present", "mean"),
            unanimous_rate=("all_agree_binary", "mean"),
            polarity_match_rate=("matches_intended_polarity", "mean"),
        )
        .sort_values("items", ascending=False)
        .reset_index()
    )


def dim_per_polarity(labels: pd.DataFrame) -> pd.DataFrame:
    return (
        labels.groupby("polarity")
        .agg(
            items=("file_name", "size"),
            majority_yes_rate=("majority_present", "mean"),
            unanimous_rate=("all_agree_binary", "mean"),
        )
        .reset_index()
    )


# ---------- shared helpers ----------

def incomplete_items(cleaned: pd.DataFrame, item_keys: list[str], required: int) -> pd.DataFrame:
    return (
        cleaned.groupby(item_keys)
        .agg(
            ratings_observed=("username", "size"),
            usernames=("username", lambda v: "|".join(sorted(v))),
        )
        .reset_index()
        .query("ratings_observed != @required")
        .sort_values(["ratings_observed", *item_keys])
        .reset_index(drop=True)
    )


def fmt(x: Any, digits: int = 3) -> str:
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if np.isnan(xf):
        return "n/a"
    return f"{xf:.{digits}f}"


# ---------- per-subset orchestration ----------

def run_emo(out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_subset("emolia-emo", EMO_ITEM_KEYS, required_raters=3)
    raw, cleaned, complete, users = data["raw"], data["cleaned"], data["complete"], data["users"]
    cleaned["rating_int"] = cleaned["rating"].map(EMO_RATING_TO_INT)
    cleaned["present"] = (cleaned["rating_int"] > 0).astype(int)
    complete["rating_int"] = complete["rating"].map(EMO_RATING_TO_INT)
    complete["present"] = (complete["rating_int"] > 0).astype(int)

    agreement = emo_agreement(complete, cleaned) if len(complete) else {
        "exact_3way_ordinal": float("nan"),
        "exact_3way_binary": float("nan"),
        "pairwise_exact_binary": float("nan"),
        "pairwise_exact_ordinal": float("nan"),
        "pairwise_ordinal_kappa": {"mean": float("nan")},
        "pairwise_binary_kappa": {"mean": float("nan")},
        "fleiss_ordinal_kappa": float("nan"),
        "fleiss_binary_kappa": float("nan"),
        "human_pairwise_accuracy": float("nan"),
        "human_pairwise_balanced_accuracy": float("nan"),
        "human_loo_accuracy": float("nan"),
        "human_loo_balanced_accuracy": float("nan"),
        "human_loo_n_annotations": 0.0,
    }
    augmented = augmented_agreement(cleaned, EMO_ITEM_KEYS, subset="emolia-emo", target_col="present")
    labels = build_emo_labels(cleaned)
    labels = append_n_raters(labels, cleaned, EMO_ITEM_KEYS)
    per_task = emo_per_task(labels)
    per_emotion = emo_per_emotion(labels)
    incomplete = incomplete_items(cleaned, EMO_ITEM_KEYS, required=3)

    labels.to_csv(out_dir / "benchmark_labels.csv", index=False)
    per_task.to_csv(out_dir / "per_task_type_summary.csv", index=False)
    per_emotion.to_csv(out_dir / "per_emotion_summary.csv", index=False)
    incomplete.to_csv(out_dir / "incomplete_items.csv", index=False)

    coverage = labels["n_raters"].value_counts().sort_index().to_dict()
    summary = {
        "subset": "emolia-emo",
        "rating_scale": EMO_RATING_ORDER,
        "rows_raw": int(len(raw)),
        "rows_deduplicated": int(len(cleaned)),
        "items_total": int(len(labels)),
        "items_complete": int(complete[EMO_ITEM_KEYS].drop_duplicates().shape[0]),
        "items_incomplete": int(len(incomplete)),
        "rater_coverage": {int(k): int(v) for k, v in coverage.items()},
        "annotators": users["username"].tolist(),
        "agreement": agreement,
        "augmented_agreement": augmented,
        "majority_present_rate": float(labels["majority_present"].mean()),
        "preselection_confirm_rate": float(labels["matches_intended_label"].mean()),
        "task_types": per_task.to_dict(orient="records"),
        "easiest_emotions": per_emotion.tail(5).to_dict(orient="records"),
        "hardest_emotions": per_emotion.head(5).to_dict(orient="records"),
        # Aliases used by benchmark.py to print the score rubric.
        "human_upper_bound_binary": agreement["pairwise_exact_binary"],
        "human_accuracy_binary": agreement["human_pairwise_accuracy"],
        "human_balanced_accuracy_binary": agreement["human_pairwise_balanced_accuracy"],
        "human_loo_accuracy_binary": agreement["human_loo_accuracy"],
        "human_loo_balanced_accuracy_binary": agreement["human_loo_balanced_accuracy"],
        "human_loo_n_annotations": agreement["human_loo_n_annotations"],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return {
        "summary": summary,
        "labels": labels,
        "per_task": per_task,
        "per_emotion": per_emotion,
        "users": users,
        "incomplete": incomplete,
        "agreement": agreement,
        "augmented": augmented,
    }


def run_dim(out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_subset("emolia-dim", DIM_ITEM_KEYS, required_raters=3)
    raw, cleaned, complete, users = data["raw"], data["cleaned"], data["complete"], data["users"]
    cleaned["rating_int"] = cleaned["rating"].map(DIM_RATING_TO_INT)
    complete["rating_int"] = complete["rating"].map(DIM_RATING_TO_INT)

    agreement = dim_agreement(complete, cleaned) if len(cleaned) else {
        "exact_3way_binary": float("nan"),
        "pairwise_exact_binary": float("nan"),
        "pairwise_binary_kappa": {"mean": float("nan")},
        "fleiss_binary_kappa": float("nan"),
        "human_pairwise_accuracy": float("nan"),
        "human_pairwise_balanced_accuracy": float("nan"),
        "human_loo_accuracy": float("nan"),
        "human_loo_balanced_accuracy": float("nan"),
        "human_loo_n_annotations": 0.0,
    }
    augmented = augmented_agreement(cleaned, DIM_ITEM_KEYS, subset="emolia-dim", target_col="rating_int")
    labels = build_dim_labels(cleaned)
    labels = append_n_raters(labels, cleaned, DIM_ITEM_KEYS)
    per_dim = dim_per_dimension(labels)
    per_pol = dim_per_polarity(labels)
    incomplete = incomplete_items(cleaned, DIM_ITEM_KEYS, required=3)

    flags_path = ANNOTATIONS_ROOT / "emolia-dim" / "flags.csv"
    flags = pd.read_csv(flags_path) if flags_path.exists() else pd.DataFrame(
        columns=DIM_ITEM_KEYS + ["username", "reason", "created_at"]
    )
    if len(flags):
        flag_keys = flags[DIM_ITEM_KEYS].drop_duplicates().astype({"level": str})
        labels_for_merge = labels.copy()
        labels_for_merge["level"] = labels_for_merge["level"].astype(str)
        merged = labels_for_merge.merge(
            flag_keys.assign(flagged=True), on=DIM_ITEM_KEYS, how="left"
        )
        labels["flagged"] = merged["flagged"].fillna(False).astype(bool).to_numpy()
        flags.to_csv(out_dir / "flags.csv", index=False)
    else:
        labels["flagged"] = False

    labels.to_csv(out_dir / "benchmark_labels.csv", index=False)
    per_dim.to_csv(out_dir / "per_dimension_summary.csv", index=False)
    per_pol.to_csv(out_dir / "per_polarity_summary.csv", index=False)
    incomplete.to_csv(out_dir / "incomplete_items.csv", index=False)

    coverage = labels["n_raters"].value_counts().sort_index().to_dict()
    flag_records = flags.to_dict(orient="records") if len(flags) else []
    summary = {
        "subset": "emolia-dim",
        "rating_scale": DIM_RATING_ORDER,
        "rows_raw": int(len(raw)),
        "rows_deduplicated": int(len(cleaned)),
        "items_total": int(len(labels)),
        "items_complete": int(complete[DIM_ITEM_KEYS].drop_duplicates().shape[0]),
        "items_incomplete": int(len(incomplete)),
        "items_flagged": int(labels["flagged"].sum()),
        "rater_coverage": {int(k): int(v) for k, v in coverage.items()},
        "annotators": users["username"].tolist(),
        "agreement": agreement,
        "augmented_agreement": augmented,
        "majority_yes_rate": float(labels["majority_present"].mean()),
        "polarity_match_rate": float(labels["matches_intended_polarity"].mean()),
        "dimensions": per_dim.to_dict(orient="records"),
        "polarities": per_pol.to_dict(orient="records"),
        "flags": flag_records,
        # Aliases used by benchmark.py to print the score rubric.
        "human_upper_bound_binary": agreement["pairwise_exact_binary"],
        "human_accuracy_binary": agreement["human_pairwise_accuracy"],
        "human_balanced_accuracy_binary": agreement["human_pairwise_balanced_accuracy"],
        "human_loo_accuracy_binary": agreement["human_loo_accuracy"],
        "human_loo_balanced_accuracy_binary": agreement["human_loo_balanced_accuracy"],
        "human_loo_n_annotations": agreement["human_loo_n_annotations"],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return {
        "summary": summary,
        "labels": labels,
        "per_dim": per_dim,
        "per_pol": per_pol,
        "users": users,
        "incomplete": incomplete,
        "agreement": agreement,
        "augmented": augmented,
        "flags": flags,
    }


# ---------- markdown reporting ----------

def render_emo_section(res: dict[str, Any]) -> list[str]:
    s = res["summary"]
    a = res["agreement"]
    coverage = s.get("rater_coverage", {})
    coverage_str = ", ".join(f"{k}-rater: {v}" for k, v in sorted(coverage.items()))
    lines = [
        "## emolia-emo (ordinal: not_present / weakly_present / strongly_present)",
        "",
        f"- Annotators: **{len(s['annotators'])}** ({', '.join(s['annotators'])})",
        f"- Raw annotations: **{s['rows_raw']}** (deduplicated: {s['rows_deduplicated']})",
        f"- Items total: **{s['items_total']}** ({coverage_str})",
        f"- Complete 3-rater items: **{s['items_complete']}**",
        f"- Majority-present (binary) rate: **{fmt(s['majority_present_rate'])}**",
        f"- Preselection-confirmed rate (annotators agreed with sampling prior): "
        f"**{fmt(s.get('preselection_confirm_rate'))}**",
        "",
        "Sampling strategies. `task_type=affirmative` items were preselected as",
        "likely to *contain* the queried emotion; the other four task types",
        "(`contrastive_1`, `contrastive_2`, `penultimate`, `ultimate`) are four",
        "sampling strategies that preselect clips judged *unlikely* to contain it.",
        "Annotators rate independently, so the majority can confirm or overturn",
        "the prior — `matches_intended_label` in `benchmark_labels.csv` records",
        "whether they did.",
        "",
        "### Inter-rater agreement",
        "",
        "| Metric | Ordinal (3-level) | Binary (present vs absent) |",
        "|---|---|---|",
        f"| Exact 3-way agreement | {fmt(a['exact_3way_ordinal'])} | {fmt(a['exact_3way_binary'])} |",
        f"| Mean pairwise exact agreement | {fmt(a['pairwise_exact_ordinal'])} | {fmt(a['pairwise_exact_binary'])} |",
        f"| Mean pairwise Cohen's kappa | {fmt(a['pairwise_ordinal_kappa']['mean'])} | {fmt(a['pairwise_binary_kappa']['mean'])} |",
        f"| Fleiss' kappa | {fmt(a['fleiss_ordinal_kappa'])} | {fmt(a['fleiss_binary_kappa'])} |",
        "",
        "### Human upper bound (binary task)",
        "",
        "These are the numbers to compare a CLAP-style model's accuracy / "
        "balanced accuracy against.",
        "",
        "| Metric | Humans only | Humans + Gemini preselection |",
        "|---|---:|---:|",
        f"| Pairwise accuracy | {fmt(a['human_pairwise_accuracy'])} | "
        f"{fmt(res['augmented']['preselection_pairwise_accuracy'])} |",
        f"| Pairwise balanced accuracy | {fmt(a['human_pairwise_balanced_accuracy'])} | "
        f"{fmt(res['augmented']['preselection_pairwise_balanced_accuracy'])} |",
        f"| Leave-one-out accuracy | {fmt(a['human_loo_accuracy'])} | "
        f"{fmt(res['augmented']['preselection_loo_accuracy'])} |",
        f"| Leave-one-out balanced accuracy | {fmt(a['human_loo_balanced_accuracy'])} | "
        f"{fmt(res['augmented']['preselection_loo_balanced_accuracy'])} |",
        f"| LOO comparisons used | {int(a.get('human_loo_n_annotations') or 0)} | "
        f"{int(res['augmented'].get('preselection_loo_n_annotations') or 0)} |",
        f"| Fleiss κ (full-coverage items only) | {fmt(a.get('fleiss_binary_kappa'))} | "
        f"{fmt(res['augmented'].get('preselection_fleiss_kappa'))} |",
        "",
        "### Per-task summary",
        "",
        "`intended` = sampling prior (1 = preselected as likely-present, 0 = likely-absent).",
        "`confirm-rate` = fraction of items where the majority vote matched the prior.",
        "",
        "| task_type | items | intended | majority-present | confirm-rate | unanimous-binary | unanimous-ordinal |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in res["per_task"].itertuples(index=False):
        lines.append(
            f"| {row.task_type} | {row.items} | {int(row.intended_present)} | "
            f"{fmt(row.majority_present_rate)} | {fmt(row.preselection_confirm_rate)} | "
            f"{fmt(row.unanimous_binary_rate)} | {fmt(row.unanimous_ordinal_rate)} |"
        )
    lines.extend([
        "",
        "### Hardest 5 queried emotions (lowest majority-present rate)",
        "",
        "| queried_emotion | items | majority-present | unanimous-binary |",
        "|---|---:|---:|---:|",
    ])
    for row in res["per_emotion"].head(5).itertuples(index=False):
        lines.append(
            f"| {row.queried_emotion} | {row.items} | "
            f"{fmt(row.majority_present_rate)} | {fmt(row.unanimous_binary_rate)} |"
        )
    lines.extend([
        "",
        "### Easiest 5 queried emotions (highest majority-present rate)",
        "",
        "| queried_emotion | items | majority-present | unanimous-binary |",
        "|---|---:|---:|---:|",
    ])
    for row in res["per_emotion"].tail(5).sort_values("majority_present_rate", ascending=False).itertuples(index=False):
        lines.append(
            f"| {row.queried_emotion} | {row.items} | "
            f"{fmt(row.majority_present_rate)} | {fmt(row.unanimous_binary_rate)} |"
        )
    lines.append("")
    return lines


def render_dim_section(res: dict[str, Any]) -> list[str]:
    s = res["summary"]
    a = res["agreement"]
    coverage = s.get("rater_coverage", {})
    coverage_str = ", ".join(f"{k}-rater: {v}" for k, v in sorted(coverage.items()))
    lines = [
        "## emolia-dim (binary yes/no against rubric descriptions)",
        "",
        f"- Annotators: **{len(s['annotators'])}** ({', '.join(s['annotators'])})",
        f"- Raw annotations: **{s['rows_raw']}** (deduplicated: {s['rows_deduplicated']})",
        f"- Items total: **{s['items_total']}** ({coverage_str})",
        f"- Complete 3-rater items: **{s['items_complete']}**",
        f"- Items flagged for removal: **{s.get('items_flagged', 0)}** "
        f"(see `flags.csv`; flagged items are kept in `benchmark_labels.csv` with "
        "`flagged=true` so they can be filtered out at training time)",
        f"- Majority-yes rate: **{fmt(s['majority_yes_rate'])}**",
        f"- Preselection-confirmed rate (majority matched the polarity prior): "
        f"**{fmt(s['polarity_match_rate'])}**",
        "",
        "Sampling strategies. `polarity=positive` items were preselected as",
        "likely to *match* the rubric description for the given dimension/level",
        "(annotators expected to answer `yes`); `polarity=negative` items were",
        "preselected as likely to *not* match (expected `no`). Annotators",
        "answer independently — `matches_intended_polarity` in",
        "`benchmark_labels.csv` records whether the majority confirmed the prior.",
        "",
        "### Inter-rater agreement",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Exact 3-way agreement | {fmt(a['exact_3way_binary'])} |",
        f"| Mean pairwise exact agreement | {fmt(a['pairwise_exact_binary'])} |",
        f"| Mean pairwise Cohen's kappa | {fmt(a['pairwise_binary_kappa']['mean'])} |",
        f"| Fleiss' kappa | {fmt(a['fleiss_binary_kappa'])} |",
        "",
        "### Human upper bound (binary task)",
        "",
        "These are the numbers to compare a CLAP-style model's accuracy / "
        "balanced accuracy against.",
        "",
        "| Metric | Humans only | Humans + Gemini preselection |",
        "|---|---:|---:|",
        f"| Pairwise accuracy | {fmt(a['human_pairwise_accuracy'])} | "
        f"{fmt(res['augmented']['preselection_pairwise_accuracy'])} |",
        f"| Pairwise balanced accuracy | {fmt(a['human_pairwise_balanced_accuracy'])} | "
        f"{fmt(res['augmented']['preselection_pairwise_balanced_accuracy'])} |",
        f"| Leave-one-out accuracy | {fmt(a['human_loo_accuracy'])} | "
        f"{fmt(res['augmented']['preselection_loo_accuracy'])} |",
        f"| Leave-one-out balanced accuracy | {fmt(a['human_loo_balanced_accuracy'])} | "
        f"{fmt(res['augmented']['preselection_loo_balanced_accuracy'])} |",
        f"| LOO comparisons used | {int(a.get('human_loo_n_annotations') or 0)} | "
        f"{int(res['augmented'].get('preselection_loo_n_annotations') or 0)} |",
        f"| Fleiss κ (full-coverage items only) | {fmt(a.get('fleiss_binary_kappa'))} | "
        f"{fmt(res['augmented'].get('preselection_fleiss_kappa'))} |",
        "",
        "### Per-polarity summary",
        "",
        "| polarity | items | majority-yes | unanimous |",
        "|---|---:|---:|---:|",
    ]
    for row in res["per_pol"].itertuples(index=False):
        lines.append(
            f"| {row.polarity} | {row.items} | "
            f"{fmt(row.majority_yes_rate)} | {fmt(row.unanimous_rate)} |"
        )
    lines.extend([
        "",
        "### Per-dimension summary (sorted by item count)",
        "",
        "| dimension | items | majority-yes | unanimous | polarity-match |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in res["per_dim"].itertuples(index=False):
        lines.append(
            f"| {row.dimension} | {row.items} | "
            f"{fmt(row.majority_yes_rate)} | {fmt(row.unanimous_rate)} | "
            f"{fmt(row.polarity_match_rate)} |"
        )
    flags = res.get("flags")
    if flags is not None and len(flags):
        lines.extend([
            "",
            "### Annotator-flagged items",
            "",
            f"{len(flags)} items have been flagged by an annotator with a reason "
            "(typically silence / no speech). They remain in `benchmark_labels.csv` "
            "but are tagged `flagged=true` so the model trainer can exclude them.",
            "",
            "| flagger | dimension | level | polarity | file_name | reason |",
            "|---|---|---|---|---|---|",
        ])
        for row in flags.itertuples(index=False):
            reason = str(getattr(row, "reason", "")).replace("|", "/")
            lines.append(
                f"| {row.username} | {row.dimension} | {row.level} | "
                f"{row.polarity} | {row.file_name} | {reason} |"
            )
    lines.append("")
    return lines


def render_demographics(emo_users: pd.DataFrame, dim_users: pd.DataFrame) -> list[str]:
    union = pd.concat([emo_users.assign(_subset="emo"), dim_users.assign(_subset="dim")])
    n_unique = union["username"].nunique()
    lines = [
        "## Annotator demographics",
        "",
        f"- Distinct annotator ids across both subsets: **{n_unique}**",
        "",
        "| subset | id | age | gender | languages | listening setup | self-rated expertise |",
        "|---|---|---:|---|---|---|---:|",
    ]
    for subset, df in [("emolia-emo", emo_users), ("emolia-dim", dim_users)]:
        for row in df.sort_values("username").itertuples(index=False):
            languages = getattr(row, "languages", "")
            listening = getattr(row, "listening_setup", "")
            gender = getattr(row, "gender", "")
            age = getattr(row, "age", "")
            expertise = getattr(row, "expertise", "")
            lines.append(
                f"| {subset} | {row.username} | {age} | {gender} | {languages} | "
                f"{listening} | {expertise} |"
            )
    lines.append("")
    return lines


def _human_bound_rows(subset_name: str, agreement: dict[str, Any]) -> str:
    return (
        f"| {subset_name} | "
        f"{fmt(agreement['human_pairwise_accuracy'])} | "
        f"{fmt(agreement['human_pairwise_balanced_accuracy'])} | "
        f"{fmt(agreement['human_loo_accuracy'])} | "
        f"{fmt(agreement['human_loo_balanced_accuracy'])} | "
        f"{int(agreement.get('human_loo_n_annotations') or 0)} |"
    )


def _augmented_rows(subset_name: str, augmented: dict[str, Any]) -> str:
    return (
        f"| {subset_name} | "
        f"{fmt(augmented['preselection_pairwise_accuracy'])} | "
        f"{fmt(augmented['preselection_pairwise_balanced_accuracy'])} | "
        f"{fmt(augmented['preselection_loo_accuracy'])} | "
        f"{fmt(augmented['preselection_loo_balanced_accuracy'])} | "
        f"{int(augmented.get('preselection_loo_n_annotations') or 0)} | "
        f"{fmt(augmented.get('preselection_fleiss_kappa'))} |"
    )


def render_combined_report(emo: dict[str, Any], dim: dict[str, Any]) -> str:
    lines = [
        "# EmoLia annotation analysis",
        "",
        "Auto-generated by `analysis.py`. Numbers here are paper-ready: counts,",
        "agreement statistics, and per-slice summaries for both subsets.",
        "",
        "## Headline: human upper bound on the standard 3-rater panel",
        "",
        "Binary task: \"is the queried emotion / dimension present in this clip?\"",
        "Random baseline = 0.500.",
        "",
        "Our **standard panel is `2 humans + Gemini preselection`** (the",
        "`gemini-3-flash` prior counted as a 3rd rater). This is the realistic",
        "configuration on the bulk of EmoLia today: every clip in the dataset",
        "has a Gemini prior, and ≥2 humans rated 100% of emolia-emo and 75% of",
        "emolia-dim items. Numbers below are computed on items with at least",
        "2 human raters.",
        "",
        "- **Pairwise** = each rater pair scored against each other on items",
        "  they both rated.",
        "- **Leave-one-out (LOO)** = each annotation scored against the",
        "  majority of the *other* raters on that item. This is the cleanest",
        "  rater-vs-consensus number.",
        "",
        "| Subset | Eligible items (≥2 humans) | Pairwise accuracy | Pairwise balanced acc. | LOO accuracy | LOO balanced acc. | LOO n |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| emolia-emo | "
            f"{emo['augmented'].get('preselection_n_eligible_items', 0)} | "
            f"{fmt(emo['augmented']['preselection_pairwise_accuracy'])} | "
            f"{fmt(emo['augmented']['preselection_pairwise_balanced_accuracy'])} | "
            f"{fmt(emo['augmented']['preselection_loo_accuracy'])} | "
            f"{fmt(emo['augmented']['preselection_loo_balanced_accuracy'])} | "
            f"{int(emo['augmented'].get('preselection_loo_n_annotations') or 0)} |"
        ),
        (
            f"| emolia-dim | "
            f"{dim['augmented'].get('preselection_n_eligible_items', 0)} | "
            f"{fmt(dim['augmented']['preselection_pairwise_accuracy'])} | "
            f"{fmt(dim['augmented']['preselection_pairwise_balanced_accuracy'])} | "
            f"{fmt(dim['augmented']['preselection_loo_accuracy'])} | "
            f"{fmt(dim['augmented']['preselection_loo_balanced_accuracy'])} | "
            f"{int(dim['augmented'].get('preselection_loo_n_annotations') or 0)} |"
        ),
        "",
        "### How well does Gemini's preselection itself match the human consensus?",
        "",
        "Treating Gemini's prior as a model and scoring it against the majority",
        "vote of the available human raters (≥2 humans):",
        "",
        "| Subset | Items | Accuracy | Balanced accuracy |",
        "|---|---:|---:|---:|",
        (
            f"| emolia-emo | "
            f"{int(emo['augmented'].get('gemini_vs_human_consensus_n_items') or 0)} | "
            f"{fmt(emo['augmented'].get('gemini_vs_human_consensus_accuracy'))} | "
            f"{fmt(emo['augmented'].get('gemini_vs_human_consensus_balanced_accuracy'))} |"
        ),
        (
            f"| emolia-dim | "
            f"{int(dim['augmented'].get('gemini_vs_human_consensus_n_items') or 0)} | "
            f"{fmt(dim['augmented'].get('gemini_vs_human_consensus_accuracy'))} | "
            f"{fmt(dim['augmented'].get('gemini_vs_human_consensus_balanced_accuracy'))} |"
        ),
        "",
        "## Reference: humans-only ceiling (≥3 human raters, smaller N)",
        "",
        "Same metrics restricted to items where **3 humans** independently rated",
        "— i.e. without using Gemini at all. Useful as a sanity check, but the",
        "small N on emolia-dim makes the headline number above more reliable.",
        "",
        "| Subset | Pairwise accuracy | Pairwise balanced acc. | LOO accuracy | LOO balanced acc. | LOO n |",
        "|---|---:|---:|---:|---:|---:|",
        _human_bound_rows("emolia-emo", emo["agreement"]),
        _human_bound_rows("emolia-dim", dim["agreement"]),
        "",
        "## Quick stats",
        "",
        "| Subset | Annotators | Items (3-rater) | Annotations | Maj-positive rate | Pairwise κ (binary) | Fleiss κ (binary) |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| emolia-emo | {len(emo['summary']['annotators'])} | "
            f"{emo['summary']['items_complete']} | "
            f"{emo['summary']['rows_deduplicated']} | "
            f"{fmt(emo['summary']['majority_present_rate'])} | "
            f"{fmt(emo['agreement']['pairwise_binary_kappa']['mean'])} | "
            f"{fmt(emo['agreement']['fleiss_binary_kappa'])} |"
        ),
        (
            f"| emolia-dim | {len(dim['summary']['annotators'])} | "
            f"{dim['summary']['items_complete']} | "
            f"{dim['summary']['rows_deduplicated']} | "
            f"{fmt(dim['summary']['majority_yes_rate'])} | "
            f"{fmt(dim['agreement']['pairwise_binary_kappa']['mean'])} | "
            f"{fmt(dim['agreement']['fleiss_binary_kappa'])} |"
        ),
        "",
    ]
    lines.extend(render_emo_section(emo))
    lines.extend(render_dim_section(dim))
    lines.extend(render_demographics(emo["users"], dim["users"]))
    lines.extend([
        "## Interpreting the agreement numbers",
        "",
        "Cohen / Fleiss kappa reads (Landis & Koch, 1977):",
        "",
        "- κ < 0.00: poor",
        "- 0.00 – 0.20: slight",
        "- 0.21 – 0.40: fair",
        "- 0.41 – 0.60: moderate",
        "- 0.61 – 0.80: substantial",
        "- 0.81 – 1.00: almost perfect",
        "",
        "The mean pairwise *exact* agreement on the binary task is the natural",
        "human upper bound for benchmark accuracy — `benchmark.py` cites it in",
        "the score rubric.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--subsets", nargs="+", default=["emolia-emo", "emolia-dim"])
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    if "emolia-emo" in args.subsets:
        results["emo"] = run_emo(args.output_root / "emolia-emo")
    if "emolia-dim" in args.subsets:
        results["dim"] = run_dim(args.output_root / "emolia-dim")

    if {"emo", "dim"} <= set(results):
        report = render_combined_report(results["emo"], results["dim"])
        (args.output_root / "report.md").write_text(report)
        print(report)
        print(f"\nWrote outputs to {args.output_root.resolve()}")
    else:
        print(f"Wrote per-subset outputs to {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
