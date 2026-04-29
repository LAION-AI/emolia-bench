from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


ANNOTATIONS_PATH = Path("annotations/annotations.csv")
USERS_PATH = Path("annotations/users.csv")
OUTPUT_DIR = Path("analysis_outputs")

RATING_ORDER = ["not_present", "weakly_present", "strongly_present"]
RATING_TO_INT = {rating: idx for idx, rating in enumerate(RATING_ORDER)}
INT_TO_RATING = {idx: rating for rating, idx in RATING_TO_INT.items()}

ITEM_KEYS = ["file_name", "queried_emotion", "task_type"]
USER_ITEM_KEYS = ITEM_KEYS + ["username"]


def cohen_kappa(a: np.ndarray, b: np.ndarray, labels: list[int]) -> float:
    observed = float((a == b).mean())
    a_probs = np.array([(a == label).mean() for label in labels], dtype=float)
    b_probs = np.array([(b == label).mean() for label in labels], dtype=float)
    expected = float((a_probs * b_probs).sum())
    if np.isclose(expected, 1.0):
        return float("nan")
    return (observed - expected) / (1.0 - expected)


def fleiss_kappa(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    n_items, n_categories = counts.shape
    del n_categories  # documented by the formula but unused afterwards
    raters_per_item = counts.sum(axis=1)
    if not np.all(raters_per_item == raters_per_item[0]):
        raise ValueError("Fleiss' kappa requires the same number of raters per item.")
    raters = raters_per_item[0]
    category_marginals = counts.sum(axis=0) / (n_items * raters)
    per_item_agreement = ((counts**2).sum(axis=1) - raters) / (raters * (raters - 1))
    mean_observed = float(per_item_agreement.mean())
    mean_expected = float((category_marginals**2).sum())
    return (mean_observed - mean_expected) / (1.0 - mean_expected)


def summarize_pairwise_kappa(pivot: pd.DataFrame, labels: list[int]) -> dict[str, float]:
    results: dict[str, float] = {}
    for user_a, user_b in combinations(pivot.columns, 2):
        results[f"{user_a}__{user_b}"] = cohen_kappa(
            pivot[user_a].to_numpy(dtype=int),
            pivot[user_b].to_numpy(dtype=int),
            labels,
        )
    results["mean"] = float(np.nanmean(list(results.values())))
    return results


def counts_for_fleiss(pivot: pd.DataFrame, labels: list[int]) -> np.ndarray:
    rows = []
    for _, row in pivot.iterrows():
        values = row.to_numpy(dtype=int)
        rows.append([(values == label).sum() for label in labels])
    return np.asarray(rows, dtype=int)


def majority_vote(values: pd.Series) -> tuple[int, int, bool]:
    counts = values.value_counts().sort_values(ascending=False)
    top_count = int(counts.iloc[0])
    is_tie = len(counts) > 1 and int(counts.iloc[1]) == top_count
    winner = int(counts.index.min()) if is_tie else int(counts.index[0])
    return winner, top_count, is_tie


def load_annotations() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    annotations = pd.read_csv(ANNOTATIONS_PATH, parse_dates=["created_at"])
    raw = annotations.copy()
    cleaned = (
        annotations.sort_values("created_at")
        .drop_duplicates(USER_ITEM_KEYS, keep="last")
        .reset_index(drop=True)
    )
    complete = (
        cleaned.groupby(ITEM_KEYS, group_keys=False)
        .filter(lambda group: len(group) == 3)
        .reset_index(drop=True)
    )
    return raw, cleaned, complete


def build_item_labels(complete: pd.DataFrame) -> pd.DataFrame:
    grouped = complete.groupby(ITEM_KEYS, sort=True)
    labels = grouped.agg(
        source_emotion=("source_emotion", lambda values: "|".join(sorted(set(values)))),
        usernames=("username", lambda values: "|".join(sorted(values))),
        reaction_time_ms_mean=("reaction_time_ms", "mean"),
        reaction_time_ms_median=("reaction_time_ms", "median"),
        sample_time_ms_mean=("sample_time_ms", "mean"),
    ).reset_index()

    ordinal_stats = grouped["rating_int"].apply(majority_vote).reset_index(name="ordinal_vote")
    binary_stats = grouped["present"].apply(majority_vote).reset_index(name="binary_vote")

    ordinal_parts = pd.DataFrame(ordinal_stats["ordinal_vote"].tolist(), columns=[
        "majority_rating_int",
        "majority_rating_votes",
        "majority_rating_tie",
    ])
    binary_parts = pd.DataFrame(binary_stats["binary_vote"].tolist(), columns=[
        "majority_present",
        "majority_present_votes",
        "majority_present_tie",
    ])

    labels = pd.concat(
        [
            labels,
            ordinal_stats[ITEM_KEYS],
            ordinal_parts,
            binary_parts,
        ],
        axis=1,
    )
    labels = labels.loc[:, ~labels.columns.duplicated()].copy()

    score_sums = grouped.agg(
        strongly_present_votes=("rating", lambda values: int((values == "strongly_present").sum())),
        weakly_present_votes=("rating", lambda values: int((values == "weakly_present").sum())),
        not_present_votes=("rating", lambda values: int((values == "not_present").sum())),
    ).reset_index()
    labels = labels.merge(score_sums, on=ITEM_KEYS, how="left")

    labels["majority_rating"] = labels["majority_rating_int"].map(INT_TO_RATING)
    labels["present_vote_share"] = labels["majority_present_votes"] / 3.0
    labels["rating_vote_share"] = labels["majority_rating_votes"] / 3.0
    labels["all_agree_binary"] = labels["majority_present_votes"] == 3
    labels["all_agree_ordinal"] = labels["majority_rating_votes"] == 3
    labels["disagreement_level"] = np.select(
        [
            labels["all_agree_ordinal"],
            labels["all_agree_binary"],
            labels["majority_present_votes"] == 2,
        ],
        [
            "none",
            "polarity_only",
            "moderate",
        ],
        default="high",
    )
    labels["benchmark_bucket"] = np.select(
        [
            labels["majority_present"].eq(1) & labels["all_agree_binary"],
            labels["majority_present"].eq(1),
            labels["majority_present"].eq(0) & labels["all_agree_binary"],
            labels["majority_present"].eq(0),
        ],
        [
            "unanimous_present",
            "majority_present",
            "unanimous_absent",
            "majority_absent",
        ],
        default="mixed",
    )
    return labels.sort_values(["task_type", "queried_emotion", "file_name"]).reset_index(drop=True)


def agreement_summary(complete: pd.DataFrame) -> dict[str, object]:
    ordinal_pivot = complete.pivot_table(
        index=ITEM_KEYS,
        columns="username",
        values="rating_int",
        aggfunc="first",
    ).sort_index(axis=1)
    binary_pivot = complete.pivot_table(
        index=ITEM_KEYS,
        columns="username",
        values="present",
        aggfunc="first",
    ).sort_index(axis=1)

    ordinal_counts = counts_for_fleiss(ordinal_pivot, labels=[0, 1, 2])
    binary_counts = counts_for_fleiss(binary_pivot, labels=[0, 1])

    return {
        "exact_3way_ordinal": float((ordinal_pivot.nunique(axis=1) == 1).mean()),
        "exact_3way_binary": float((binary_pivot.nunique(axis=1) == 1).mean()),
        "pairwise_ordinal_kappa": summarize_pairwise_kappa(ordinal_pivot, labels=[0, 1, 2]),
        "pairwise_binary_kappa": summarize_pairwise_kappa(binary_pivot, labels=[0, 1]),
        "fleiss_ordinal_kappa": float(fleiss_kappa(ordinal_counts)),
        "fleiss_binary_kappa": float(fleiss_kappa(binary_counts)),
    }


def per_task_summary(labels: pd.DataFrame) -> pd.DataFrame:
    summary = (
        labels.groupby("task_type")
        .agg(
            items=("file_name", "size"),
            majority_present_rate=("majority_present", "mean"),
            unanimous_binary_rate=("all_agree_binary", "mean"),
            unanimous_ordinal_rate=("all_agree_ordinal", "mean"),
            unanimous_present_rate=("benchmark_bucket", lambda values: float((values == "unanimous_present").mean())),
            unanimous_absent_rate=("benchmark_bucket", lambda values: float((values == "unanimous_absent").mean())),
        )
        .sort_values("items", ascending=False)
        .reset_index()
    )
    return summary


def per_emotion_summary(labels: pd.DataFrame) -> pd.DataFrame:
    summary = (
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
    return summary


def incomplete_item_summary(cleaned: pd.DataFrame) -> pd.DataFrame:
    return (
        cleaned.groupby(ITEM_KEYS)
        .agg(
            ratings_observed=("username", "size"),
            usernames=("username", lambda values: "|".join(sorted(values))),
            source_emotions=("source_emotion", lambda values: "|".join(sorted(set(values)))),
        )
        .reset_index()
        .query("ratings_observed != 3")
        .sort_values(["ratings_observed", "task_type", "queried_emotion", "file_name"])
        .reset_index(drop=True)
    )


def build_summary_json(
    raw: pd.DataFrame,
    cleaned: pd.DataFrame,
    complete: pd.DataFrame,
    users: pd.DataFrame,
    agreement: dict[str, object],
    per_task: pd.DataFrame,
    per_emotion: pd.DataFrame,
    incomplete: pd.DataFrame,
) -> dict[str, object]:
    top_present = per_emotion.tail(10).sort_values("majority_present_rate", ascending=False)
    low_present = per_emotion.head(10)
    return {
        "dataset": {
            "raw_annotation_rows": int(len(raw)),
            "deduplicated_rows": int(len(cleaned)),
            "complete_rows_used_for_agreement": int(len(complete)),
            "duplicate_rows_removed": int(len(raw) - len(cleaned)),
            "incomplete_items_excluded": int(len(incomplete)),
            "complete_items_used_for_agreement": int(complete[ITEM_KEYS].drop_duplicates().shape[0]),
            "annotators": users["username"].tolist(),
            "annotator_counts": users.set_index("username")["annotation_count"].astype(int).to_dict(),
        },
        "agreement": agreement,
        "task_type_summary": per_task.to_dict(orient="records"),
        "bottom_emotions_by_majority_present_rate": low_present.to_dict(orient="records"),
        "top_emotions_by_majority_present_rate": top_present.to_dict(orient="records"),
    }


def render_report(
    summary: dict[str, object],
    per_task: pd.DataFrame,
    per_emotion: pd.DataFrame,
    incomplete: pd.DataFrame,
) -> str:
    agreement = summary["agreement"]
    dataset = summary["dataset"]
    hardest = per_emotion.head(5)
    easiest = per_emotion.tail(5).sort_values("majority_present_rate", ascending=False)

    lines = [
        "# Annotation Analysis",
        "",
        "## Overview",
        f"- Raw annotations: {dataset['raw_annotation_rows']}",
        f"- Deduplicated annotations: {dataset['deduplicated_rows']}",
        f"- Duplicate user-item rows removed: {dataset['duplicate_rows_removed']}",
        f"- Complete 3-rater items used for agreement: {dataset['complete_items_used_for_agreement']}",
        f"- Incomplete items excluded from agreement: {dataset['incomplete_items_excluded']}",
        "",
        "## Agreement",
        f"- Exact 3-way ordinal agreement: {agreement['exact_3way_ordinal']:.3f}",
        f"- Exact 3-way binary agreement: {agreement['exact_3way_binary']:.3f}",
        f"- Fleiss' kappa (ordinal 3-level): {agreement['fleiss_ordinal_kappa']:.3f}",
        f"- Fleiss' kappa (binary present/absent): {agreement['fleiss_binary_kappa']:.3f}",
        f"- Mean pairwise Cohen's kappa (ordinal): {agreement['pairwise_ordinal_kappa']['mean']:.3f}",
        f"- Mean pairwise Cohen's kappa (binary): {agreement['pairwise_binary_kappa']['mean']:.3f}",
        "",
        "## Task-Type Summary",
    ]

    for row in per_task.itertuples(index=False):
        lines.append(
            "- "
            f"{row.task_type}: {row.items} items, "
            f"majority-present {row.majority_present_rate:.3f}, "
            f"unanimous-binary {row.unanimous_binary_rate:.3f}, "
            f"unanimous-ordinal {row.unanimous_ordinal_rate:.3f}"
        )

    lines.extend([
        "",
        "## Hardest Queried Emotions",
    ])
    for row in hardest.itertuples(index=False):
        lines.append(
            f"- {row.queried_emotion}: majority-present {row.majority_present_rate:.3f} across {row.items} items"
        )

    lines.extend([
        "",
        "## Easiest Queried Emotions",
    ])
    for row in easiest.itertuples(index=False):
        lines.append(
            f"- {row.queried_emotion}: majority-present {row.majority_present_rate:.3f} across {row.items} items"
        )

    lines.extend([
        "",
        "## Benchmark Guidance",
        "- Use `benchmark_labels.csv` as the item-level benchmark table.",
        "- `majority_present` is the clean binary target for benchmarking retrieval/classification.",
        "- `benchmark_bucket` separates stricter subsets such as `unanimous_present` and `unanimous_absent`.",
        "- `all_agree_binary` and `all_agree_ordinal` are useful for confidence-tiered evaluation.",
        "",
        "## Incomplete Items",
        f"- Excluded items: {len(incomplete)}",
    ])
    for row in incomplete.itertuples(index=False):
        lines.append(
            f"- {row.file_name} | {row.queried_emotion} | {row.task_type} | ratings observed: {row.ratings_observed}"
        )

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    raw, cleaned, complete = load_annotations()
    users = pd.read_csv(USERS_PATH)
    cleaned["rating_int"] = cleaned["rating"].map(RATING_TO_INT)
    complete["rating_int"] = complete["rating"].map(RATING_TO_INT)
    complete["present"] = (complete["rating_int"] > 0).astype(int)

    agreement = agreement_summary(complete)
    benchmark_labels = build_item_labels(complete)
    per_task = per_task_summary(benchmark_labels)
    per_emotion = per_emotion_summary(benchmark_labels)
    incomplete = incomplete_item_summary(cleaned)
    summary = build_summary_json(raw, cleaned, complete, users, agreement, per_task, per_emotion, incomplete)
    report = render_report(summary, per_task, per_emotion, incomplete)

    benchmark_labels.to_csv(OUTPUT_DIR / "benchmark_labels.csv", index=False)
    per_task.to_csv(OUTPUT_DIR / "per_task_type_summary.csv", index=False)
    per_emotion.to_csv(OUTPUT_DIR / "per_emotion_summary.csv", index=False)
    incomplete.to_csv(OUTPUT_DIR / "incomplete_items.csv", index=False)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUTPUT_DIR / "report.md").write_text(report)

    print(report)
    print(f"Wrote outputs to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
