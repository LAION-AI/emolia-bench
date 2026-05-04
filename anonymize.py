"""Anonymize raw annotations and copy them into the public ``annotations/`` tree.

Source layout (gitignored, contains real usernames):

    annotations_raw/
        emolia-emo/{annotations.csv, users.csv}
        emolia-dim/{annotations.csv, users.csv}

Destination layout (committed):

    annotations/
        emolia-emo/{annotations.csv, users.csv}
        emolia-dim/{annotations.csv, users.csv}

Usernames are replaced with stable ``user_0``, ``user_1``, ... ids derived from
sorted username order. Each subset has its own independent mapping so that the
anonymization is reproducible and easy to verify.

Run with: ``uv run anonymize.py``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

RAW_ROOT = Path("annotations_raw")
OUT_ROOT = Path("annotations")
SUBSETS = ("emolia-emo", "emolia-dim")


def build_username_map(*frames: pd.DataFrame) -> dict[str, str]:
    usernames: set[str] = set()
    for frame in frames:
        if frame is None or "username" not in frame.columns:
            continue
        usernames |= set(frame["username"].dropna().astype(str))
    ordered = sorted(usernames)
    return {raw: f"user_{i}" for i, raw in enumerate(ordered)}


def anonymize_subset(subset: str, raw_root: Path, out_root: Path) -> dict[str, int]:
    raw_dir = raw_root / subset
    out_dir = out_root / subset
    out_dir.mkdir(parents=True, exist_ok=True)

    annotations = pd.read_csv(raw_dir / "annotations.csv")
    users = pd.read_csv(raw_dir / "users.csv")
    flags_path = raw_dir / "flags.csv"
    flags = pd.read_csv(flags_path) if flags_path.exists() else None

    mapping = build_username_map(users, annotations, flags)
    annotations["username"] = annotations["username"].map(mapping)
    users["username"] = users["username"].map(mapping)
    users = users.sort_values("username").reset_index(drop=True)

    annotations.to_csv(out_dir / "annotations.csv", index=False)
    users.to_csv(out_dir / "users.csv", index=False)

    flag_count = 0
    if flags is not None:
        flags["username"] = flags["username"].map(mapping)
        flags.to_csv(out_dir / "flags.csv", index=False)
        flag_count = len(flags)

    mapping_path = raw_dir / "_anon_map.csv"
    pd.DataFrame(
        sorted(mapping.items()), columns=["raw_username", "anon_username"]
    ).to_csv(mapping_path, index=False)

    return {
        "annotators": len(mapping),
        "annotations": len(annotations),
        "users": len(users),
        "flags": flag_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--subsets", nargs="+", default=list(SUBSETS))
    args = parser.parse_args()

    for subset in args.subsets:
        stats = anonymize_subset(subset, args.raw_root, args.out_root)
        print(
            f"[{subset}] annotators={stats['annotators']} "
            f"annotations={stats['annotations']} users={stats['users']} "
            f"flags={stats['flags']} "
            f"-> {(args.out_root / subset).resolve()}"
        )


if __name__ == "__main__":
    main()
