#!/usr/bin/env python3
"""Analysis of the interim MOSS-Audio human audit (paper appendix "Human Audit of the
MOSS-Audio Labels (Interim)").

Input is the anonymized per-item judgements (one row per rater and item), as published at
https://github.com/LAION-AI/emolia-bench/blob/main/annotations/moss_audit_judgements.csv
and produced by src/anonymize_moss_audit.py. Standard library only.

Statistics (real items only unless stated; hidden checks have is_quality_check = 1):

* human-MOSS agreement: share of judgements whose rater level equals the MOSS level
  (exact) or lies within one level, over judgements that carry a MOSS level; the MOSS
  sentence is mapped to a level only where both mapping directions agree;
* permutation baseline: the same statistic with MOSS levels shuffled across those
  judgements, computed as its exact expectation over all permutations;
* human-human agreement: the same statistics over all pairs of raters judging the same item;
* paired difference: human-human minus human-MOSS, on the same bootstrap resamples;
* Spearman correlation between MOSS level and rater level;
* fit acceptance on real items, rejections of hidden checks, and the mean distance between
  the rater level and the planted MOSS level on hidden checks versus real items.

Confidence intervals are 95% percentile intervals from a bootstrap that resamples items
(not judgements), because several raters judge the same item.

    python3 src/moss_analysis.py                        # local anonymized CSV
    python3 src/moss_analysis.py --csv <path-or-URL> --json results.json
"""

import argparse
import csv
import io
import itertools
import json
import random
import statistics
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "neurips-2026-rejection" / "uploads" / "anonymized_judgements" / "moss_audit_judgements.csv"


def load(source):
    if str(source).startswith(("http://", "https://")):
        url = str(source).replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
        with urllib.request.urlopen(url) as resp:
            text = resp.read().decode("utf-8")
    else:
        text = Path(source).read_text()
    return list(csv.DictReader(io.StringIO(text)))


def agreement(pairs):
    """(exact, within-one) agreement over (level, level) pairs."""
    n = len(pairs)
    return (sum(a == b for a, b in pairs) / n,
            sum(abs(a - b) <= 1 for a, b in pairs) / n)


def human_moss_pairs(items, ids):
    return [(r["moss"], r["rater"]) for i in ids for r in items[i] if r["moss"] is not None]


def human_human_pairs(items, ids):
    return [(a["rater"], b["rater"]) for i in ids for a, b in itertools.combinations(items[i], 2)]


def permutation_baseline(pairs):
    """Expected agreement when MOSS levels are shuffled across judgements."""
    moss = [m for m, _ in pairs]
    rater = [r for _, r in pairs]
    n = len(pairs)
    exact = sum(m == r for m in moss for r in rater) / n ** 2
    within = sum(abs(m - r) <= 1 for m in moss for r in rater) / n ** 2
    return exact, within


def spearman(pairs):
    def ranks(values):
        order = sorted(range(len(values)), key=values.__getitem__)
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            for k in range(i, j + 1):
                out[order[k]] = (i + j) / 2
            i = j + 1
        return out
    return statistics.correlation(ranks([m for m, _ in pairs]), ranks([r for _, r in pairs]))


def item_statistics(items, ids):
    hm_exact, hm_within = agreement(human_moss_pairs(items, ids))
    hh_exact, hh_within = agreement(human_human_pairs(items, ids))
    return {
        "human_moss_exact": hm_exact,
        "human_moss_within_one": hm_within,
        "human_human_exact": hh_exact,
        "human_human_within_one": hh_within,
        "difference_exact": hh_exact - hm_exact,
        "difference_within_one": hh_within - hm_within,
    }


def bootstrap(items, n_boot, seed):
    ids = sorted(items)
    rng = random.Random(seed)
    draws = defaultdict(list)
    for _ in range(n_boot):
        sample = [rng.choice(ids) for _ in ids]
        for key, value in item_statistics(items, sample).items():
            draws[key].append(value)
    ci = {}
    for key, values in draws.items():
        values.sort()
        ci[key] = (values[int(0.025 * (n_boot - 1))], values[int(0.975 * (n_boot - 1))])
    return ci


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--csv", default=str(DEFAULT_CSV), help="judgements CSV (path or URL)")
    ap.add_argument("--n-boot", type=int, default=10000, help="bootstrap replicates")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", help="also write the results to this JSON file")
    args = ap.parse_args()

    rows = load(args.csv)
    for r in rows:
        r["moss"] = int(r["moss_level"]) if r["moss_level"] != "" else None
        r["rater"] = int(r["rater_level"])
    real = [r for r in rows if r["is_quality_check"] != "1"]
    checks = [r for r in rows if r["is_quality_check"] == "1"]
    items = defaultdict(list)
    for r in real:
        items[r["item_id"]].append(r)
    ids = sorted(items)

    hm_pairs = human_moss_pairs(items, ids)
    point = item_statistics(items, ids)
    ci = bootstrap(items, args.n_boot, args.seed)
    perm_exact, perm_within = permutation_baseline(hm_pairs)

    results = {
        "n_raters": len({r["rater_id"] for r in rows}),
        "n_judgements": len(real),
        "n_items": len(ids),
        "n_mapped_judgements": len(hm_pairs),
        "n_rater_pairs": len(human_human_pairs(items, ids)),
        "n_hidden_checks": len(checks),
        **{k: {"estimate": v, "ci95": ci[k]} for k, v in point.items()},
        "permutation_exact": perm_exact,
        "permutation_within_one": perm_within,
        "spearman": spearman(hm_pairs),
        "fit_accepted": sum(r["fit"] == "fits" for r in real) / len(real),
        "hidden_checks_rejected": sum(r["fit"] == "no" for r in checks),
        "mean_distance_hidden_checks": statistics.mean(abs(r["moss"] - r["rater"]) for r in checks),
        "mean_distance_real": statistics.mean(abs(m - r) for m, r in hm_pairs),
        "bootstrap": {"unit": "item", "n_boot": args.n_boot, "seed": args.seed},
    }

    def fmt(key):
        e, (lo, hi) = results[key]["estimate"], results[key]["ci95"]
        return f"{e:+.3f} [{lo:+.3f}, {hi:+.3f}]" if key.startswith("difference") else f"{e:.3f} [{lo:.3f}, {hi:.3f}]"

    print(f"{results['n_judgements']} judgements on {results['n_items']} items from "
          f"{results['n_raters']} raters ({results['n_mapped_judgements']} with a MOSS level, "
          f"{results['n_rater_pairs']} rater pairs, {results['n_hidden_checks']} hidden checks)")
    print(f"human-MOSS exact        {fmt('human_moss_exact')}   permutation {perm_exact:.3f}")
    print(f"human-MOSS within one   {fmt('human_moss_within_one')}   permutation {perm_within:.3f}")
    print(f"human-human exact       {fmt('human_human_exact')}")
    print(f"human-human within one  {fmt('human_human_within_one')}")
    print(f"difference exact        {fmt('difference_exact')}")
    print(f"difference within one   {fmt('difference_within_one')}")
    print(f"Spearman MOSS vs rater  {results['spearman']:.3f}")
    print(f"MOSS sentence accepted  {results['fit_accepted']:.3f} of real judgements; "
          f"hidden checks rejected {results['hidden_checks_rejected']} of {len(checks)}")
    print(f"mean level distance     {results['mean_distance_hidden_checks']:.1f} on hidden checks, "
          f"{results['mean_distance_real']:.1f} on real items")

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
