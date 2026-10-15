#!/usr/bin/env python3
"""Paired item-bootstrap differences between two fixed 80-item evaluations."""
import argparse
import json
from pathlib import Path

import numpy as np


def load(path):
    rows = [json.loads(line) for line in path.open()]
    if len(rows) != 80:
        raise ValueError(f"Expected 80 items in {path}")
    return {row["path"]: row for row in rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    a, b = load(args.a), load(args.b)
    if a.keys() != b.keys():
        raise ValueError("Evaluation item paths differ")
    rng = np.random.default_rng(2026)
    result = {"a": str(args.a), "b": str(args.b),
              "difference": "a minus b",
              "note": "Paired item-bootstrap 95% intervals on the fixed set; media resamples within each modality to preserve the 20/20/20 mixture. Reference cosine uses the same frozen encoder and is a diagnostic, not independent human judgment."}
    for kind in ("image", "audio", "video", "text", "media"):
        keys = [key for key in a if (a[key]["modality"] == kind or
                                     (kind == "media" and a[key]["modality"] != "text"))]
        result[kind] = {"count": len(keys)}
        for metric in ("reference_cosine", "reference_token_f1", "generated_tokens_approx"):
            diff = np.array([a[key][metric] - b[key][metric] for key in keys], dtype=float)
            if kind == "media":
                grouped = [np.array([a[key][metric] - b[key][metric] for key in keys
                                     if a[key]["modality"] == modality], dtype=float)
                           for modality in ("image", "audio", "video")]
                draws = np.mean([group[rng.integers(0, len(group), size=(10000, len(group)))].mean(axis=1)
                                 for group in grouped], axis=0)
            else:
                draws = diff[rng.integers(0, len(diff), size=(10000, len(diff)))].mean(axis=1)
            lo, hi = np.quantile(draws, [.025, .975])
            result[kind][metric] = {"mean_difference": round(float(diff.mean()), 5),
                                    "bootstrap_ci95": [round(float(lo), 5), round(float(hi), 5)]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf8")
    print(args.out)


if __name__ == "__main__":
    main()
