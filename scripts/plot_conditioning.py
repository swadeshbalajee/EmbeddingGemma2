#!/usr/bin/env python3
"""Plot fixed-set conditional likelihood differences for poster inspection."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    data = json.loads(args.summary.read_text())
    kinds = ("image", "audio", "video", "text")
    rows = [json.loads(line) for line in args.summary.with_name("conditioning_per_item.jsonl").open()]
    rng = np.random.default_rng(42)

    def values(field):
        means, lower, upper = [], [], []
        for kind in kinds:
            sample = np.array([row[field] for row in rows if row["modality"] == kind])
            draws = sample[rng.integers(0, len(sample), size=(5000, len(sample)))].mean(axis=1)
            mean = sample.mean()
            lo, hi = np.quantile(draws, [.025, .975])
            means.append(mean)
            lower.append(mean - lo)
            upper.append(hi - mean)
        return means, np.array([lower, upper])

    shuffled, shuffled_err = values("correct_minus_shuffled")
    zero, zero_err = values("correct_minus_zero")
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(8.2, 4.6), layout="constrained")
    ax.bar(x - .18, shuffled, .35, yerr=shuffled_err, capsize=3,
           color="#3775a9", label="Correct − shuffled")
    ax.bar(x + .18, zero, .35, yerr=zero_err, capsize=3,
           color="#e3a03f", label="Correct − zero input")
    ax.axhline(0, color="black", linewidth=.8)
    ax.set_xticks(x, [kind.capitalize() for kind in kinds])
    ax.set_ylabel("Mean token log-likelihood difference (nats)")
    ax.set_title(f"Frozen embedding conditioning, step {data['step']:,} (n = 20 per modality)\n"
                 "Error bars: 95% item bootstrap intervals")
    ax.legend(frameon=False, ncols=2)
    ax.spines[["top", "right"]].set_visible(False)
    out = args.out or args.summary.with_name("conditioning_deltas.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    print(out)


if __name__ == "__main__":
    main()
