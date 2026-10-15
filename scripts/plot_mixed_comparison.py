#!/usr/bin/env python3
"""Poster figure for complete fixed-set runs; same-encoder cosine is diagnostic."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def final_rows(run):
    candidates = sorted((Path("runs") / run / "analysis").glob("step_*/summary.json"))
    if not candidates:
        raise FileNotFoundError(f"No scored mixed evaluation for {run}")
    summary = json.loads(candidates[-1].read_text())
    if summary["count"] != 80 or summary["max_new_tokens"] != 128:
        raise ValueError(f"{run}: expected the complete 80-item, 128-token evaluation")
    rows = [json.loads(line) for line in (candidates[-1].parent / "per_item.jsonl").open()]
    if len(rows) != 80:
        raise ValueError(f"{run}: expected 80 per-item scores")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.labels and len(args.labels) != len(args.runs):
        parser.error("--labels must have one label per run")
    labels = args.labels or args.runs
    kinds = ("image", "audio", "video", "text")
    rng = np.random.default_rng(2026)
    means, lows, highs = [], [], []
    for run in args.runs:
        rows = final_rows(run)
        m, lo, hi = [], [], []
        for kind in kinds:
            values = np.array([r["reference_cosine"] for r in rows if r["modality"] == kind])
            if len(values) != 20:
                raise ValueError(f"{run}: expected 20 {kind} scores")
            draws = values[rng.integers(0, 20, size=(10000, 20))].mean(axis=1)
            m.append(values.mean())
            bounds = np.quantile(draws, [.025, .975])
            lo.append(bounds[0]); hi.append(bounds[1])
        means.append(m); lows.append(lo); highs.append(hi)
    means, lows, highs = map(np.asarray, (means, lows, highs))
    fig, ax = plt.subplots(figsize=(9.2, 4.9))
    fig.subplots_adjust(left=.09, right=.98, top=.87, bottom=.25)
    centers = np.arange(4)
    width = .78 / len(args.runs)
    palette = ("#4C78A8", "#F58518", "#54A24B", "#B279A2")
    for i, label in enumerate(labels):
        x = centers - .39 + width * (i + .5)
        yerr = np.stack((means[i] - lows[i], highs[i] - means[i]))
        ax.bar(x, means[i], width=width * .94, label=label, color=palette[i % len(palette)])
        ax.errorbar(x, means[i], yerr=yerr, fmt="none", color="#262626", capsize=2, lw=.9)
    ax.set_xticks(centers, [k.capitalize() for k in kinds])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Generated text ↔ reference cosine")
    ax.set_title("Fixed 80-item mixed set · 128-token greedy decoding")
    ax.legend(frameon=False, ncol=min(len(labels), 3), loc="upper left")
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    fig.text(.09, .035, "Bars: mean over 20 items; whiskers: item-bootstrap 95% intervals.\n"
             "Scoring reuses the frozen encoder; video references are coarse action labels.",
             fontsize=8, color="#444444")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight", pad_inches=.15)
    print(args.out)


if __name__ == "__main__":
    main()
