#!/usr/bin/env python3
"""Plot actual decoding lengths and EOS rates on the fixed 80-item set."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("preview", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.preview.open()]
    if len(rows) != 80 or any(row.get("max_new_tokens") != 128 for row in rows):
        raise ValueError("Expected 80 fixed items decoded with max_new_tokens=128")
    kinds = ("image", "audio", "video", "text")
    groups = [[row for row in rows if row["modality"] == kind] for kind in kinds]
    if any(len(group) != 20 or "ended_with_eos" not in group[0] for group in groups):
        raise ValueError("Need 20 items and EOS fields per modality")
    lengths = [[row["generated_token_count"] for row in group] for group in groups]
    labels = [f"{kind.capitalize()}\nEOS {sum(row['ended_with_eos'] for row in group)}/20"
              for kind, group in zip(kinds, groups)]
    fig, ax = plt.subplots(figsize=(7.4, 4.8), layout="constrained")
    bp = ax.boxplot(lengths, tick_labels=labels, patch_artist=True, showmeans=True)
    for patch, color in zip(bp["boxes"], ("#4c78a8", "#72b7b2", "#54a24b", "#f2cf5b")):
        patch.set_facecolor(color)
        patch.set_alpha(.82)
    ax.axhline(128, color="#d55e56", linestyle="--", linewidth=1, label="128-token cap")
    ax.set_ylabel("Generated tokens, including EOS")
    ax.set_title("Learned early stopping on media inputs (20 items per modality)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    out = args.out or args.preview.parent.parent / "analysis" / args.preview.stem / "output_lengths.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    print(out)


if __name__ == "__main__":
    main()
