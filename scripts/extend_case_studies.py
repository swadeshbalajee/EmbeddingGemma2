#!/usr/bin/env python3
"""Add complete final-run outputs to the existing curated cross-run case studies."""
import argparse
import json
from pathlib import Path


def final_preview(run):
    roots = sorted((Path("runs") / run / "analysis").glob("step_*/summary.json"))
    if not roots:
        raise FileNotFoundError(f"No mixed summary for {run}")
    summary = json.loads(roots[-1].read_text())
    if summary["count"] != 80 or summary["max_new_tokens"] != 128:
        raise ValueError(f"{run} needs its complete final 128-token evaluation")
    return {row["path"]: row for row in
            (json.loads(line) for line in Path(summary["preview"]).open())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("runs/analysis/case_studies_200k_vs_1m.jsonl"))
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    studies = [json.loads(line) for line in args.input.open()]
    previews = {run: final_preview(run) for run in args.runs}
    for study in studies:
        for run, rows in previews.items():
            study["outputs"][run] = rows[study["path"]]["generated_text"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as target:
        for study in studies:
            target.write(json.dumps(study, ensure_ascii=False) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
