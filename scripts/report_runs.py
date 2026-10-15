#!/usr/bin/env python3
"""Make a compact, auditable table from complete fixed-set evaluations."""
import argparse
import json
from pathlib import Path


def latest_analysis(run):
    roots = sorted((Path("runs") / run / "analysis").glob("step_*"))
    eligible = [root for root in roots if (root / "summary.json").exists()
                and (root / "conditioning_summary.json").exists()]
    if not eligible:
        raise FileNotFoundError(f"Missing mixed and conditioning scores for {run}")
    return eligible[-1]


def value(value):
    return f"{value:.3f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for name in args.runs:
        root = latest_analysis(name)
        mixed = json.loads((root / "summary.json").read_text())
        if mixed["count"] != 80 or mixed["max_new_tokens"] != 128:
            raise ValueError(f"{name}: expected all 80 items at 128 new tokens")
        cond = json.loads((root / "conditioning_summary.json").read_text())
        preview = [json.loads(line) for line in Path(mixed["preview"]).open()]
        if any("ended_with_eos" not in item for item in preview):
            raise ValueError(f"{name}: preview lacks EOS diagnostics")
        media = [item for item in preview if item["modality"] != "text"]
        text_file = root / "text_validation.json"
        text = json.loads(text_file.read_text()) if text_file.exists() else None
        row = {"run": name, "step": cond["step"],
               "image_ref_cosine": mixed["image"]["mean_reference_cosine"],
               "audio_ref_cosine": mixed["audio"]["mean_reference_cosine"],
               "video_ref_cosine": mixed["video"]["mean_reference_cosine"],
               "text_ref_cosine": mixed["text"]["mean_reference_cosine"],
               "image_correct_minus_zero": cond["image"]["correct_minus_zero"],
               "image_correct_minus_shuffled": cond["image"]["correct_minus_shuffled"],
               "image_mean_words": mixed["image"]["mean_generated_words"],
               "audio_mean_words": mixed["audio"]["mean_generated_words"],
               "video_mean_words": mixed["video"]["mean_generated_words"],
               "text_val_ce": text["metrics"]["ce"] if text else None,
               "media_mean_generated_tokens": sum(item["generated_token_count"] for item in media) / len(media),
               "media_eos_rate": sum(item["ended_with_eos"] for item in media) / len(media),
               "analysis_dir": str(root)}
        row["media_macro_ref_cosine"] = sum(row[f"{kind}_ref_cosine"]
                                            for kind in ("image", "audio", "video")) / 3
        rows.append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    (args.out.with_suffix(".json")).write_text(json.dumps(rows, indent=2), encoding="utf8")
    lines = ["# Fixed 80-item mixed evaluation", "",
             "All outputs use greedy decoding with a 128-token limit. Each modality has 20 fixed items.", "",
             "| Variant | Image ref cosine | Audio ref cosine | Video ref cosine | Media macro | Media tokens | Media EOS | Image correct−zero | Image correct−shuffled | Text val CE |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        ce = value(row["text_val_ce"]) if row["text_val_ce"] is not None else "pending"
        lines.append(f"| {row['run']} | {value(row['image_ref_cosine'])} | "
                     f"{value(row['audio_ref_cosine'])} | {value(row['video_ref_cosine'])} | "
                     f"{value(row['media_macro_ref_cosine'])} | "
                     f"{row['media_mean_generated_tokens']:.1f} | {row['media_eos_rate']:.0%} | "
                     f"{value(row['image_correct_minus_zero'])} | "
                     f"{value(row['image_correct_minus_shuffled'])} | {ce} |")
    lines += ["", "Media macro is an unweighted mean of image, audio, and video reference cosine. "
              "Cosine and within-modality retrieval reuse the frozen encoder and are diagnostics, "
              "not independent human judgments. Video references are action labels. "
              "Conditioning likelihood is teacher-forced and token-normalized; zero input is outside "
              "the encoder's unit-norm output manifold. The sample size is small, so inspect the "
              "full per-item JSONL and selected failures alongside the averages.", ""]
    args.out.write_text("\n".join(lines), encoding="utf8")
    print(args.out)


if __name__ == "__main__":
    main()
