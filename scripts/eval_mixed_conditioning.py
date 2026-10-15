#!/usr/bin/env python3
"""Teacher-forced correct/shuffled/zero embedding scores on the fixed mixed set."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from t2m.config import load_config
from t2m.data import collate
from t2m.train import make_model


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = make_model(cfg, device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    step = int(state["step"])
    bos = tokenizer.bos_token_id or tokenizer.eos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    root = Path(cfg["eval_cache"])
    output = Path(args.out or Path(cfg["output_dir"]) / "analysis" / f"step_{step:07d}")
    output.mkdir(parents=True, exist_ok=True)
    result = []
    max_target = int(cfg["max_seq_len"]) - 3

    for kind in ("image", "audio", "video", "text"):
        vectors = np.load(root / f"{kind}_embeddings.npy").astype(np.float32)
        manifest = [json.loads(line) for line in (root / f"{kind}_manifest.jsonl").open()]
        if len(vectors) != len(manifest):
            raise ValueError(f"{kind}: cache and manifest count differ")
        for i, row in enumerate(manifest):
            references = row.get("references") or [row["reference_text"]]
            tokens = tokenizer(references, add_special_tokens=False, truncation=True,
                               max_length=max_target)["input_ids"]
            targets = [torch.tensor(ids + [eos], dtype=torch.long) for ids in tokens]
            wrong = vectors[(i + 1) % len(vectors)]
            z = torch.tensor(vectors[i], device=device).expand(len(targets), -1)
            z_wrong = torch.tensor(wrong, device=device).expand(len(targets), -1)
            z_zero = torch.zeros_like(z)
            x = collate([(torch.zeros(768), target) for target in targets], bos, pad)
            inp, labels, mask = [x[k].to(device) for k in ("input_ids", "labels", "mask")]
            scores = []
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=device.type == "cuda" and torch.cuda.is_bf16_supported()):
                for embedding in (z, z_wrong, z_zero):
                    score, _, _ = model.token_logprobs(embedding, inp, labels, mask)
                    scores.append(float(score.mean()))
            correct, shuffled, zero = scores
            result.append({"modality": kind, "path": row["path"],
                           "reference_count": len(references),
                           "reference_type": row.get("reference_type"),
                           "correct_logprob_per_token": round(correct, 5),
                           "shuffled_logprob_per_token": round(shuffled, 5),
                           "zero_logprob_per_token": round(zero, 5),
                           "correct_minus_shuffled": round(correct - shuffled, 5),
                           "correct_minus_zero": round(correct - zero, 5),
                           "correct_beats_shuffled": correct > shuffled,
                           "correct_beats_zero": correct > zero})
    with (output / "conditioning_per_item.jsonl").open("w", encoding="utf8") as f:
        for row in result:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"checkpoint": args.checkpoint, "step": step,
               "note": "Teacher-forced token-normalized log likelihood; shuffled embedding is the next fixed item within modality. Images average over five captions; video references are coarse action labels."
               }
    for kind in ("image", "audio", "video", "text"):
        rows = [r for r in result if r["modality"] == kind]
        summary[kind] = {"count": len(rows)}
        for field in ("correct_minus_shuffled", "correct_minus_zero",
                      "correct_beats_shuffled", "correct_beats_zero"):
            summary[kind][field] = round(float(np.mean([r[field] for r in rows])), 5)
        rng = np.random.default_rng(1234)
        for field in ("correct_minus_shuffled", "correct_minus_zero"):
            values = np.asarray([r[field] for r in rows], dtype=np.float64)
            means = values[rng.integers(0, len(values), size=(10000, len(values)))].mean(axis=1)
            summary[kind][field + "_bootstrap_ci95"] = [
                round(float(value), 5) for value in np.quantile(means, [.025, .975])]
    (output / "conditioning_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
