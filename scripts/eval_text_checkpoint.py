#!/usr/bin/env python3
"""Reproducible held-out text metrics for a completed checkpoint."""
import argparse
import json
from pathlib import Path

import torch

from t2m.config import load_config
from t2m.data import CachedDataset
from t2m.eval import evaluate_text
from t2m.train import make_model, check_cache_metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--examples", type=int, default=1024)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CachedDataset(cfg["val_cache"])
    check_cache_metadata(dataset, cfg)
    model, tok = make_model(cfg, device)
    package = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(package["model"])
    # Keep both the sampled examples and in-batch negatives identical across variants.
    batch_size = 8
    batches = min(len(dataset) // batch_size, args.examples // batch_size)
    metrics = evaluate_text(model, dataset, tok.bos_token_id or tok.eos_token_id,
                            tok.pad_token_id or tok.eos_token_id, device,
                            batch_size=batch_size, n_batches=batches)
    result = {"checkpoint": args.checkpoint, "step": int(package["step"]),
              "validation_examples": batches * batch_size, "metrics": metrics}
    out = Path(cfg["output_dir"]) / "analysis" / f"step_{package['step']:07d}" / "text_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
