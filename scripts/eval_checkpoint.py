#!/usr/bin/env python3
"""Generate mixed-modality previews from a saved decoder checkpoint."""
import argparse
from pathlib import Path

import torch

from t2m.config import load_config
from t2m.eval import evaluate_media
from t2m.train import make_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v1c.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = make_model(cfg, device)
    package = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(package["model"])
    step = args.step if args.step is not None else int(package["step"])
    output_dir = args.output_dir or cfg["output_dir"]
    path, count = evaluate_media(
        model, tokenizer, cfg["eval_cache"], output_dir, step, cfg["variant"], device,
        limit=int(cfg.get("media_eval_limit", 32)),
        max_new_tokens=int(args.max_new_tokens or cfg.get("max_new_tokens", 128)),
        repetition_penalty=float(cfg.get("repetition_penalty", 1.08)))
    print(f"Wrote {count} rows to {Path(path).resolve()}")


if __name__ == "__main__":
    main()
