"""Encode a unified raw evaluation manifest into the cache used by previews."""
import json
from pathlib import Path

import numpy as np

from .encoder import encode_media, encode_text, make_encoder


def cache_mixed_eval(manifest_path, output_dir, model_id="google/embeddinggemma-2"):
    rows = [json.loads(line) for line in Path(manifest_path).open(encoding="utf8")]
    model = make_encoder(model_id, only_text=False)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    count = 0
    for kind in ("image", "audio", "video", "text"):
        subset = [r for r in rows if r.get("modality") == kind]
        if not subset:
            raise ValueError(f"No {kind} examples in {manifest_path}")
        if kind == "text":
            vectors = encode_text(model, [r["reference_text"] for r in subset])
        else:
            vectors = np.concatenate([encode_media(model, kind, [r["path"]])
                                      for r in subset], axis=0)
        np.save(output / f"{kind}_embeddings.npy", vectors.astype(np.float16))
        with (output / f"{kind}_manifest.jsonl").open("w", encoding="utf8") as f:
            for row in subset:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        count += len(subset)
        print(f"Cached {len(subset)} {kind} examples", flush=True)
    print(f"Cached {count} total examples under {output}", flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/eval_mixed/manifest.jsonl")
    parser.add_argument("--out", default="data/cache/eval_mixed")
    parser.add_argument("--encoder-id", default="google/embeddinggemma-2")
    args = parser.parse_args()
    cache_mixed_eval(args.manifest, args.out, args.encoder_id)


if __name__ == "__main__":
    main()
