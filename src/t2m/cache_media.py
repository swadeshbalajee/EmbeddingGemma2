import argparse
import json
from pathlib import Path
import numpy as np
from .encoder import make_encoder, encode_media

SUFFIXES = {"image": {".jpg", ".jpeg", ".png", ".webp"},
            "audio": {".wav", ".flac", ".mp3"},
            "video": {".mp4", ".mov", ".webm"}}


def cache_media(input_dir, output_dir, kind, model_id="google/embeddinggemma-2", limit=256):
    source = Path(input_dir)
    files = sorted(p for p in source.rglob("*") if p.suffix.lower() in SUFFIXES[kind])[:limit]
    if not files:
        raise FileNotFoundError(f"No {kind} inputs in {source}")
    model = make_encoder(model_id, only_text=False)
    kept, embeddings = [], []
    for i,p in enumerate(files):
        try:
            z = encode_media(model, kind, [p])[0]
            kept.append({"path": str(p.resolve()), "kind": kind})
            embeddings.append(z)
        except Exception as exc:
            print("Skipping", p, str(exc))
        if i % 10 == 0:
            print(f"Processed {i+1}/{len(files)}", flush=True)
    if not embeddings:
        raise RuntimeError("No media files encoded")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{kind}_embeddings.npy", np.stack(embeddings).astype(np.float16))
    with (out / f"{kind}_manifest.jsonl").open("w", encoding="utf8") as f:
        for row in kept:
            f.write(json.dumps(row)+"\n")
    print("Cached", len(kept), kind, "records in", out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True)
    p.add_argument("--out", default="data/cache/eval_media")
    p.add_argument("--kind", choices=list(SUFFIXES), default="image")
    p.add_argument("--encoder-id", default="google/embeddinggemma-2")
    p.add_argument("--limit", type=int, default=256)
    a = p.parse_args()
    cache_media(a.input_dir, a.out, a.kind, a.encoder_id, a.limit)

if __name__ == "__main__":
    main()
