#!/usr/bin/env python3
"""Download a deterministic 20-per-modality real-world evaluation set."""
import json
import random
from pathlib import Path
from urllib.parse import quote

import requests
from datasets import Audio, load_dataset


ROOT = Path("data/eval_mixed")
SEED = 206
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "GoogleEmbeddingModel-eval-set/1.0"})


def api_tree(repo):
    url = f"https://huggingface.co/api/datasets/{repo}/tree/main?recursive=true&expand=false"
    response = SESSION.get(url, timeout=45)
    response.raise_for_status()
    return response.json()


def download(repo, filename, dest):
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{quote(filename, safe='/')}"
    response = SESSION.get(url, timeout=90)
    response.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(response.content)


def main():
    rng = random.Random(SEED)
    rows = []
    for kind in ("image", "audio", "video", "text"):
        (ROOT / kind).mkdir(parents=True, exist_ok=True)

    image_repo = "EGOISTyrh/flickr30k"
    image_files = [x for x in api_tree(image_repo)
                   if x.get("path", "").lower().endswith((".jpg", ".jpeg", ".png"))
                   and x.get("size", 0) > 10_000]
    rng.shuffle(image_files)
    chosen = image_files[:20]
    if len(chosen) != 20:
        raise RuntimeError(f"Only found {len(chosen)} Flickr30k images")
    for i, item in enumerate(chosen, 1):
        name = Path(item["path"]).name
        dest = ROOT / "image" / f"{i:02d}_{name}"
        if not dest.exists():
            download(image_repo, item["path"], dest)
        rows.append({"modality": "image", "path": str(dest.resolve()),
                     "dataset": image_repo, "source": f"https://huggingface.co/datasets/{image_repo}"})
        print(f"Downloaded image {i}/20", flush=True)

    audio_repo = "PolyAI/minds14"
    audio_ds = load_dataset(audio_repo, "en-US", split="train")
    audio_ds = audio_ds.cast_column("audio", Audio(decode=False))
    indexes = list(range(len(audio_ds)))
    rng.shuffle(indexes)
    for i, index in enumerate(indexes[:20], 1):
        item = audio_ds[index]
        audio = item["audio"]
        suffix = Path(audio.get("path") or "sample.wav").suffix or ".wav"
        dest = ROOT / "audio" / f"{i:02d}{suffix.lower()}"
        if not dest.exists():
            dest.write_bytes(audio["bytes"])
        rows.append({"modality": "audio", "path": str(dest.resolve()),
                     "dataset": audio_repo, "source": f"https://huggingface.co/datasets/{audio_repo}",
                     "reference_text": item.get("english_transcription") or item.get("transcription", "")})
        print(f"Saved audio {i}/20", flush=True)

    video_repo = "bitmind/UCF101-Videos"
    video_files = [x for x in api_tree(video_repo)
                   if x.get("path", "").lower().endswith((".avi", ".mp4", ".mov"))
                   and 20_000 <= x.get("size", 0) <= 1_000_000]
    rng.shuffle(video_files)
    chosen = video_files[:20]
    if len(chosen) != 20:
        raise RuntimeError(f"Only found {len(chosen)} small UCF101 videos")
    for i, item in enumerate(chosen, 1):
        name = Path(item["path"].replace("\\", "/")).name
        dest = ROOT / "video" / f"{i:02d}_{name}"
        if not dest.exists():
            download(video_repo, item["path"], dest)
        rows.append({"modality": "video", "path": str(dest.resolve()),
                     "dataset": video_repo, "source": f"https://huggingface.co/datasets/{video_repo}"})
        print(f"Downloaded video {i}/20", flush=True)

    # These paragraphs were sampled from the repo's FineWeb-Edu validation split.
    val = [json.loads(line) for line in Path("data/mixed/val.jsonl").open(encoding="utf8")]
    prose = [r["text"] for r in val if r.get("source") == "prose" and r.get("text")]
    rng.shuffle(prose)
    if len(prose) < 20:
        raise RuntimeError(f"Only found {len(prose)} FineWeb-Edu validation paragraphs")
    for i, text in enumerate(prose[:20], 1):
        dest = ROOT / "text" / f"{i:02d}.txt"
        dest.write_text(text, encoding="utf8")
        rows.append({"modality": "text", "path": str(dest.resolve()), "dataset": "HuggingFaceFW/fineweb-edu",
                     "source": "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu",
                     "reference_text": text})

    manifest = ROOT / "manifest.jsonl"
    with manifest.open("w", encoding="utf8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} examples to {manifest}", flush=True)


if __name__ == "__main__":
    main()
