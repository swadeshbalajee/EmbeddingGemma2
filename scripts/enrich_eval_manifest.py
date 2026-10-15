#!/usr/bin/env python3
"""Attach evaluation references without changing the frozen media embeddings."""
import json
import re
from pathlib import Path

import requests


ROOT = Path("data/eval_mixed")
CACHE = Path("data/cache/eval_mixed")


def write_jsonl(path, rows):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(path)


def main():
    captions = {}
    for split in ("train", "val", "test"):
        url = f"https://huggingface.co/datasets/EGOISTyrh/flickr30k/resolve/main/captions/{split}.json"
        response = requests.get(url, timeout=90)
        response.raise_for_status()
        for item in response.json().values():
            captions[item["filename"]] = (split, item["captions"])

    rows = [json.loads(line) for line in (ROOT / "manifest.jsonl").open(encoding="utf8")]
    for row in rows:
        kind = row["modality"]
        if kind == "image":
            name = Path(row["path"]).name.split("_", 1)[1]
            if name not in captions:
                raise ValueError(f"No Flickr30k reference captions for {name}")
            split, refs = captions[name]
            row.update({"references": refs, "reference_text": refs[0],
                        "reference_type": "five_human_captions", "source_split": split})
        elif kind == "video":
            name = Path(row["path"]).stem
            action = name.split("_g", 1)[0].split("_v_", 1)[1]
            label = re.sub(r"(?<!^)(?=[A-Z])", " ", action)
            row.update({"action_label": label, "reference_text": label,
                        "reference_type": "action_label"})
        elif kind == "audio":
            row["reference_type"] = "speech_transcript"
        else:
            row["reference_type"] = "source_passage"
    write_jsonl(ROOT / "manifest.jsonl", rows)
    for kind in ("image", "audio", "video", "text"):
        write_jsonl(CACHE / f"{kind}_manifest.jsonl",
                    [r for r in rows if r["modality"] == kind])
    print("Enriched", len(rows), "evaluation records with references")


if __name__ == "__main__":
    main()
