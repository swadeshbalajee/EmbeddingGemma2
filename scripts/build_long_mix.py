#!/usr/bin/env python3
"""Build one million text-only examples with substantially longer image captions."""
import hashlib
import json
import re
from collections import Counter
from contextlib import closing
from pathlib import Path

import ijson
import requests

from t2m.mix import clean, iter_source


OUT = Path("data/mixed_1m")
TRAIN_TARGET = {"image": 450_000, "audio": 150_000, "video": 150_000, "prose": 250_000}
VAL_TARGET = {"image": 4_500, "audio": 1_500, "video": 1_500, "prose": 2_500}
DETAIL_TARGET = 300_000
DETAIL_VAL_TARGET = 3_000
SHAREGPT4V_URL = (
    "https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/resolve/main/"
    "share-captioner_coco_lcs_sam_1246k_1107.json"
)


def detailed_caption(value):
    """Keep a coherent 25-105 word description from a longer image report."""
    value = re.sub(r"\s+", " ", value).strip()
    if not value:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", value)
    result = []
    for sentence in sentences:
        candidate = " ".join(result + [sentence])
        if len(candidate.split()) > 105 and result:
            break
        result.append(sentence)
        if len(candidate.split()) >= 55:
            break
    candidate = " ".join(result).strip()
    words = candidate.split()
    if len(words) > 105:
        candidate = " ".join(words[:105]).rstrip(",;:") + "."
    return candidate if 25 <= len(candidate.split()) <= 105 else ""


def sharegpt4v_captions():
    response = requests.get(SHAREGPT4V_URL, stream=True, timeout=120)
    response.raise_for_status()
    response.raw.decode_content = True
    try:
        for item in ijson.items(response.raw, "item"):
            replies = [turn.get("value", "") for turn in item.get("conversations", [])
                       if turn.get("from") == "gpt"]
            if replies:
                caption = detailed_caption(replies[0])
                if caption:
                    yield caption
    finally:
        response.close()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train_temp, val_temp = OUT / "train.jsonl.tmp", OUT / "val.jsonl.tmp"
    train_counts, val_counts = Counter(), Counter()
    source_counts = Counter()
    seen = set()
    evaluated = {json.loads(s).get("reference_text", "").strip().lower()
                 for s in Path("data/eval_mixed/manifest.jsonl").open(encoding="utf8")}

    last_reported = Counter()
    with train_temp.open("w", encoding="utf8") as train, val_temp.open("w", encoding="utf8") as val:
        def add(text, category, dataset, allow_train=True, allow_val=True,
                train_cap=None, val_cap=None, force_val=False):
            text = clean(text)
            if not text or text.lower() in evaluated:
                return False
            digest = hashlib.sha1(text.lower().encode("utf8")).hexdigest()
            if digest in seen:
                return False
            tcap = min(TRAIN_TARGET[category], train_cap or TRAIN_TARGET[category])
            vcap = min(VAL_TARGET[category], val_cap or VAL_TARGET[category])
            record = json.dumps({"text": text, "source": category, "dataset": dataset}, ensure_ascii=False) + "\n"
            if allow_val and val_counts[category] < vcap and (force_val or int(digest[:8], 16) % 100 < 10):
                val.write(record)
                val_counts[category] += 1
            elif allow_train and train_counts[category] < tcap:
                train.write(record)
                train_counts[category] += 1
            else:
                return False
            seen.add(digest)
            source_counts[dataset] += 1
            return True

        old_datasets = {"image": "google-research-datasets/conceptual_captions",
                        "audio": "cvssp/WavCaps", "video": "TempoFunk/webvid-10M",
                        "prose": "HuggingFaceFW/fineweb-edu"}
        for split, filename in (("train", "data/mixed/train.jsonl"), ("val", "data/mixed/val.jsonl")):
            for line in Path(filename).open(encoding="utf8"):
                row = json.loads(line)
                add(row["text"], row["source"], old_datasets[row["source"]],
                    allow_train=split == "train", allow_val=split == "val",
                    force_val=split == "val")
        print("Copied existing data:", dict(train_counts), dict(val_counts), flush=True)

        # From ShareGPT4V use only the generated descriptions, never its images.
        with closing(sharegpt4v_captions()) as captions:
            for caption in captions:
                add(caption, "image", "Lin-Chen/ShareGPT4V",
                    train_cap=70_000 + DETAIL_TARGET, val_cap=700 + DETAIL_VAL_TARGET)
                if train_counts["image"] >= 70_000 + DETAIL_TARGET and val_counts["image"] >= 700 + DETAIL_VAL_TARGET:
                    break
                if train_counts["image"] and train_counts["image"] % 25_000 == 0 and train_counts["image"] != last_reported["image"]:
                    print("Detailed image captions:", train_counts["image"], flush=True)
                    last_reported["image"] = train_counts["image"]
        print("After detailed captions:", train_counts["image"], val_counts["image"], flush=True)

        for category in ("image", "audio", "video", "prose"):
            for caption in iter_source(category):
                add(caption, category, old_datasets[category])
                if train_counts[category] >= TRAIN_TARGET[category] and val_counts[category] >= VAL_TARGET[category]:
                    break
                if train_counts[category] and train_counts[category] % 25_000 == 0 and train_counts[category] != last_reported[category]:
                    print(category, train_counts[category], "/", TRAIN_TARGET[category], flush=True)
                    last_reported[category] = train_counts[category]
            print("Completed", category, train_counts[category], val_counts[category], flush=True)

    if dict(train_counts) != TRAIN_TARGET or dict(val_counts) != VAL_TARGET:
        raise RuntimeError(f"Incomplete 1M mixture: train={dict(train_counts)}, val={dict(val_counts)}")
    train_temp.replace(OUT / "train.jsonl")
    val_temp.replace(OUT / "val.jsonl")
    metadata = {"train": sum(train_counts.values()), "val": sum(val_counts.values()),
                "train_by_source": dict(train_counts), "val_by_source": dict(val_counts),
                "datasets": dict(source_counts), "text_only": True,
                "image_detailed_target": DETAIL_TARGET}
    (OUT / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf8")
    print("Completed 1M text-only mixture:", metadata, flush=True)


if __name__ == "__main__":
    main()
