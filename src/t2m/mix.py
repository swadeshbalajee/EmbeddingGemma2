"""Text-only mixture builder. Never loads paired media."""
import argparse
import hashlib
import json
import re
from pathlib import Path
from itertools import chain

MIX = {"image": .35, "audio": .25, "video": .20, "prose": .20}


def clean(s):
    if not isinstance(s, str):
        return ""
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) < 22 or len(s) > 1400 or len(s.split()) < 5:
        return ""
    if s.startswith(("http://", "https://")):
        return ""
    return s


def split_prose(text):
    if not isinstance(text, str):
        return
    # Split into coherent segments, not blindly into arbitrary 256-token cuts.
    paras = re.split(r"\n\s*\n", text)
    for para in paras:
        para = clean(para)
        if not para:
            continue
        if len(para) <= 1100:
            yield para
        else:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            buf = ""
            for sentence in sentences:
                if len(buf) + len(sentence) > 950 and buf:
                    yield buf.strip()
                    buf = ""
                buf += " " + sentence
            if clean(buf):
                yield buf.strip()


def extract_wavcaps(obj):
    if isinstance(obj, dict):
        for key in ("caption", "text", "description"):
            if isinstance(obj.get(key), str) and obj[key].strip():
                yield obj[key]
                return  # prefer processed caption over noisy raw description
        for value in obj.values():
            if isinstance(value, (dict, list)):
                yield from extract_wavcaps(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from extract_wavcaps(item)


def iter_source(name, local_path=None):
    if local_path:
        with open(local_path, encoding="utf8") as f:
            for line in f:
                record = json.loads(line)
                if isinstance(record, str):
                    yield record
                elif name == "prose":
                    yield from split_prose(record.get("text", record.get("caption", "")))
                else:
                    val = record.get("text", record.get("caption", record.get("name", "")))
                    if val:
                        yield val
        return
    if name == "audio":
        # WavCaps is stored as nested JSON metadata, NOT a standard arrow table.
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi()
        filenames = [p for p in api.list_repo_files("cvssp/WavCaps", repo_type="dataset")
                     if p.startswith("json_files/") and p.endswith(".json")
                     and "blacklist" not in p]
        if not filenames:
            raise RuntimeError("Cannot find WavCaps caption JSON files")
        for filename in filenames:
            path = hf_hub_download(repo_id="cvssp/WavCaps", filename=filename, repo_type="dataset")
            with open(path, encoding="utf8") as f:
                obj = json.load(f)
            yield from extract_wavcaps(obj)
        return
    from datasets import load_dataset
    if name == "image":
        ds = load_dataset("google-research-datasets/conceptual_captions", "unlabeled",
                          split="train", streaming=True)
        for row in ds:
            yield row.get("caption", "")
    elif name == "video":
        ds = load_dataset("TempoFunk/webvid-10M", split="train", streaming=True)
        for row in ds:
            # Multiple published WebVid versions use 'name' rather than 'caption'.
            yield row.get("name") or row.get("caption") or row.get("text") or ""
    elif name == "prose":
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                          split="train", streaming=True)
        for row in ds:
            yield from split_prose(row.get("text", ""))
    else:
        raise ValueError(name)


def build(outdir, train_size, val_size, local_files=None, allow_shortfall=False):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    target = {k: round(v*train_size) for k,v in MIX.items()}
    target["prose"] += train_size - sum(target.values())
    val_target = {k: round(v*val_size) for k,v in MIX.items()}
    val_target["prose"] += val_size - sum(val_target.values())
    count = {k: 0 for k in MIX}
    val_count = {k: 0 for k in MIX}
    seen = set()
    local_files = local_files or {}
    with (out / "train.jsonl").open("w", encoding="utf8") as train, \
         (out / "val.jsonl").open("w", encoding="utf8") as val:
        for source in MIX:
            try:
                iterator = iter_source(source, local_files.get(source))
                for raw in iterator:
                    sentence = clean(raw)
                    if not sentence:
                        continue
                    h = hashlib.sha1(sentence.lower().encode("utf8")).hexdigest()
                    if h in seen:
                        continue
                    seen.add(h)
                    # Deterministic disjoint split; evaluate with unseen text.
                    is_val = int(h[:8], 16) % 100 < 10
                    if is_val and val_count[source] < val_target[source]:
                        val.write(json.dumps({"text": sentence, "source": source}) + "\n")
                        val_count[source] += 1
                    elif count[source] < target[source]:
                        train.write(json.dumps({"text": sentence, "source": source}) + "\n")
                        count[source] += 1
                    if count[source] >= target[source] and val_count[source] >= val_target[source]:
                        break
            except Exception as exc:
                if not allow_shortfall:
                    raise RuntimeError(f"Failed to load {source}: {exc}. Use --local-{source} for a local JSONL") from exc
                print(f"WARNING: {source} shortfall: {exc}")
            print(source, "train", count[source], "/", target[source], "val", val_count[source], "/", val_target[source], flush=True)
    if (count != target or val_count != val_target) and not allow_shortfall:
        raise RuntimeError("Dataset shortfall: see per-source counts above. No silent repetition performed.")
    metadata = {"train": sum(count.values()), "val": sum(val_count.values()),
                "train_by_source": count, "val_by_source": val_count,
                "mixture": MIX, "text_only": True}
    (out / "manifest.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/mixed")
    p.add_argument("--train-size", type=int, default=200000)
    p.add_argument("--val-size", type=int, default=2000)
    p.add_argument("--allow-shortfall", action="store_true")
    for x in MIX:
        p.add_argument(f"--local-{x}", default=None)
    a = p.parse_args()
    local = {s: getattr(a, f"local_{s}") for s in MIX}
    print(build(a.out, a.train_size, a.val_size, local, a.allow_shortfall))

if __name__ == "__main__":
    main()
