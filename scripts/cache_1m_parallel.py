#!/usr/bin/env python3
"""Encode the 1M text set on both GPUs, then merge in original row order."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


RAW = Path("data/mixed_1m")
CACHE = Path("data/cache")
LOGS = Path("logs")
MAX_SEQ_LEN = 160


def split_train():
    shards = [RAW / "train_shard_0.jsonl", RAW / "train_shard_1.jsonl"]
    if all(p.exists() for p in shards):
        return shards
    counts = [0, 0]
    with (RAW / "train.jsonl").open(encoding="utf8") as source, \
         shards[0].open("w", encoding="utf8") as first, \
         shards[1].open("w", encoding="utf8") as second:
        for i, line in enumerate(source):
            shard = 0 if i < 500_000 else 1
            (first if shard == 0 else second).write(line)
            counts[shard] += 1
    if counts != [500_000, 500_000]:
        raise ValueError(f"Expected two 500K shards, found {counts}")
    return shards


def launch(input_path, output_path, gpu, log_name):
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "TOKENIZERS_PARALLELISM": "false",
                "HF_HUB_DISABLE_PROGRESS_BARS": "1", "PYTHONPATH": "src"})
    log = (LOGS / log_name).open("w", encoding="utf8")
    cmd = [sys.executable, "-u", "-m", "t2m.cache_text", "--input", str(input_path),
           "--out", str(output_path), "--max-seq-len", str(MAX_SEQ_LEN), "--batch-size", "32"]
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    return proc, log


def merge(shard_dirs, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    metas = [json.loads((root / "metadata.json").read_text()) for root in shard_dirs]
    total = sum(meta["count"] for meta in metas)
    if total != 1_000_000:
        raise ValueError(f"Expected 1M cached rows, found {total}")
    if len({meta["max_seq_len"] for meta in metas}) != 1:
        raise ValueError("Shard sequence length mismatch")
    specs = {"embeddings": (np.float16, (total, 768)),
             "tokens": (np.int32, (total, MAX_SEQ_LEN - 2)),
             "lengths": (np.uint16, (total,))}
    for stem, (dtype, shape) in specs.items():
        dest = np.lib.format.open_memmap(output_dir / f"{stem}.npy", mode="w+", dtype=dtype, shape=shape)
        offset = 0
        for root, meta in zip(shard_dirs, metas):
            source = np.load(root / f"{stem}.npy", mmap_mode="r")
            for start in range(0, len(source), 8192):
                stop = min(len(source), start + 8192)
                dest[offset + start:offset + stop] = source[start:stop]
            offset += meta["count"]
        dest.flush()
        print("Merged", stem, shape, flush=True)
    metadata = {**metas[0], "count": total, "input_file": str(RAW / "train.jsonl")}
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def main():
    LOGS.mkdir(exist_ok=True)
    shards = split_train()
    roots = [CACHE / "train_1m_shard0", CACHE / "train_1m_shard1"]
    jobs = [launch(shards[i], roots[i], i, f"cache_1m_gpu{i}.log") for i in range(2)]
    watch = LOGS / "cache_1m_watch.jsonl"
    while any(proc.poll() is None for proc, _ in jobs):
        row = {"time": time.time(), "status": [proc.poll() for proc, _ in jobs],
               "rows": [sum(1 for line in (LOGS / f"cache_1m_gpu{i}.log").open()
                            if line.startswith("Cached ")) for i in range(2)]}
        with watch.open("a", encoding="utf8") as f:
            f.write(json.dumps(row) + "\n")
        print(row, flush=True)
        time.sleep(60)
    for proc, log in jobs:
        log.close()
        if proc.returncode:
            raise RuntimeError(f"Cache shard process {proc.pid} exited with {proc.returncode}")
    merge(roots, CACHE / "train_1m")
    print("Encoding 10K validation rows", flush=True)
    proc, log = launch(RAW / "val.jsonl", CACHE / "val_1m", 0, "cache_1m_val.log")
    proc.wait()
    log.close()
    if proc.returncode:
        raise RuntimeError(f"Validation cache process exited with {proc.returncode}")
    for root in roots:
        for stem in ("embeddings.npy", "tokens.npy", "lengths.npy", "metadata.json"):
            (root / stem).unlink()
        root.rmdir()
    for shard in shards:
        shard.unlink()
    print("Ready: data/cache/train_1m and data/cache/val_1m", flush=True)


if __name__ == "__main__":
    main()
