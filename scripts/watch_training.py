#!/usr/bin/env python3
"""Record run, GPU, and disk health while long experiments execute."""
import argparse
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def latest_json(path):
    if not path.exists():
        return None
    with path.open("rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 8192))
        lines = f.read().splitlines()
    for line in reversed(lines):
        try:
            return json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
    return None


def running(config):
    result = subprocess.run(["ps", "-eo", "pid,args"], text=True, capture_output=True, check=True)
    return [line.strip() for line in result.stdout.splitlines()[1:]
            if "-m t2m.train --config" in line and config in line]


def gpu_status():
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                             "--format=csv,noheader,nounits"], text=True, capture_output=True)
    return result.stdout.strip().splitlines() if result.returncode == 0 else []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True,
                        help="CONFIG:RUN_DIR, repeat for each run")
    parser.add_argument("--out", default="logs/watch_training.jsonl")
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()
    tracked = [entry.split(":", 1) for entry in args.run]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    while True:
        disk_gb = shutil.disk_usage(".").free / 1e9
        row = {"time_utc": datetime.now(timezone.utc).isoformat(),
               "disk_free_gb": round(disk_gb, 2), "gpus": gpu_status(), "runs": {}}
        done = True
        failed = False
        for config, run_dir in tracked:
            root = Path(run_dir)
            latest = latest_json(root / "metrics.jsonl")
            processes = running(config)
            final = (root / "checkpoints" / "final.pt").exists()
            state = {"running": bool(processes), "completed": final,
                     "last_step": latest.get("step") if latest else None,
                     "last_phase": latest.get("phase") if latest else None,
                     "val_ce": latest.get("val/ce") if latest else None,
                     "metrics_age_seconds": round(time.time() - (root / "metrics.jsonl").stat().st_mtime)
                     if (root / "metrics.jsonl").exists() else None}
            row["runs"][run_dir] = state
            done &= final
            failed |= (not processes and not final)
        with out.open("a", encoding="utf8") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        if disk_gb < 1.0:
            print("ALERT: less than 1 GB free disk", flush=True)
        if failed:
            print("ALERT: a run exited without a final checkpoint", flush=True)
            raise SystemExit(1)
        if done:
            print("All monitored runs completed", flush=True)
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
