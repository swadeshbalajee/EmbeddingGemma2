#!/usr/bin/env python3
"""Continue the fixed 80-item comparison through the three 1M runs.

Run with the text2modal interpreter from the repository root. Every subprocess has
its own log, and the stage journal makes interrupted work inspectable.
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PILOTS = (
    ("noise1_200k", "configs/exp_noise_200k.yaml"),
    ("prefix8_200k", "configs/exp_prefix8_200k.yaml"),
)
FULL = (
    ("baseline_1m", "configs/exp_baseline_1m.yaml"),
    ("noise1_1m", "configs/exp_noise_1m.yaml"),
    ("prefix8_1m", "configs/exp_prefix8_1m.yaml"),
)
LOGS = Path("logs")
JOURNAL = LOGS / "research_pipeline.jsonl"


def record(stage, **data):
    row = {"time_utc": datetime.now(timezone.utc).isoformat(), "stage": stage, **data}
    with JOURNAL.open("a", encoding="utf8") as out:
        out.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def command(stage, argv, *, watch=None):
    env = os.environ.copy()
    env.update({"PYTHONPATH": "src", "TOKENIZERS_PARALLELISM": "false",
                "HF_HUB_DISABLE_PROGRESS_BARS": "1"})
    log_path = LOGS / f"{stage}.log"
    record(stage, state="started", command=argv, log=str(log_path))
    watcher = None
    with log_path.open("w", encoding="utf8") as log:
        proc = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT)
        if watch:
            config, run_dir = watch
            watcher = subprocess.Popen(
                [sys.executable, "-u", "scripts/watch_training.py", "--run",
                 f"{config}:{run_dir}", "--out", str(LOGS / f"watch_{stage}.jsonl"),
                 "--interval", "60"],
                env=env, stdout=(LOGS / f"watch_{stage}.log").open("w", encoding="utf8"),
                stderr=subprocess.STDOUT)
        while proc.poll() is None:
            record(stage, state="running", pid=proc.pid)
            time.sleep(60)
        if watcher:
            watcher.wait(timeout=120)
    record(stage, state="completed" if proc.returncode == 0 else "failed",
           exit_code=proc.returncode)
    if proc.returncode:
        raise RuntimeError(f"{stage} failed; see {log_path}")


def evaluate(name, config):
    final = Path("runs") / name / "checkpoints/final.pt"
    if not final.exists():
        raise FileNotFoundError(final)
    command(f"eval_{name}", [sys.executable, "-u", "scripts/eval_checkpoint.py",
                            "--config", config, "--checkpoint", str(final),
                            "--max-new-tokens", "128"])
    import torch
    package = torch.load(final, map_location="cpu", weights_only=False)
    step = package["step"]
    del package
    preview = Path("runs") / name / "previews" / f"step_{step:07d}.jsonl"
    command(f"score_{name}", [sys.executable, "-u", "scripts/score_mixed.py",
                             "--preview", str(preview)])
    command(f"conditioning_{name}", [sys.executable, "-u", "scripts/eval_mixed_conditioning.py",
                                    "--config", config, "--checkpoint", str(final)])


def wait_for_pilots():
    while True:
        complete = [Path("runs") / name / "checkpoints/final.pt" for name, _ in PILOTS]
        if all(path.exists() for path in complete):
            record("pilots", state="completed")
            return
        record("pilots", state="waiting", complete=[path.exists() for path in complete])
        time.sleep(60)


def compare(names, title):
    comparison = {"comparison": title, "runs": {}}
    for name in names:
        root = Path("runs") / name / "analysis"
        summaries = sorted(root.glob("step_*/summary.json"))
        if not summaries:
            raise FileNotFoundError(f"No mixed score for {name}")
        comparison["runs"][name] = json.loads(summaries[-1].read_text())
    dest = Path("runs") / f"comparison_{title}.json"
    dest.write_text(json.dumps(comparison, indent=2), encoding="utf8")
    record("comparison", name=title, path=str(dest))


def main():
    LOGS.mkdir(exist_ok=True)
    wait_for_pilots()
    command("conditioning_v1b", [sys.executable, "-u", "scripts/eval_mixed_conditioning.py",
                                "--config", "configs/v1b.yaml", "--checkpoint",
                                "runs/v1b/checkpoints/final.pt"])
    for name, config in PILOTS:
        evaluate(name, config)
    compare(("v1b", "noise1_200k", "prefix8_200k"), "200k")
    if not (Path("data/cache/train_1m") / "metadata.json").exists():
        command("cache_1m", [sys.executable, "-u", "scripts/cache_1m_parallel.py"])
    for name, config in FULL:
        final = Path("runs") / name / "checkpoints/final.pt"
        if not final.exists():
            command(f"train_{name}", [sys.executable, "-u", "-m", "torch.distributed.run",
                                      "--standalone", "--nproc_per_node=2", "-m",
                                      "t2m.train", "--config", config],
                    watch=(config, str(Path("runs") / name)))
        evaluate(name, config)
    compare((name for name, _ in FULL), "1m")
    record("pipeline", state="completed")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        record("pipeline", state="failed", error=repr(exc))
        raise
