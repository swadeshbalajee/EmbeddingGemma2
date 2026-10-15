#!/usr/bin/env bash
set -euo pipefail
# Twenty optimizer steps, with preview generation every five steps.
# Requires data caches; run scripts/prepare_data.sh first.
mkdir -p logs
torchrun --standalone --nproc_per_node=2 -m t2m.train --config configs/smoke.yaml 2>&1 | tee -a logs/benchmark.log
