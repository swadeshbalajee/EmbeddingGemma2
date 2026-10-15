#!/usr/bin/env bash
set -euo pipefail
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
mkdir -p logs
torchrun --standalone --nproc_per_node=2 -m t2m.train --config configs/v1c.yaml "$@" 2>&1 | tee -a logs/v1c.log
