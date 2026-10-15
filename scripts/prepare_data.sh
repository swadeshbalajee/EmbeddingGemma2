#!/usr/bin/env bash
set -euo pipefail
# All HF downloads and caches occur on your machine, not in the provided ZIP.
python -m t2m.mix --out data/mixed --train-size "${TRAIN_SIZE:-200000}" --val-size "${VAL_SIZE:-2000}"
python -m t2m.cache_text --input data/mixed/train.jsonl --out data/cache/train --max-seq-len 256 --batch-size "${EMB_BATCH:-32}"
python -m t2m.cache_text --input data/mixed/val.jsonl --out data/cache/val --max-seq-len 256 --batch-size "${EMB_BATCH:-32}"
python scripts/prepare_demo_images.py
python -m t2m.cache_media --input-dir "${EVAL_IMAGES:-data/eval_images}" --kind image --out data/cache/eval_media
