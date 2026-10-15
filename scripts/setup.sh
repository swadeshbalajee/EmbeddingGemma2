#!/usr/bin/env bash
set -euo pipefail
conda create -n text2modal python=3.11 -y
# Run `conda activate text2modal` manually in your shell afterward.
echo 'Run: conda activate text2modal && pip install -e . -r requirements.txt'
