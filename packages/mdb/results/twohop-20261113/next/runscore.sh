#!/usr/bin/env bash
# Run 1 scoring (CPU; reads captures + committed forecasts + arms files)
set -uo pipefail
export PYTHONPATH="$PWD/src:$PWD/packages/mdb/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
echo "== revision $(cat REVISION)"
/usr/bin/time -v timeout ${BUDGET:-580} python r1_score.py /mnt/big/llm-models/Qwen3-1.7B-Base ${CAPDIR:-/mnt/ssd1tb/cache/beast/twohop-run1} r1-score.json 2> time.err
echo "DRIVER_RC $?"
grep -E "Maximum resident|Elapsed" time.err; grep -v -E "^\s|Loading weights|Warning|warn" time.err | tail -25
