#!/usr/bin/env bash
# TP1 offline analysis (CPU; reads the per-row arrays on the cache drive)
set -uo pipefail
export PYTHONPATH="$PWD/src:$PWD/packages/mdb/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
echo "== revision $(cat REVISION)"
/usr/bin/time -v timeout ${BUDGET:-580} python tp1_analyze.py /mnt/big/llm-models/Qwen3-1.7B-Base /mnt/ssd1tb/cache/beast/twohop-tp1 tp1-analysis.json 2> time.err
echo "DRIVER_RC $?"
grep -E "Maximum resident|Elapsed" time.err; grep -v -E "^\s|Loading weights|Warning|warn" time.err | tail -25
