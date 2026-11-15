#!/usr/bin/env bash
set -uo pipefail
export PYTHONPATH="$PWD/src:$PWD/packages/mdb/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
echo "== revision $(cat REVISION)"
/usr/bin/time -v timeout ${BUDGET:-280} python r1_smoke.py /mnt/big/llm-models/Qwen3-1.7B-Base /mnt/ssd1tb/cache/beast/twohop-run1-smoke 2> time.err
echo "DRIVER_RC $?"
grep -E "Maximum resident|Elapsed" time.err; grep -v -E "^\s|Loading weights|Warning|warn" time.err | tail -25
