#!/usr/bin/env bash
# B1 native job (prereg_b1.json): MODE=F freeze on base, MODE=D depth arms
set -uo pipefail
export PYTHONPATH="$PWD/src:$PWD/packages/mdb/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
echo "== revision $(cat REVISION) mode $MODE pick ${PICK:-all}"
/usr/bin/time -v timeout ${BUDGET:-580} python b1.py /mnt/big/llm-models/Qwen3-1.7B-Base $MODE ${OUTDIR:-/mnt/ssd1tb/cache/beast/twohop-b1} ${PICK:-} 2> time.err
echo "DRIVER_RC $?"
grep -E "Maximum resident|Elapsed" time.err; grep -v -E "^\s|Loading weights|Warning|warn" time.err | tail -25 || true
