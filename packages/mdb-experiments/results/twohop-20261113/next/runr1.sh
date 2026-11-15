#!/usr/bin/env bash
# Run 1 (prereg_run1.json). MODE=1a|1b, PICK = item indices for a smoke.
set -uo pipefail
export PYTHONPATH="$PWD/src:$PWD/packages/mdb/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
echo "== revision $(cat REVISION) mode $MODE pick ${PICK:-all}"
/usr/bin/time -v timeout ${BUDGET:-580} python r1.py /mnt/big/llm-models/Qwen3-1.7B-Base $MODE ${OUTDIR:-/mnt/ssd1tb/cache/beast/twohop-run1} ${PICK:-} 2> time.err
echo "DRIVER_RC $?"
grep -E "Maximum resident|Elapsed" time.err; grep -v -E "^\s|Loading weights|Warning|warn" time.err | tail -25
du -sh ${OUTDIR:-/mnt/ssd1tb/cache/beast/twohop-run1}
