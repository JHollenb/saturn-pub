#!/usr/bin/env bash
# FT1 first-token / sink job (prereg_firsttoken.json). PICK = comma-separated item indices (smoke).
set -uo pipefail
export PYTHONPATH="$PWD/src:$PWD/packages/mdb/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
M=${MODEL:-/mnt/big/llm-models/Qwen3-1.7B-Base}
echo "== revision $(cat REVISION) pick ${PICK:-all}"
/usr/bin/time -v timeout ${BUDGET:-580} python ft1.py $M ft1-$OUT ${PICK:-} 2> time.err
echo "DRIVER_RC $?"
grep -E "Maximum resident|Elapsed" time.err; grep -v -E "^\s|Loading weights|Warning|warn" time.err | tail -25
ls -la ft1-*.jsonl
