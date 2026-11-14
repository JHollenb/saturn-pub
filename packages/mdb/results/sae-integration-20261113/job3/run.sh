#!/usr/bin/env bash
set -uo pipefail
export PYTHONPATH="$PWD/src:$PWD/packages/mdb/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
M=/mnt/big/llm-models/Qwen3-1.7B-Base
SAE=/mnt/big/llm-models/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50
NPAIRS=${NPAIRS:-16}
echo "== revision $(cat REVISION) npairs $NPAIRS"
/usr/bin/time -v timeout ${BUDGET:-840} python driver.py $M $SAE job3 $NPAIRS 16,18,19,20,21,22 4,10,16 8,14,18,20,22,24 ${DETAIL:-1} ${PRELUDE:-prelude} ${OFFSET:-0} 2> time.err
echo "DRIVER_RC $?"
grep -E "Maximum resident|Elapsed" time.err; grep -v -E "^\s|Loading weights|Warning|warn" time.err | tail -15
