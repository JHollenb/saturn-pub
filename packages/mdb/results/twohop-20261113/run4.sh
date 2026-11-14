#!/usr/bin/env bash
# Job 4: natural errors at the carrier (LAYER, KVG from job 2).
set -uo pipefail
export PYTHONPATH="$PWD/src:$PWD/packages/mdb/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
M=/mnt/big/llm-models/Qwen3-1.7B-Base
echo "== revision $(cat REVISION) carrier L${LAYER} kv${KVG}"
/usr/bin/time -v timeout ${BUDGET:-570} python job4_errors.py $M candidates.json job1-panel.json $LAYER $KVG job4 2> time.err
echo "DRIVER_RC $?"
grep -E "Maximum resident|Elapsed" time.err; grep -v -E "^\s|Loading weights|Warning|warn" time.err | tail -15
ls -la job4*.jsonl
