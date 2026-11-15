#!/usr/bin/env bash
# Entity vs bridge: EB1 MEMIT edits (START..STOP) or EB2 (MODE=eb2).
set -uo pipefail
export PYTHONPATH="$PWD/src:$PWD/packages/mdb/src:$PWD/packages/mdb-lanes/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
M=/mnt/big/llm-models/Qwen3-1.7B-Base
echo "== revision $(cat REVISION) mode ${MODE:-eb1} ${START:-0}..${STOP:-0}"
/usr/bin/time -v timeout ${BUDGET:-580} python eb_driver.py $M candidates.json job1-panel.json eb-${MODE:-eb1}-${START:-0}-${STOP:-0} ${START:-0} ${STOP:-0} ${MODE:-eb1} 2> time.err
echo "DRIVER_RC $?"
grep -E "Maximum resident|Elapsed" time.err; grep -v -E "^\s|Loading weights|Warning|warn" time.err | tail -25
ls -la eb-*.jsonl
