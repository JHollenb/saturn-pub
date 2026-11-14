#!/usr/bin/env bash
# Job 2 (interventions) for panel recipients START..STOP-1.
set -uo pipefail
export PYTHONPATH="$PWD/src:$PWD/packages/mdb/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
M=/mnt/big/llm-models/Qwen3-1.7B-Base
echo "== revision $(cat REVISION) recipients ${START}..${STOP}"
/usr/bin/time -v timeout ${BUDGET:-570} python job2_transplants.py $M candidates.json job1-panel.json job1-forecast.json job2-${START}-${STOP} ${START} ${STOP} 2> time.err
echo "DRIVER_RC $?"
grep -E "Maximum resident|Elapsed" time.err; grep -v -E "^\s|Loading weights|Warning|warn" time.err | tail -15
ls -la job2-*.jsonl
