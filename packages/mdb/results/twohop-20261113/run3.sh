#!/usr/bin/env bash
# Job 3: SAE census at the job-2 carrier, recipients START..STOP-1 of sorted ids in gt.json.
set -uo pipefail
export PYTHONPATH="$PWD/src:$PWD/packages/mdb/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
M=/mnt/big/llm-models/Qwen3-1.7B-Base
SAE=/mnt/big/llm-models/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50
echo "== revision $(cat REVISION) recipients ${START}..${STOP}"
/usr/bin/time -v timeout ${BUDGET:-570} python job3_census.py $M $SAE candidates.json job1-panel.json gt.json job3-${START}-${STOP} ${START} ${STOP} 2> time.err
echo "DRIVER_RC $?"
grep -E "Maximum resident|Elapsed" time.err; grep -v -E "^\s|Loading weights|Warning|warn" time.err | tail -15
ls -la job3-*.jsonl
