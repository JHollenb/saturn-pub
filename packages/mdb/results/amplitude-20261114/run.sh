#!/usr/bin/env bash
# Amplitude job wrapper: bash run.sh STEP...   each STEP = "native PHASE [SHARD]" | "replica DEVICE [SHARD]"
# env: MODEL (dir name under /mnt/big/llm-models), OUTROOT, BUDGET (s, whole job), SMOKE, PART
set -uo pipefail
export PYTHONPATH="$PWD/src:$PWD/packages/mdb/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME=/mnt/ssd1tb/cache/beast/huggingface WANDB_DISABLED=true DATA=$PWD/data OMP_NUM_THREADS=${OMP:-12}
OUT=${OUTROOT:-/mnt/ssd1tb/cache/beast/amplitude}/$MODEL
mkdir -p $OUT/results
echo "== revision $(cat REVISION) model $MODEL steps: $*"
END=$(( $(date +%s) + ${BUDGET:-570} ))
for step in "$@"; do
  left=$(( END - $(date +%s) ))
  echo "== step [$step] budget ${left}s"
  /usr/bin/time -v timeout $left python amp.py $(echo $step | awk '{print $1}') /mnt/big/llm-models/$MODEL $OUT $(echo $step | cut -d' ' -f2-) 2> time.err
  echo "STEP_RC $?"
  grep -E "Maximum resident|Elapsed" time.err; grep -v -E "^\s|Loading weights|Warning|warn" time.err | tail -12 || true
done
cp -f *.jsonl pools-*.json $OUT/results/ 2>/dev/null
nvidia-smi --query-gpu=memory.used --format=csv 2>/dev/null | tail -1
ls $OUT/results | head -40
