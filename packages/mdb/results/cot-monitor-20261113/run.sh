#!/usr/bin/env bash
set -uo pipefail
export PYTHONPATH="$PWD/src:$PWD/packages/mdb/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=/mnt/big/llm-models/DeepSeek-R1-Distill-Qwen-1.5B
OUTDIR=/mnt/big/mdb-results/cot-monitor-20261113
echo "== revision $(cat REVISION) args: $*"
python -c "import sys,torch,transformers; print(sys.version.split()[0], torch.__version__, transformers.__version__, torch.cuda.get_device_name(0))"
TAG="$1"; shift
/usr/bin/time -v python driver.py $M items.json "$OUTDIR/$TAG.jsonl" "$@" 2> time.err
RC=$?
grep -E "Maximum resident|Elapsed" time.err; grep -v -E "Command being timed|time \(seconds\)|Percent|switches|size|page|File system|Socket|Signals|Swaps|Exit status|Average|Major|Minor" time.err | tail -30
echo "RC $RC"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
exit $RC