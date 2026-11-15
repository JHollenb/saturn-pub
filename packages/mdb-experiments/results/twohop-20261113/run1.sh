#!/usr/bin/env bash
# Job 1 (passive): accuracy, admission, donors, sealed forecast. LIMIT=8 for the smoke.
set -uo pipefail
export PYTHONPATH="$PWD/src:$PWD/packages/mdb/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
M=/mnt/big/llm-models/Qwen3-1.7B-Base
echo "== revision $(cat REVISION) limit ${LIMIT:-all}"
python -c "import sys,torch,transformers; print(sys.version.split()[0], torch.__version__, transformers.__version__, torch.cuda.get_device_name(0))"
/usr/bin/time -v timeout ${BUDGET:-560} python job1_panel_forecast.py $M candidates.json job1 ${LIMIT:-} 2> time.err
echo "DRIVER_RC $?"
grep -E "Maximum resident|Elapsed" time.err; grep -v -E "^\s|Loading weights|Warning|warn" time.err | tail -15
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
ls -la job1-*.json
