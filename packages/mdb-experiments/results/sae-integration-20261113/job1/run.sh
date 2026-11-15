#!/usr/bin/env bash
# Job 1: Qwen3-1.7B-Base selfcheck + recall-circuit discovery with engine verbs (no SAE yet).
set -uo pipefail
export PYTHONPATH="$PWD/src:$PWD/packages/mdb/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
M=/mnt/big/llm-models/Qwen3-1.7B-Base
echo "== revision $(cat REVISION)"
python -c "import sys,torch,transformers; print(sys.version.split()[0], torch.__version__, transformers.__version__, torch.cuda.get_device_name(0))"
python gen.py $M || exit 1
cat prompts.json | head -c 4000; echo
echo "== selfcheck"
timeout 300 python -m mdb.engine.selfcheck --model "$M" --dtype bf16 --device cuda --prompt "The Eiffel Tower is located in the city of" --tokens 2 --json selfcheck.json
echo "SELFCHECK_RC $?"
python -c "import json; d=json.load(open('selfcheck.json')); print(json.dumps({k: d.get(k) for k in ('passed','exactness','environment')}, sort_keys=True))" 2>&1 | head -c 6000; echo
for s in recall followup; do
  echo "source -k $s.mdb" > wrap-$s.mdb
  echo "===== $s"
  start=$(date +%s)
  /usr/bin/time -v timeout 420 python -m mdb engine $M --device cuda --dtype bf16 --script wrap-$s.mdb 2> time-$s.err
  echo "RC_$s $? seconds $(( $(date +%s) - start ))"
  grep -E "Maximum resident|Elapsed" time-$s.err; grep -v -E "^\s|Loading weights|Warning|warn" time-$s.err | tail -5
done
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
