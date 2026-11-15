#!/usr/bin/env bash
# Release-gate 0.2.0 probe job: custody sink naming, then example 03 (smoke countries) with the
# batched f-port probe wrapper. Environment as examples/03 run.sh.
set -uo pipefail
S=/mnt/ssd1tb/cache/beast/mdb-interop
export PYTHONPATH="$PWD/src:$PWD/packages/mdb/src:$S/site-libs2:$S/site-tl440-nn082"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME=/mnt/ssd1tb/cache/beast/huggingface TMPDIR=$S/tmp WANDB_DISABLED=true
export MDB_MODEL="${MDB_MODEL:-/mnt/big/llm-models/Qwen3-1.7B-Base}"
export MDB_COUNTRIES="${MDB_COUNTRIES:-Spain,Japan}"
echo "== revision $(cat REVISION 2>/dev/null || echo unknown)  MRUN_JOB_ID=${MRUN_JOB_ID:-unset}"
python probes/custody_probe.py
echo "CUSTODY_RC $?"
/usr/bin/time -v timeout "${BUDGET:-540}" python probes/probe03_batched.py composer_neurons.py 2> time.err
RC=$?
echo "SCRIPT_RC $RC"
grep -E "Maximum resident|Elapsed" time.err; grep -v -E "^\s|Loading weights|Warning|warn" time.err | tail -40 || true
exit $RC
