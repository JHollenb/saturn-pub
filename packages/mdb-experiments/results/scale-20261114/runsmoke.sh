#!/usr/bin/env bash
set -uo pipefail
export PYTHONPATH="$PWD/src:$PWD/packages/mdb/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
M=/mnt/big/llm-models
echo "== revision $(cat REVISION)"
/usr/bin/time -v timeout 580 python scale_smoke.py $MODELS 2> time.err
echo "DRIVER_RC $?"
grep -E "Maximum resident|Elapsed" time.err; grep -A8 Traceback time.err | tail -10; true
