#!/usr/bin/env bash
# Extension X1 (passive) for families FAMS on MODEL.
set -uo pipefail
export PYTHONPATH="$PWD/src:$PWD/packages/mdb/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
echo "== revision $(cat REVISION) model $MODEL fams $FAMS"
/usr/bin/time -v timeout ${BUDGET:-580} python x1_panel_forecast.py $MODEL candidates_ext.json x1-$TAG $FAMS ${PERFAM:-16} 2> time.err
echo "DRIVER_RC $?"
grep -E "Maximum resident|Elapsed" time.err; grep -v -E "^\s|Loading weights|Warning|warn" time.err | tail -15
ls -la x1-*.json
