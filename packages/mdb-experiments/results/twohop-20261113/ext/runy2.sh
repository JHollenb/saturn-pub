#!/usr/bin/env bash
# Y2 chain view + causal check + GT for families FAMS (TAG names the X1 shard files).
set -uo pipefail
export PYTHONPATH="$PWD/src:$PWD/packages/mdb/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
M=${MODEL:-/mnt/big/llm-models/Qwen3-1.7B-Base}
echo "== revision $(cat REVISION) fams $FAMS tag $TAG"
/usr/bin/time -v timeout ${BUDGET:-580} python y2_view.py $M candidates_ext.json x1-$TAG-panel.json x1-$TAG-forecast.json $FAMS y2-$OUT 2> time.err
echo "DRIVER_RC $?"
grep -E "Maximum resident|Elapsed" time.err; grep -v -E "^\s|Loading weights|Warning|warn" time.err | tail -25
ls -la y2-*.jsonl
