#!/usr/bin/env bash
# Showcase job wrapper: bash run.sh SCRIPT ARGS...  (payload = git archive src + packages/mdb/src + showcase files)
set -uo pipefail
S=/mnt/ssd1tb/cache/beast/mdb-interop
export PYTHONPATH="$PWD/src:$PWD/packages/mdb/src:$PWD:$S/site-tl440-nn082:$S/site-saelens"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME=/mnt/ssd1tb/cache/beast/huggingface TMPDIR=$S/tmp WANDB_DISABLED=true DATA=$PWD
OUT=${OUT:-/mnt/ssd1tb/cache/beast/showcase}
mkdir -p $OUT
echo "== revision $(cat REVISION) script $1"
/usr/bin/time -v timeout ${BUDGET:-570} python "$@" 2> time.err
echo "DRIVER_RC $?"
grep -E "Maximum resident|Elapsed" time.err; grep -v -E "^\s|Loading weights|Warning|warn" time.err | tail -25 || true
