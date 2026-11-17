#!/usr/bin/env bash
# mrun wrapper: bash run.sh q3|q25   (payload: src + packages/mdb/src + this dir)
set -uo pipefail
S=/mnt/ssd1tb/cache/beast/mdb-interop
export PYTHONPATH="$PWD/src:$PWD/packages/mdb/src:$PWD/nr:$S/site-libs2"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME=/mnt/ssd1tb/cache/beast/huggingface TMPDIR=$S/tmp
OUT=/mnt/ssd1tb/cache/beast/null-recheck/$(cat REVISION)-$1
mkdir -p $OUT
echo "== revision $(cat REVISION) part $1 out $OUT"
cd nr
if [ "$1" = q3 ]; then
  /usr/bin/time -v timeout 540 python q3_recheck.py /mnt/big/llm-models/Qwen3-1.7B-Base data $OUT 2> $OUT/time.err
else
  /usr/bin/time -v timeout 540 python q25_recheck.py /mnt/big/llm-models/Qwen2.5-1.5B data $OUT 2> $OUT/time.err
fi
echo "DRIVER_RC $?"
grep -E "Maximum resident|Elapsed" $OUT/time.err; grep -v -E "^\s|Loading weights|Warning|warn" $OUT/time.err | tail -15
ls -la $OUT
