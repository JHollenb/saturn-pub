#!/usr/bin/env bash
# TP1 token x position x sensitivity, integrated (nnsight backward, mdb exact parts, SAELens, TL)
set -uo pipefail
S=/mnt/ssd1tb/cache/beast/mdb-interop
export PYTHONPATH="$PWD/src:$PWD/packages/mdb/src:$PWD/interop_pkg:$S/site-tl440-nn082:$S/site-saelens"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME=/mnt/ssd1tb/cache/beast/huggingface TMPDIR=$S/tmp WANDB_DISABLED=true
echo "== revision $(cat REVISION) interop $(cat interop_pkg/SOURCE_REV)"
/usr/bin/time -v timeout ${BUDGET:-300} python tp1.py /mnt/big/llm-models/Qwen3-1.7B-Base /mnt/big/llm-models/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50 /mnt/ssd1tb/cache/beast/twohop-tp1 2> time.err
echo "DRIVER_RC $?"
grep -E "Maximum resident|Elapsed" time.err; grep -v -E "^\s|Loading weights|Warning|warn" time.err | tail -25
du -sh /mnt/ssd1tb/cache/beast/twohop-tp1
