#!/usr/bin/env bash
# mrun wrapper for qwen_backtest.py (payload: packages/mdb/src + this dir + r1_items.json)
set -uo pipefail
S=/mnt/ssd1tb/cache/beast/mdb-interop
export PYTHONPATH="$PWD/packages/mdb/src:$PWD:$S/site-libs2"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME=/mnt/ssd1tb/cache/beast/huggingface TMPDIR=$S/tmp
OUT=/mnt/ssd1tb/cache/beast/binlessons/$(cat REVISION)
mkdir -p $OUT
echo "== revision $(cat REVISION)"
timeout 540 python qwen_backtest.py /mnt/big/llm-models/Qwen3-1.7B-Base r1_items.json $OUT 8
echo "DRIVER_RC $? OUT $OUT"
