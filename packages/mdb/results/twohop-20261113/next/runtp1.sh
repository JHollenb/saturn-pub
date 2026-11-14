#!/usr/bin/env bash
# TP1 token x position x sensitivity (prereg_tp1.json)
set -uo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
echo "== revision $(cat REVISION)"
/usr/bin/time -v timeout ${BUDGET:-280} python tp1.py /mnt/big/llm-models/Qwen3-1.7B-Base /mnt/ssd1tb/cache/beast/twohop-tp1 2> time.err
echo "DRIVER_RC $?"
grep -E "Maximum resident|Elapsed" time.err; grep -v -E "^\s|Loading weights|Warning|warn" time.err | tail -25
du -sh /mnt/ssd1tb/cache/beast/twohop-tp1
