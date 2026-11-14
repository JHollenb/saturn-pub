#!/usr/bin/env bash
# Payload entry point for mrun: runs composer_neurons.py with the mdb source shipped in the payload.
set -uo pipefail
S=/mnt/ssd1tb/cache/beast/mdb-interop          # site dirs: nnsight 0.8.2 (site-tl440-nn082)
export PYTHONPATH="$PWD/src:$PWD/packages/mdb/src:$S/site-libs2:$S/site-tl440-nn082"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME=/mnt/ssd1tb/cache/beast/huggingface TMPDIR=$S/tmp WANDB_DISABLED=true
export MDB_MODEL="${MDB_MODEL:-/mnt/big/llm-models/Qwen3-1.7B-Base}"
export MDB_COUNTRIES="${MDB_COUNTRIES:-}"
echo "== revision $(cat REVISION 2>/dev/null || echo unknown)  countries ${MDB_COUNTRIES:-all}"
/usr/bin/time -v timeout "${BUDGET:-570}" python composer_neurons.py 2> time.err
echo "SCRIPT_RC $?"
grep -E "Maximum resident|Elapsed" time.err; grep -v -E "^\s|Loading weights|Warning|warn" time.err | tail -25
