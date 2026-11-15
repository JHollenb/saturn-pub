#!/usr/bin/env bash
# Payload entry point for mrun: runs tl_reparam_floor.py with the mdb source shipped in the payload.
set -uo pipefail
S=/mnt/ssd1tb/cache/beast/mdb-interop          # site dirs: transformer_lens 4.4.0 (site-tl440-nn082), its deps (site-libs2)
export PYTHONPATH="$PWD/src:$PWD/packages/mdb/src:$S/site-libs2:$S/site-tl440-nn082"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME=/mnt/ssd1tb/cache/beast/huggingface TMPDIR=$S/tmp WANDB_DISABLED=true
export MDB_MODELS="${MDB_MODELS:-/mnt/big/llm-models}" MDB_EXAMPLE_DIR="$PWD"
export MDB_NITEMS="${MDB_NITEMS:-24}"
echo "== revision $(cat REVISION 2>/dev/null || echo unknown)  items per model $MDB_NITEMS"
/usr/bin/time -v timeout "${BUDGET:-570}" python tl_reparam_floor.py 2> time.err
echo "SCRIPT_RC $?"
grep -E "Maximum resident|Elapsed" time.err; grep -v -E "^\s|Loading weights|Warning|warn" time.err | tail -25
