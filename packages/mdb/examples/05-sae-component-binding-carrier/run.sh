#!/usr/bin/env bash
# Payload entry point for mrun: runs sae_component_binding.py with the mdb source shipped in the payload.
set -uo pipefail
S=/mnt/ssd1tb/cache/beast/mdb-interop          # site dirs: sae-lens 6.55.5 (site-saelens), einops (site-tl440-nn082)
export PYTHONPATH="$PWD/src:$PWD/packages/mdb/src:$S/site-tl440-nn082:$S/site-saelens"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME=/mnt/ssd1tb/cache/beast/huggingface TMPDIR=$S/tmp MDB_EXAMPLE_DIR=$PWD
export MDB_MODEL="${MDB_MODEL:-/mnt/big/llm-models/Qwen3-1.7B-Base}"
export MDB_SAE="${MDB_SAE:-/mnt/big/llm-models/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50}"
export MDB_NITEMS="${MDB_NITEMS:-24}"
echo "== revision $(cat REVISION 2>/dev/null || echo unknown)  items $MDB_NITEMS"
/usr/bin/time -v timeout "${BUDGET:-570}" python sae_component_binding.py 2> time.err
echo "SCRIPT_RC $?"
grep -E "Maximum resident|Elapsed" time.err
# a failure: the traceback (the resource lines of time -v are dropped)
sed -n "/^Traceback/,/^\tCommand being timed/p" time.err | grep -v "Command being timed" | tail -30 || true
