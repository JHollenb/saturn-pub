#!/usr/bin/env bash
# Payload entry point for mrun: runs sae_census.py with the mdb source shipped in the payload.
set -uo pipefail
S=/mnt/ssd1tb/cache/beast/mdb-interop          # site dirs: sae-lens 6.55.5 (SAELens) and einops (site-tl440-nn082)
export PYTHONPATH="$PWD/src:$PWD/packages/mdb/src:$S/site-tl440-nn082:$S/site-saelens"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export MDB_MODEL="${MDB_MODEL:-/mnt/big/llm-models/Qwen3-1.7B-Base}"
export MDB_SAE="${MDB_SAE:-/mnt/big/llm-models/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50}"
export MDB_NPAIRS="${MDB_NPAIRS:-6}"
echo "== revision $(cat REVISION 2>/dev/null || echo unknown)  npairs $MDB_NPAIRS"
/usr/bin/time -v timeout "${BUDGET:-540}" python sae_census.py 2> time.err
echo "SCRIPT_RC $?"
grep -E "Maximum resident|Elapsed" time.err; grep -v -E "^\s|Loading weights|Warning|warn" time.err | tail -15
