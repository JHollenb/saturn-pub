#!/usr/bin/env bash
# Payload entry point for mrun: runs interchange_subspace.py with the mdb source shipped in the payload.
set -uo pipefail
S=/mnt/ssd1tb/cache/beast/mdb-interop          # site dirs: pyvene 0.1.8 (site-libs2), einops (site-tl440-nn082), sae-lens
export PYTHONPATH="$PWD/src:$PWD/packages/mdb/src:$S/site-libs2:$S/site-tl440-nn082:$S/site-saelens"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME=/mnt/ssd1tb/cache/beast/huggingface TMPDIR=$S/tmp WANDB_DISABLED=true
export MDB_MODEL="${MDB_MODEL:-/mnt/big/llm-models/Qwen3-1.7B-Base}"
export MDB_SAE="${MDB_SAE:-/mnt/big/llm-models/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50}"
export MDB_NPAIRS="${MDB_NPAIRS:-34}" MDB_EXAMPLE_DIR="$PWD"
echo "== revision $(cat REVISION 2>/dev/null || echo unknown)  npairs $MDB_NPAIRS"
/usr/bin/time -v timeout "${BUDGET:-570}" python interchange_subspace.py 2> time.err
RC=$?
echo "SCRIPT_RC $RC"
grep -E "Maximum resident|Elapsed" time.err; grep -v -E "^\s|Loading weights|Warning|warn" time.err | tail -15 || true
exit $RC
