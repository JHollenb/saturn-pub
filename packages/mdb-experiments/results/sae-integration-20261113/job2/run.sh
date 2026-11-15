#!/usr/bin/env bash
set -uo pipefail
export PYTHONPATH="$PWD/src:$PWD/packages/mdb/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
M=/mnt/big/llm-models/Qwen3-1.7B-Base
SAE=/mnt/big/llm-models/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50
echo "== revision $(cat REVISION)"
python -c "import sys,torch,transformers; print(sys.version.split()[0], torch.__version__, transformers.__version__, torch.cuda.get_device_name(0))"
echo "== smoke: tiny SAE tests on this env"
timeout 120 python -m pytest -q -p no:cacheprovider packages/mdb/tests/test_sae.py 2>&1 | tail -3 || true
/usr/bin/time -v timeout 840 python driver.py $M $SAE job2 2> time.err
echo "DRIVER_RC $?"
grep -E "Maximum resident|Elapsed" time.err; grep -v -E "^\s|Loading weights|Warning|warn" time.err | tail -15
echo "== jsonl bytes $(wc -c < job2.jsonl)"
mkdir -p "$MRUN_ARTIFACTS_DIR" 2>/dev/null && cp job2.jsonl "$MRUN_ARTIFACTS_DIR"/ 2>/dev/null || true
gzip -c job2.jsonl | base64 -w0 > job2.jsonl.gz.b64; echo "JSONL_B64_BYTES $(wc -c < job2.jsonl.gz.b64)"
