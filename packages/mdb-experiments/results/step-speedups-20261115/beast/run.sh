#!/usr/bin/env bash
# mrun payload entry: the stepping bench and example 04 (smoke) on main and on the branch,
# one process per tree, then a byte comparison of their outputs. MODE=smoke|full.
set -uo pipefail
MODE="${MODE:-full}"
S=/mnt/ssd1tb/cache/beast/mdb-interop
SITE="$S/site-libs2:$S/site-tl440-nn082:$S/site-saelens"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME=/mnt/ssd1tb/cache/beast/huggingface TMPDIR=$S/tmp WANDB_DISABLED=true
cat REVISIONS
python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, torch.cuda.get_device_name())"
for tree in main branch; do
  echo "== bench $tree"
  PYTHONPATH="$PWD/$tree/src:$PWD/$tree/packages/mdb/src:$SITE" /usr/bin/time -v \
    timeout 240 python beast_step.py $tree $([ "$MODE" = smoke ] && echo smoke) > bench_$tree.out 2> bench_$tree.err
  echo "rc $?"; cat bench_$tree.out; grep -E "Maximum resident|Elapsed" bench_$tree.err; grep -E "Error|Traceback" -A3 bench_$tree.err | tail -20
done
if [ "$MODE" != smoke ]; then
  for tree in main branch; do
    echo "== example 04 smoke $tree"
    rm -rf ex04_$tree && cp -r ex04 ex04_$tree
    ( cd ex04_$tree && PYTHONPATH="$PWD/../$tree/src:$PWD/../$tree/packages/mdb/src:$SITE" \
      MDB_MODEL=/mnt/big/llm-models/Qwen3-1.7B-Base MDB_SAE=/mnt/big/llm-models/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50 \
      MDB_SMOKE=1 MDB_OUT=out /usr/bin/time -v timeout 150 python watch_edit_continue.py > run.out 2> time.err
      echo "rc $?"; grep -E "RESULT_JSON|done" run.out | cut -c1-400; grep -E "Maximum resident|Elapsed" time.err )
  done
fi
python compare.py
