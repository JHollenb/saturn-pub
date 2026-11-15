#!/usr/bin/env bash
# generic runner: env SCRIPT, ARGS (space separated), BUDGET (s)
set -uo pipefail
export PYTHONPATH="$PWD/src:$PWD/packages/mdb/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=${OMP:-8}
echo "== revision $(cat REVISION) script $SCRIPT args $ARGS"
/usr/bin/time -v timeout ${BUDGET:-580} python $SCRIPT $ARGS 2> time.err
echo "DRIVER_RC $?"
grep -E "Maximum resident|Elapsed" time.err; grep -A12 Traceback time.err | tail -14; true
