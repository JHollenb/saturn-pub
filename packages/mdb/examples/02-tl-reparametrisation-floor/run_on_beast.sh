#!/usr/bin/env bash
# Submit the TL reparametrisation-floor example to our GPU host through the job scheduler (mrun) as one job (priority 70).
# Needs the mrun tunnel (MRUN_URL=http://127.0.0.1:19025):
#   ssh -f -N -L 127.0.0.1:19025:127.0.0.1:9025 beast
# Check first that the GPU is free: ~/domains/common/bin/common mrun jobs --state running
# Smoke: MDB_NITEMS=2 bash run_on_beast.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MDB="$(cd "$HERE/../.." && pwd)"            # packages/mdb
ROOT="$(cd "$MDB/../.." && pwd)"            # repo root (holds src/ = saturn_pub)
export MRUN_URL="${MRUN_URL:-http://127.0.0.1:19025}"
NITEMS="${MDB_NITEMS:-24}"
VRAM="${VRAM:-13500}"                       # MB; measured peak 11930 MB (smoke run);
                                            # x1.3 exceeds what the scheduler admits (ceiling = ask x 1.1 <= ~15100)

# Stage a payload: this example + the mdb source + saturn_pub, no web build or tests.
STAGE="$(mktemp -d "${TMPDIR:-/tmp}/mdb-example-tl.XXXXXX")"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/packages/mdb"
cp "$HERE/run.sh" "$HERE/tl_reparam_floor.py" "$HERE/items.json" "$HERE/reference.json" \
   "$HERE/prereg.json" "$STAGE/"
rsync -a --exclude __pycache__ --exclude static "$MDB/src" "$STAGE/packages/mdb/"
rsync -a --exclude __pycache__ "$ROOT/src" "$STAGE/"
git -C "$ROOT" rev-parse HEAD > "$STAGE/REVISION" 2>/dev/null || echo unknown > "$STAGE/REVISION"

cd "$STAGE"
# --vram and --ram are in MB. Host RAM: two fp32 copies of the model on the CPU per model.
~/domains/common/bin/mrun submit -e mdb-example-tl-floor --payload . --pin beast --gpu \
  --vram "$VRAM" --ram 20000 --priority 70 --env-alias mrun --detach -- \
  env MDB_NITEMS="$NITEMS" bash run.sh
