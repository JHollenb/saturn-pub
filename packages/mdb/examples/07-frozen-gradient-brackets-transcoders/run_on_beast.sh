#!/usr/bin/env bash
# Submit the bracket example to Beast through mrun as one job (about 4 minutes, priority 70).
# Needs the mrun tunnel (MRUN_URL=http://127.0.0.1:19025):
#   ssh -f -N -L 127.0.0.1:19025:127.0.0.1:9025 beast
# Check first that the GPU is free: ~/domains/common/bin/common mrun jobs --state running
# Smoke run on two items:  MDB_ITEMS=0,1 ./run_on_beast.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MDB="$(cd "$HERE/../.." && pwd)"            # packages/mdb
ROOT="$(cd "$MDB/../.." && pwd)"            # repo root (holds src/ = saturn_pub)
export MRUN_URL="${MRUN_URL:-http://127.0.0.1:19025}"

# Stage a payload: this example + the mdb source + saturn_pub, no web build or tests.
STAGE="$(mktemp -d "${TMPDIR:-/tmp}/mdb-example-brackets.XXXXXX")"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/packages/mdb"
cp "$HERE/run.sh" "$HERE/frozen_gradient_brackets.py" "$HERE/items.json" "$HERE/reference.json" \
   "$HERE/prereg.json" "$STAGE/"
rsync -a --exclude __pycache__ --exclude static "$MDB/src" "$STAGE/packages/mdb/"
rsync -a --exclude __pycache__ "$ROOT/src" "$STAGE/"
git -C "$ROOT" rev-parse HEAD > "$STAGE/REVISION" 2>/dev/null || echo unknown > "$STAGE/REVISION"
if [ -n "${MDB_ITEMS:-}" ]; then sed -i.bak "s/^export MDB_ITEMS=.*/export MDB_ITEMS=$MDB_ITEMS/" "$STAGE/run.sh"; rm -f "$STAGE/run.sh.bak"; fi

cd "$STAGE"
# --vram and --ram are in MB. VRAM_MB / RAM_MB override; defaults come from the measured peaks
# (smoke job-2092d2852921: 3.9 GB VRAM, 8.5 GB host RAM with five 1.3 GB transcoder layers
# cached) x1.3 / with headroom.
~/domains/common/bin/mrun submit -e mdb-example-brackets --payload . --pin beast --gpu \
  --vram "${VRAM_MB:-5200}" --ram "${RAM_MB:-12000}" --priority 70 --env-alias mrun --detach -- bash run.sh
