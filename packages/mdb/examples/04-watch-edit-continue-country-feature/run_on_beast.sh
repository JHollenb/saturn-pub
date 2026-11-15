#!/usr/bin/env bash
# Submit the watch -> edit -> continue example to our GPU host through the job scheduler (mrun) as one job (about 5 minutes,
# priority 70). MDB_SMOKE=1 runs two items only.
# Needs the mrun tunnel (MRUN_URL=http://127.0.0.1:19025):
#   ssh -f -N -L 127.0.0.1:19025:127.0.0.1:9025 beast
# Check first that the GPU is free: ~/domains/common/bin/common mrun jobs --state running
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MDB="$(cd "$HERE/../.." && pwd)"            # packages/mdb
ROOT="$(cd "$MDB/../.." && pwd)"            # repo root (holds src/ = saturn_pub)
export MRUN_URL="${MRUN_URL:-http://127.0.0.1:19025}"
SMOKE="${MDB_SMOKE:-0}"

# Stage a payload: this example (script, data, prereg) + the mdb source + saturn_pub.
STAGE="$(mktemp -d "${TMPDIR:-/tmp}/mdb-example-watch.XXXXXX")"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/packages/mdb"
cp "$HERE/run.sh" "$HERE/watch_edit_continue.py" "$HERE/items.json" "$HERE/census.json" \
   "$HERE/prereg.json" "$HERE/reference.json" "$STAGE/"
rsync -a --exclude __pycache__ --exclude static "$MDB/src" "$STAGE/packages/mdb/"
rsync -a --exclude __pycache__ "$ROOT/src" "$STAGE/"
git -C "$ROOT" rev-parse HEAD > "$STAGE/REVISION" 2>/dev/null || echo unknown > "$STAGE/REVISION"
( cd "$STAGE" && shasum -a 256 watch_edit_continue.py packages/mdb/src/mdb/analysis/pools.py >> REVISION )

cd "$STAGE"
# --vram and --ram are in MB. The original runs peaked at 3.9 GB VRAM and 4.4 GB RSS.
~/domains/common/bin/mrun submit -e mdb-example-watch --payload . --pin beast --gpu \
  --vram "${VRAM_MB:-6000}" --ram "${RAM_MB:-12000}" --priority 70 --env-alias mrun --detach \
  -- env MDB_SMOKE="$SMOKE" bash run.sh
