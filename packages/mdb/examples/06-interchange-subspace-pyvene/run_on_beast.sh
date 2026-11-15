#!/usr/bin/env bash
# Submit the interchange example to our GPU host through the job scheduler (mrun) as one job (about 4 minutes, priority 70).
# Needs the mrun tunnel (MRUN_URL=http://127.0.0.1:19025):
#   ssh -f -N -L 127.0.0.1:19025:127.0.0.1:9025 beast
# Check first that the GPU is free: ~/domains/common/bin/common mrun jobs --state running
# Smoke: MDB_NPAIRS=2 bash run_on_beast.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MDB="$(cd "$HERE/../.." && pwd)"            # packages/mdb
ROOT="$(cd "$MDB/../.." && pwd)"            # repo root (holds src/ = saturn_pub)
export MRUN_URL="${MRUN_URL:-http://127.0.0.1:19025}"
NPAIRS="${MDB_NPAIRS:-34}"

# Stage a payload: this example (script, data, prereg) + the mdb source + saturn_pub.
STAGE="$(mktemp -d "${TMPDIR:-/tmp}/mdb-example-interchange.XXXXXX")"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/packages/mdb"
cp "$HERE/run.sh" "$HERE/interchange_subspace.py" "$HERE/items.json" "$HERE/reference.json" \
   "$HERE/prereg.json" "$STAGE/"
rsync -a --exclude __pycache__ --exclude static "$MDB/src" "$STAGE/packages/mdb/"
rsync -a --exclude __pycache__ "$ROOT/src" "$STAGE/"
git -C "$ROOT" rev-parse HEAD > "$STAGE/REVISION" 2>/dev/null || echo unknown > "$STAGE/REVISION"

cd "$STAGE"
# --vram and --ram are in MB. Measured peak 3866 MB VRAM, 3.5 GB RSS (smoke run);
# the asks are the measured peak x1.3 (VRAM) and RSS with headroom.
~/domains/common/bin/mrun submit -e mdb-example-interchange --payload . --pin beast --gpu \
  --vram "${VRAM:-5100}" --ram "${RAM:-6000}" --priority 70 --env-alias mrun --detach \
  -- env MDB_NPAIRS="$NPAIRS" bash run.sh
