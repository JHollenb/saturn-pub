#!/usr/bin/env bash
# Submit the component-binding example to our GPU host through the job scheduler (mrun) as one job (about 7 minutes, priority 70).
# Needs the mrun tunnel (MRUN_URL=http://127.0.0.1:19025):
#   ssh -f -N -L 127.0.0.1:19025:127.0.0.1:9025 beast
# Check first that the GPU is free: ~/domains/common/bin/common mrun jobs --state running
# MDB_NITEMS=2 bash run_on_beast.sh   runs a 2-item smoke.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MDB="$(cd "$HERE/../.." && pwd)"            # packages/mdb
ROOT="$(cd "$MDB/../.." && pwd)"            # repo root (holds src/ = saturn_pub)
export MRUN_URL="${MRUN_URL:-http://127.0.0.1:19025}"
NITEMS="${MDB_NITEMS:-24}"

# Stage a payload: this example (script, data, prereg) + the mdb source + saturn_pub.
STAGE="$(mktemp -d "${TMPDIR:-/tmp}/mdb-example-components.XXXXXX")"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/packages/mdb"
cp "$HERE/run.sh" "$HERE/sae_component_binding.py" "$HERE/items.json" "$HERE/reference.json" \
   "$HERE/prereg.json" "$STAGE/"
rsync -a --exclude __pycache__ --exclude static "$MDB/src" "$STAGE/packages/mdb/"
rsync -a --exclude __pycache__ "$ROOT/src" "$STAGE/"
git -C "$ROOT" rev-parse HEAD > "$STAGE/REVISION" 2>/dev/null || echo unknown > "$STAGE/REVISION"

cd "$STAGE"
# --vram and --ram are in MB: the smoke run peaked at 3976 MB VRAM and 10.9 GB host RAM
# (the SAE cache holds 14 layers); the asks are about the peaks x1.3.
~/domains/common/bin/mrun submit -e mdb-example-components --payload . --pin beast --gpu \
  --vram 5200 --ram 14200 --priority 70 --env-alias mrun --detach -- env MDB_NITEMS="$NITEMS" bash run.sh
