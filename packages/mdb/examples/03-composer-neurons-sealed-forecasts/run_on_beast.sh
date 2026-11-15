#!/usr/bin/env bash
# Submit the composer-neuron example to our GPU host through the job scheduler (mrun) as one job (about 3.5 minutes, priority 70).
# Needs the mrun tunnel (MRUN_URL=http://127.0.0.1:19025):
#   ssh -f -N -L 127.0.0.1:19025:127.0.0.1:9025 beast
# Check first that the GPU is free: ~/domains/common/bin/common mrun jobs --state running
# Smoke: MDB_COUNTRIES=Spain,Japan bash run_on_beast.sh   (4 held-out items; pools still on all 64)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MDB="$(cd "$HERE/../.." && pwd)"            # packages/mdb
ROOT="$(cd "$MDB/../.." && pwd)"            # repo root (holds src/ = saturn_pub)
export MRUN_URL="${MRUN_URL:-http://127.0.0.1:19025}"

# Stage a payload: this example + the mdb source + saturn_pub, no web build or tests.
STAGE="$(mktemp -d "${TMPDIR:-/tmp}/mdb-example-composer.XXXXXX")"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/packages/mdb"
cp "$HERE/run.sh" "$HERE/composer_neurons.py" "$HERE/items.json" "$HERE/reference.json" \
   "$HERE/prereg.json" "$STAGE/"
rsync -a --exclude __pycache__ --exclude static "$MDB/src" "$STAGE/packages/mdb/"
rsync -a --exclude __pycache__ "$ROOT/src" "$STAGE/"
git -C "$ROOT" rev-parse HEAD > "$STAGE/REVISION" 2>/dev/null || echo unknown > "$STAGE/REVISION"
if [ -n "${MDB_COUNTRIES:-}" ]; then
  sed -i.bak "s/^export MDB_COUNTRIES=.*/export MDB_COUNTRIES=\"$MDB_COUNTRIES\"/" "$STAGE/run.sh"
  rm -f "$STAGE/run.sh.bak"
fi

cd "$STAGE"
# --vram and --ram are in MB. Measured peak: 6890 MB VRAM, 2.4 GB RSS;
# --vram = peak x 1.3. mrun clamps --ram to its measured-history grant.
~/domains/common/bin/mrun submit -e mdb-example-composer --payload . --pin beast --gpu \
  --vram "${MDB_VRAM:-9000}" --ram 12000 --priority 70 --env-alias mrun --detach -- bash run.sh
