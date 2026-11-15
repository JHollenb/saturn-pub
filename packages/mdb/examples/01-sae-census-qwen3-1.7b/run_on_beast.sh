#!/usr/bin/env bash
# Submit the SAE census to our GPU host through the job scheduler (mrun) as one job (about 8 minutes, priority 70).
# Needs the mrun tunnel (MRUN_URL=http://127.0.0.1:19025):
#   ssh -f -N -L 127.0.0.1:19025:127.0.0.1:9025 beast
# Check first that the GPU is free: ~/domains/common/bin/common mrun jobs --state running
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MDB="$(cd "$HERE/../.." && pwd)"            # packages/mdb
ROOT="$(cd "$MDB/../.." && pwd)"            # repo root (holds src/ = saturn_pub)
export MRUN_URL="${MRUN_URL:-http://127.0.0.1:19025}"

# Stage a payload: this example + the mdb source + saturn_pub, no web build or tests.
STAGE="$(mktemp -d "${TMPDIR:-/tmp}/mdb-example-sae.XXXXXX")"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/packages/mdb"
cp "$HERE/run.sh" "$HERE/sae_census.py" "$STAGE/"
rsync -a --exclude __pycache__ --exclude static "$MDB/src" "$STAGE/packages/mdb/"
rsync -a --exclude __pycache__ "$ROOT/src" "$STAGE/"
git -C "$ROOT" rev-parse HEAD > "$STAGE/REVISION" 2>/dev/null || echo unknown > "$STAGE/REVISION"

cd "$STAGE"
# --vram and --ram are in MB. Measured peak was 3.7 GB VRAM (bf16 1.7B model) and about 8 GB host
# RAM with the SAE cache at 6 layers; the asks leave headroom.
~/domains/common/bin/mrun submit -e mdb-example-sae --payload . --pin beast --gpu \
  --vram 6000 --ram 12000 --priority 70 --env-alias mrun --detach -- bash run.sh
