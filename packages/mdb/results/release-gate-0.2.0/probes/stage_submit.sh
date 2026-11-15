#!/usr/bin/env bash
# Release-gate 0.2.0: stage an example payload exactly as its run_on_beast.sh does (example files,
# packages/mdb/src without static, repo src, REVISION = HEAD) plus probes/, then submit one job
# with a custom command. Usage: stage_submit.sh <example-dir> <job-name> <vram MB> <ram MB> -- cmd...
set -euo pipefail
EX="$(cd "$1" && pwd)"; NAME="$2"; VRAM="$3"; RAM="$4"; shift 4; [ "$1" = "--" ] && shift
PROBES="$(cd "$(dirname "$0")" && pwd)"
MDB="$(cd "$EX/../.." && pwd)"
ROOT="$(cd "$MDB/../.." && pwd)"
export MRUN_URL="${MRUN_URL:-http://127.0.0.1:19025}"
STAGE="$(mktemp -d "${TMPDIR:-/tmp}/mdb-rg.XXXXXX")"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/packages/mdb" "$STAGE/probes"
find "$EX" -maxdepth 1 -type f ! -name '*.ipynb' ! -name '*.log' ! -name 'run_on_beast.sh' -exec cp {} "$STAGE/" \;
cp "$PROBES"/*.py "$STAGE/probes/"
rsync -a --exclude __pycache__ --exclude static "$MDB/src" "$STAGE/packages/mdb/"
rsync -a --exclude __pycache__ "$ROOT/src" "$STAGE/"
git -C "$ROOT" rev-parse HEAD > "$STAGE/REVISION"
cd "$STAGE"
~/domains/common/bin/mrun submit -e "$NAME" --payload . --pin beast --gpu \
  --vram "$VRAM" --ram "$RAM" --priority 70 --env-alias mrun --detach -- "$@"
