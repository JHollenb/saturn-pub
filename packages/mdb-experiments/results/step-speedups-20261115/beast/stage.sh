#!/usr/bin/env bash
# Stage the payload: main (tag mdb-v0.2.0) and the branch HEAD as committed, the bench, example 04.
# usage: stage.sh DEST
set -euo pipefail
DEST=$1; HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
mkdir -p "$DEST"
for pair in "main:mdb-v0.2.0" "branch:HEAD"; do
  tree=${pair%%:*}; rev=${pair#*:}
  mkdir -p "$DEST/$tree"
  git -C "$ROOT" archive "$rev" src packages/mdb/src | tar -x -C "$DEST/$tree"
  echo "$tree $(git -C "$ROOT" rev-parse "$rev^{commit}")" >> "$DEST/REVISIONS"
done
mkdir -p "$DEST/ex04"
git -C "$ROOT" archive HEAD packages/mdb/examples/04-watch-edit-continue-country-feature | tar -x -C "$DEST/ex04" --strip-components=4
cp "$HERE/run.sh" "$HERE/beast_step.py" "$HERE/compare.py" "$DEST/"
