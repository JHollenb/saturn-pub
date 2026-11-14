#!/usr/bin/env bash
# submit.sh NAME VRAM_MB RAM_MB TIMEOUT_S -- SCRIPT ARGS...   (env OUT, BUDGET passed through run.sh)
set -euo pipefail
NAME=$1 VRAM=$2 RAM=$3 TMO=$4; shift 5
W=/Users/jakeholl/domains/.worktrees/mdb-showcase
SC=$W/packages/mdb/results/showcase-20261114
TW=$W/packages/mdb/results/twohop-20261113/next
P=/private/tmp/claude-501/-Users-jakeholl-domains/f9daae27-47d0-4118-8855-2c7a6f58eecf/scratchpad/payload-$NAME-$(date +%H%M%S)
mkdir -p $P
git -C $W archive HEAD src packages/mdb/src | tar -x -C $P
git -C $W rev-parse --short HEAD > $P/REVISION
cp $SC/common.py $SC/run.sh $SC/smokes.sh $TW/r1_items.json $TW/b1_sets.json $TW/b1-score.json $P/
for d in exp1 exp2 exp3 exp4 exp5 exp6 exp7; do cp $SC/$d/*.py $SC/$d/*.json $P/ 2>/dev/null || true; done
cp $SC/data/* $P/ 2>/dev/null || true
cd $P
export MRUN_URL=http://127.0.0.1:19025
# shared GPU: wait until nothing runs on Beast (coordinator rule)
until [ -z "$(~/domains/common/bin/mrun jobs --state running 2>/dev/null | grep -i beast)" ] && [ -z "$(~/domains/common/bin/mrun jobs --state queued 2>/dev/null | grep -i beast)" ]; do sleep 20; done
~/domains/common/bin/mrun submit -e showcase-$NAME --payload $P --pin beast --gpu --vram $VRAM --ram $RAM --priority 70 --timeout $TMO --env-alias mrun --note "mdb showcase $NAME" --detach -- env OUT=${OUT:-/mnt/ssd1tb/cache/beast/showcase} BUDGET=${BUDGET:-570} bash ${RUNNER:-run.sh} "$@"
