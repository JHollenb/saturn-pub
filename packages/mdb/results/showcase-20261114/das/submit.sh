#!/usr/bin/env bash
# submit.sh NAME VRAM_MB RAM_MB TIMEOUT_S -- SCRIPT ARGS...
# Payload = git archive of HEAD (src, packages/mdb/src, the showcase dir incl. das/), so the job's
# REVISION pins the drivers too (audit item 4); the das.py sha256 goes into the job note.
set -euo pipefail
NAME=$1 VRAM=$2 RAM=$3 TMO=$4; shift 5
W=/Users/jakeholl/domains/.worktrees/mdb-sc-das
SC=packages/mdb/results/showcase-20261114
P=/private/tmp/claude-501/-Users-jakeholl-domains/f9daae27-47d0-4118-8855-2c7a6f58eecf/scratchpad/payload-das-$NAME-$(date +%H%M%S)
mkdir -p $P
if [ -n "$(git -C $W status --porcelain -- $SC/das/*.py $SC/common.py)" ]; then echo "uncommitted das driver; commit first"; exit 3; fi
git -C $W archive HEAD src packages/mdb/src $SC/common.py $SC/run.sh $SC/das $SC/exp1/census.json \
  packages/mdb/results/twohop-20261113/next/r1_items.json | tar -x -C $P
git -C $W rev-parse --short HEAD > $P/REVISION
cp $P/$SC/common.py $P/$SC/run.sh $P/$SC/das/*.py $P/$SC/das/*.sh $P/$SC/das/train_candidates.json $P/$SC/exp1/census.json \
   $P/packages/mdb/results/twohop-20261113/next/r1_items.json $P/
SHA=$(shasum -a 256 $P/das.py | cut -c1-16)
echo "payload $P rev $(cat $P/REVISION) das.py sha256:$SHA"
cd $P
export MRUN_URL=http://127.0.0.1:19025
~/domains/common/bin/mrun submit -e das-$NAME --payload $P --pin beast --gpu --vram $VRAM --ram $RAM --priority 70 \
  --timeout $TMO --env-alias mrun --note "mdb DAS $NAME rev $(cat REVISION) das.py sha256:$SHA" --detach -- \
  env OUT=${OUT:-/mnt/ssd1tb/cache/beast/showcase-das} BUDGET=${BUDGET:-570} bash ${RUNNER:-run.sh} "$@"
