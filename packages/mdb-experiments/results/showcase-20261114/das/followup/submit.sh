#!/usr/bin/env bash
# submit.sh NAME VRAM_MB RAM_MB TIMEOUT_S -- ARGS... ; payload = git archive HEAD of this worktree (committed drivers only)
set -euo pipefail
NAME=$1 VRAM=$2 RAM=$3 TMO=$4; shift 5
W=/Users/jakeholl/domains/.worktrees/mdb-engine-refactor
SC=packages/mdb/results/showcase-20261114
P=/private/tmp/claude-501/-Users-jakeholl-domains/f9daae27-47d0-4118-8855-2c7a6f58eecf/scratchpad/payload-fu-$NAME-$(date +%H%M%S)
mkdir -p $P
if [ -n "$(git -C $W status --porcelain -- $SC/das/followup $SC/common.py)" ]; then echo "uncommitted follow-up files; commit first"; exit 3; fi
git -C $W archive HEAD src packages/mdb/src $SC/common.py $SC/run.sh $SC/das/followup $SC/exp1/census.json \
  packages/mdb/results/twohop-20261113/next/r1_items.json \
  packages/mdb/results/twohop-20261113/next/b1_sets.json packages/mdb/results/twohop-20261113/next/b1-score.json | tar -x -C $P
git -C $W rev-parse --short HEAD > $P/REVISION
cp $P/$SC/common.py $P/$SC/run.sh $P/$SC/das/followup/{fu.py,prereg.json,vocab.json} $P/$SC/exp1/census.json \
   $P/packages/mdb/results/twohop-20261113/next/{r1_items.json,b1_sets.json,b1-score.json} $P/
echo "payload $P rev $(cat $P/REVISION) fu.py sha256:$(shasum -a 256 $P/fu.py | cut -c1-16) prereg sha256:$(shasum -a 256 $P/prereg.json | cut -c1-16)"
cd $P
export MRUN_URL=http://127.0.0.1:19025
~/domains/common/bin/mrun submit -e fu-$NAME --payload $P --pin beast --gpu --vram $VRAM --ram $RAM --priority 70 \
  --timeout $TMO --env-alias mrun --note "mdb 11b follow-up $NAME rev $(cat REVISION) fu.py sha256:$(shasum -a 256 fu.py | cut -c1-16) prereg sha256:$(shasum -a 256 prereg.json | cut -c1-16)" --detach -- \
  env OUT=${OUT:-/mnt/ssd1tb/cache/beast/showcase-fu} BUDGET=${BUDGET:-570} bash run.sh "$@"
