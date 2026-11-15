#!/usr/bin/env bash
# submit.sh NAME MODEL VRAM_MB RAM_MB [GPU|CPU] -- "STEP" "STEP" ...
# Payload = git archive of HEAD (src, packages/mdb/src, this directory), so REVISION pins the driver.
set -euo pipefail
NAME=$1 MODEL=$2 VRAM=$3 RAM=$4 DEV=$5; shift 6
W=/Users/jakeholl/domains/.worktrees/mdb-amplitude
D=packages/mdb/results/amplitude-20261114
P=/private/tmp/claude-501/-Users-jakeholl-domains/f9daae27-47d0-4118-8855-2c7a6f58eecf/scratchpad/payload-amp-$NAME-$(date +%H%M%S)
if [ -n "$(git -C $W status --porcelain -- $D/amp.py $D/run.sh $D/prereg.json $D/data packages/mdb/src)" ]; then echo "uncommitted driver/library; commit first"; exit 3; fi
mkdir -p $P
git -C $W archive HEAD src packages/mdb/src $D/amp.py $D/run.sh $D/prereg.json $D/data | tar -x -C $P
git -C $W rev-parse --short HEAD > $P/REVISION
cp $P/$D/amp.py $P/$D/run.sh $P/$D/prereg.json $P/
cp -r $P/$D/data $P/data
SHA=$(shasum -a 256 $P/amp.py | cut -c1-16)
echo "payload $P rev $(cat $P/REVISION) amp.py sha256:$SHA"
cd $P
export MRUN_URL=http://127.0.0.1:19025
GPUFLAG="--gpu --vram $VRAM"; [ "$DEV" = CPU ] && GPUFLAG=""
~/domains/common/bin/mrun submit -e amp-$NAME --payload $P --pin beast $GPUFLAG --ram $RAM --priority 70 \
  --timeout ${TMO:-600} --env-alias mrun --note "mdb amplitude $NAME $MODEL rev $(cat REVISION) amp.py sha256:$SHA" --detach -- \
  env MODEL=$MODEL BUDGET=${BUDGET:-570} SMOKE=${SMOKE:-} PART=${PART:-C,F} OUTROOT=${OUTROOT:-/mnt/ssd1tb/cache/beast/amplitude} bash run.sh "$@"
