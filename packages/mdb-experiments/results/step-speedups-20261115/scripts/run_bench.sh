#!/bin/bash
# usage: run_bench.sh TREE OUT [case ...]   cases: "row 28 1024 4" etc.
S=/private/tmp/claude-501/-Users-jakeholl-domains/f9daae27-47d0-4118-8855-2c7a6f58eecf/scratchpad
TREE=$1; OUT=$2; shift 2
cd $TREE/packages/mdb
export PYTHONPATH=$PWD/src:$PWD/../../src
: > $OUT
for c in "$@"; do $S/libvenv/bin/python $S/speed/bench.py $c >> $OUT 2>&1; done
cat $OUT
