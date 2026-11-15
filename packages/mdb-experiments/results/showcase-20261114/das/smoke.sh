#!/usr/bin/env bash
# DAS smoke: admit a subset (30 train items + the first 4 test pairs' items), train k=4 for 60 steps,
# score whole/das/symtab/meandiff/random/sae on those pairs.  OUT=.../showcase-das-smoke
set -uo pipefail
TR=$(python3 -c "print(','.join(map(str,range(0,130,4))))")
TE=$(python3 -c "print(','.join(map(str,range(130,138))))")
BUDGET=200 bash run.sh das.py admit $OUT $TR,$TE
ADMIT_TAG=-smoke BUDGET=200 bash run.sh das.py train $OUT 4 60
ADMIT_TAG=-smoke BUDGET=150 bash run.sh das.py score $OUT whole,das,symtab,symcap,meandiff,random,sae 0,1 smoke
