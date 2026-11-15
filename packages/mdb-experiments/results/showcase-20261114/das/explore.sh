#!/usr/bin/env bash
# Exploratory (prereg_explore.json): in-distribution DAS on the test countries' items, exp7 pairs excluded.
set -uo pipefail
INDIST=1 BUDGET=560 bash run.sh das.py train $OUT 4,8,16,32,64 300
