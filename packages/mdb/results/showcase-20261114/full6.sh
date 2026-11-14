#!/usr/bin/env bash
O=/mnt/ssd1tb/cache/beast/showcase/f
mkdir -p $O
BUDGET=560 bash run.sh exp6_tlnull.py "$@" $O
