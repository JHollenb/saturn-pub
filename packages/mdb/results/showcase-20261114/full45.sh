#!/usr/bin/env bash
O=/mnt/ssd1tb/cache/beast/showcase/d
mkdir -p $O
BUDGET=300 bash run.sh exp4_brackets.py $O
BUDGET=240 bash run.sh exp5_neurons.py $O
