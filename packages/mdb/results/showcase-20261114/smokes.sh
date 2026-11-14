#!/usr/bin/env bash
# one queue wait for the exp4/exp5/exp6 smokes (each a separate python process via run.sh)
O=/mnt/ssd1tb/cache/beast/showcase/smoke
BUDGET=200 bash run.sh exp4_brackets.py $O 0,9
BUDGET=200 bash run.sh exp5_neurons.py $O 20
BUDGET=200 bash run.sh exp6_tlnull.py /mnt/big/llm-models/Qwen3-0.6B-Base 20 6 $O 2
