#!/usr/bin/env bash
# Scored DAS training job: SC1 check (SAELens's own loader vs mdb, CPU, seconds), then DAS per k.
set -uo pipefail
BUDGET=60 bash run.sh sc1_check.py /mnt/big/llm-models/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50 19
BUDGET=560 bash run.sh das.py train $OUT 1,2,4,8,16,32 300
