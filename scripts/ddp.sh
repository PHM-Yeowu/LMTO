#!/bin/bash
# One configuration of lmto.config on several GPUs: one lmto.train process per GPU computes its share of every batch,
# and the gradients are summed, so the steps, the order of the problems and the schedule are those of one GPU. Rank 0
# prints the log and writes the checkpoints and the weights; the other ranks write to results/<config>/rank<r>.log.
# usage: scripts/ddp.sh CONFIG "GPUS" [--n N]      (e.g. scripts/ddp.sh corrector "0 1 2")
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
R=${LMTO_RESULTS:-results}
C=$1 GPUS=($2)
shift 2
mkdir -p $R/$C
export MASTER_ADDR=127.0.0.1 MASTER_PORT=${MASTER_PORT:-$((29500 + RANDOM % 2000))} WORLD_SIZE=${#GPUS[@]}
pids=()
for r in "${!GPUS[@]}"; do
  if [ $r = 0 ]; then
    RANK=$r CUDA_VISIBLE_DEVICES=${GPUS[$r]} python -m lmto.train --config $C "$@" &
  else
    RANK=$r CUDA_VISIBLE_DEVICES=${GPUS[$r]} python -m lmto.train --config $C "$@" > $R/$C/rank$r.log 2>&1 &
  fi
  pids+=($!)
done
for p in "${pids[@]}"; do
  wait $p
done
