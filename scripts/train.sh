#!/bin/bash
# The final models of the paper: fifteen epochs of the 0.5B generator over all 894,000 training problems, one draft of
# the fifteenth epoch per training problem, and the corrector trained on these drafts. Run after scripts/prepare_data.py;
# finished steps are skipped, and a stopped training resumes from its last checkpoint. With GPUS="0 1 2", each training
# runs on these GPUs together (scripts/ddp.sh) with the steps of one GPU.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
R=${LMTO_RESULTS:-results}
train() {
  [ -f $R/$1/weights.pt ] && return
  if [ -n "${GPUS:-}" ]; then bash scripts/ddp.sh $1 "$GPUS"; else python -m lmto.train --config $1; fi
}
for e in $(seq 1 15); do
  train generator-e$e
done
[ -f $R/generator-e15/train_k1.pkl ] || python -m lmto.sample --split train --k 1
train corrector
