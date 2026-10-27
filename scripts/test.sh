#!/bin/bash
# Four drafts per problem of the OpenTO, NITO and OpenTO-NL test sets, their corrections, and the scores of both, with
# the weights in results/generator-e4/weights.pt and results/corrector/weights.pt. The FE solves of the scores run on
# CPU workers with one thread each, as for the paper.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
R=${LMTO_RESULTS:-results}
for s in test NITO_test NLtest; do
  python -m lmto.sample --split $s
  python -m lmto.correct --split $s
  for m in generator-e4 corrector; do
    OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= python -m lmto.evaluate $R/$m/${s}_k4.npz
  done
done
