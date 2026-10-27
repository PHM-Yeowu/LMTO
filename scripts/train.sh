#!/bin/bash
# The final models of the paper: four epochs of the 0.5B generator over all 894,000 training problems, one draft of
# the fourth epoch per training problem, and the corrector trained on these drafts. Run after scripts/prepare_data.py;
# finished steps are skipped, and a stopped training resumes from its last checkpoint.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
R=${LMTO_RESULTS:-results}
for e in 1 2 3 4; do
  [ -f $R/generator-e$e/weights.pt ] || python -m lmto.train --config generator-e$e
done
[ -f $R/generator-e4/train_k1.pkl ] || python -m lmto.sample --split train --k 1
[ -f $R/corrector/weights.pt ] || python -m lmto.train --config corrector
