#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh
EXPERIMENT=run_multi_modal_test

for i in {1..5}; do
    echo "Running seed $i for experiment ${EXPERIMENT}"
    python src/train.py experiment=${EXPERIMENT} seed=$i modal.modal=multi
    python src/train.py experiment=${EXPERIMENT} seed=$i modal.modal=text
    python src/train.py experiment=${EXPERIMENT} seed=$i modal.modal=image
done
