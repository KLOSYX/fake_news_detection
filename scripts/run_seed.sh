#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh
EXPERIMENT=run_multi_modal_test

for i in {1..10}; do
  echo "Running seed $i for experiment ${EXPERIMENT}"
  python src/train.py experiment=${EXPERIMENT} seed="$i" model.modal=multi trainer.devices=\'6,7\' trainer.min_epochs=15 trainer.max_epochs=20
  python src/train.py experiment=${EXPERIMENT} seed="$i" model.modal=text trainer.devices=\'6,7\' trainer.min_epochs=15 trainer.min_epochs=15
  python src/train.py experiment=${EXPERIMENT} seed="$i" model.modal=image trainer.devices=\'6,7\' trainer.min_epochs=15 trainer.min_epochs=15
done
