#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh
EXPERIMENT=run_multi_modal_test

for i in {1..5}; do
    echo "Running seed $i for experiment ${EXPERIMENT}"
    python src/train.py experiment=${EXPERIMENT} seed=$i model.modal=multi datamodule.batch_size_per_gpu=16 trainer.devices=\'6,7\' trainer.min_epochs=20
    python src/train.py experiment=${EXPERIMENT} seed=$i model.modal=text datamodule.batch_size_per_gpu=16 trainer.devices=\'6,7\' trainer.min_epochs=20
    python src/train.py experiment=${EXPERIMENT} seed=$i model.modal=image datamodule.batch_size_per_gpu=16 trainer.devices=\'6,7\' trainer.min_epochs=10
done
