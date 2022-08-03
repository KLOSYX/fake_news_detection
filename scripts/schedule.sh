#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

python src/train.py -m local=only_img_test seed=1,2,3,4,5

python src/train.py -m experiment=street_classify_exp model.learning_rate=0.0001 datamodule.eda_prob=1 model.dropout_prob=0.1 datamodule.only_img=true model.num_warmup_steps=0 datamodule.batch_size_per_gpu=8 +trainer.log_every_n_steps=1 trainer.max_epochs=8 trainer.min_epochs=8 seed=1,2,3,4,5
