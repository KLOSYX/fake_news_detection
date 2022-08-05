#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

python src/train.py -m experiment=street_classify_exp task_name=street_eda_effectiveness ++logger.wandb.project=street_eda_effectiveness datamodule.eda_prob=0,1 seed=1,2,3
sleep 60
python src/train.py -m experiment=district_classify_exp task_name=district_eda_effectiveness ++logger.wandb.project=district_eda_effectiveness datamodule.eda_prob=0,1 seed=1,2,3
sleep 60
python src/train.py -m experiment=street_classify_exp task_name=street_mlm_effectiveness ++logger.wandb.project=street_eda_effectiveness bert_name_or_path=hfl/chinese-roberta-wwm-ext,/data/teamwork/multimodal/pretrained_models/chinese-roberta-gov-affairs-wwm,/data/teamwork/multimodal/pretrained_models/chinese-roberta-gov-affairs seed=1,2,3
sleep 60
python src/train.py -m experiment=district_classify_exp task_name=district_mlm_effectiveness ++logger.wandb.project=district_eda_effectiveness bert_name_or_path=hfl/chinese-roberta-wwm-ext,/data/teamwork/multimodal/pretrained_models/chinese-roberta-gov-affairs-wwm,/data/teamwork/multimodal/pretrained_models/chinese-roberta-gov-affairs seed=1,2,3
