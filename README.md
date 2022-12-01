<div align="center">

# Fake News Detection

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

## Description

A collection of PyTorch Lightning models for fake news detection.

## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/KLOSYX/fake_news_detection.git
cd fake_news_detection

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

Preprocess data

```bash
# download data
# twitter: https://github.com/MKLab-ITI/image-verification-corpus
# weibo: https://drive.google.com/file/d/14VQ7EWPiFeGzxp3XC2DeEHi-BEisDINn/view?usp=sharing
# extract the data to the data folder

# preprocess twitter data
python data/preprocess_twitter.py
# preprocess weibo data
python data/preprocess_weibo.py
```

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 datamodule.batch_size=64
```

## Available models

### SpotFake

```bash
# on weibo dataset
python src/train.py experiment=spotfake_weibo

# on twitter dataset
python src/train.py experiment=spotfake_twitter
```

> Reference:
>
> - [shiivangii/SpotFake](https://github.com/shiivangii/SpotFake)
> - [SpotFake: A Deep Learning Approach for Fake News Detection](http://arxiv.org/abs/2108.10509)

### BDANN

```bash
# on weibo dataset
python src/train.py experiment=bdann_weibo

# on twitter dataset
python src/train.py experiment=bdann_twitter
```

> Reference:
>
> - [xiaolan98/BDANN-IJCNN2020](https://github.com/xiaolan98/BDANN-IJCNN2020)
> - Zhang, T., Wang, D., Chen, H., Zeng, Z., Guo, W., Miao, C., & Cui, L. (n.d.). BDANN: BERT-Based Domain Adaptation Neural Network for Multi-Modal Fake News Detection. 9.
