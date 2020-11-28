<img width="896" alt="experimental_result" src="https://user-images.githubusercontent.com/5164000/80803806-5a332500-8bee-11ea-862c-9db27e7091ba.png">


# Optuna using AllenNLP

Demonstration for using [Optuna](https://github.com/optuna/optuna) with [AllenNLP](https://github.com/allenai/allennlp) integration.


# Quick Start

## Google Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/himkt/optuna-allennlp/blob/master/notebook/Optuna_AllenNLP.ipynb)

## On your computer

```bash
# create virtual environment
python3 -m venv venv
. venv/bin/activate

# install libraries
pip install -r requirements.txt

# train a model using AllenNLP cli
allennlp train -s result/allennlp config/imdb_baseline.jsonnet

# run hyperparameter optimization
python optuna_train.py

# define-and-run style example
python optuna_train_custom_trainer.py --device 0 --target_metric accuracy --base_serialization_dir result
```

## [New!!] Use `allennlp-optuna`

You can use [`allennlp-optuna`](https://github.com/himkt/allennlp-optuna), an AllenNLP plugin for hyperparameter optimization.

```bash

# Installation
pip install allennlp-optuna

# You need to register allennlp-optuna to allennlp using .allennlp_plugins
# It is not required if .allennlp_plugins already exists on your working directory
echo 'allennlp_optuna' >> .allennlp_plugins

# optimization
allennlp tune config/imdb_optuna.jsonnet config/hparams.json --serialization-dir result
```


# Attention!

Demonstration uses GPU.
If you want to run the scripts in this repository,
please update `cuda_device = -1` in [allennlp config](https://github.com/himkt/optuna-allennlp/blob/master/config/imdb_baseline.jsonnet#L3) and [optuna_config](https://github.com/himkt/optuna-allennlp/blob/master/config/imdb_optuna.jsonnet#L3).


# Blog Articles

- Japanese: https://medium.com/p/41ad5e8b2d1a
- English: https://medium.com/p/54b4bfecd78b
