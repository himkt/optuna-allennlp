# Optuna using AllenNLP

Demonstration for using Optuna with AllenNLP integration.


# Quick Start

```bash
# create virtual environment
python3 -m venv venv
. venv/bin/activate

# install libraries
pip install -r requirements.txt

# run hyperparameter optimization
python optuna_train.py
```


# Attention

Demonstration uses GPU.
If you want to run the scripts in this repository,
please update `cuda_device = -1` in [allennlp config](https://github.com/himkt/optuna-allennlp/blob/master/config/imdb.0.jsonnet#L3) and [optuna_config](https://github.com/himkt/optuna-allennlp/blob/master/config/imdb.1.jsonnet#L3).


# Result

<img width="896" alt="experimental_result" src="https://user-images.githubusercontent.com/5164000/80803806-5a332500-8bee-11ea-862c-9db27e7091ba.png">

