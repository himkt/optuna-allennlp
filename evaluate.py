import os.path

import click
import torch
from allennlp.data import Vocabulary, allennlp_collate
from allennlp.data.dataset_readers import TextClassificationJsonReader
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import WhitespaceTokenizer
from allennlp.training.util import evaluate
from optuna.study import load_study
from torch.utils.data import DataLoader

from optuna_train_custom_trainer import create_model


@click.command()
@click.option("--device", type=int, default=-1)
@click.option("--base_serialization_dir", type=str)
def main(device, base_serialization_dir):
    storage = "sqlite:///" + os.path.join(base_serialization_dir, "optuna.db")
    study = load_study("optuna_allennlp", storage)
    best_trial = study.best_trial
    print(f"best_trial: {best_trial.number}")

    reader = TextClassificationJsonReader(
        token_indexers={"tokens": SingleIdTokenIndexer()},
        tokenizer=WhitespaceTokenizer(),
    )
    serialization_dir = os.path.join(base_serialization_dir, f"trial_{best_trial.number}")
    vocab = Vocabulary.from_files(os.path.join(serialization_dir, "vocabulary"))
    data = reader.read("https://s3-us-west-2.amazonaws.com/allennlp/datasets/imdb/test.jsonl")
    data.index_with(vocab)

    hyperparams = best_trial.params
    hyperparams.pop("lr")
    model = create_model(vocab=vocab, **hyperparams)
    model.load_state_dict(torch.load(os.path.join(serialization_dir, "best.th")))

    if device >= 0:
        model.to(device)
    data_loader = DataLoader(data, batch_size=64, collate_fn=allennlp_collate)
    print(evaluate(model, data_loader, cuda_device=device))


if __name__ == "__main__":
    main()
