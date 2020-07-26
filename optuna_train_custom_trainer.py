import functools
import os
import random
import shutil

from allennlp.data import Vocabulary, allennlp_collate
from allennlp.data.dataset_readers import TextClassificationJsonReader
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import WhitespaceTokenizer
from allennlp.models import BasicClassifier
from allennlp.modules import Embedding
from allennlp.modules.seq2vec_encoders import CnnEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.training import GradientDescentTrainer
import allennlp
import click
import numpy
import optuna
import pkg_resources
import torch
from torch.optim import SGD
from torch.utils.data import DataLoader

from optuna.integration import AllenNLPPruningCallback
from optuna import Trial


def prepare_data():
    reader = TextClassificationJsonReader(
        token_indexers={"tokens": SingleIdTokenIndexer()},
        tokenizer=WhitespaceTokenizer(),
    )
    train_dataset = reader.read("https://s3-us-west-2.amazonaws.com/allennlp/datasets/imdb/train.jsonl")  # NOQA
    valid_dataset = reader.read("https://s3-us-west-2.amazonaws.com/allennlp/datasets/imdb/dev.jsonl")  # NOQA
    vocab = Vocabulary.from_instances(train_dataset)
    train_dataset.index_with(vocab)
    valid_dataset.index_with(vocab)
    return train_dataset, valid_dataset, vocab


def create_model(
        vocab: Vocabulary,
        embedding_dim: int,
        max_filter_size: int,
        num_filters: int,
        output_dim: int,
        dropout: float,
):
    model = BasicClassifier(
        text_field_embedder=BasicTextFieldEmbedder(
            {
                "tokens": Embedding(
                    embedding_dim=embedding_dim,
                    trainable=True,
                    vocab=vocab
                )
            }
        ),
        seq2vec_encoder=CnnEncoder(
            ngram_filter_sizes=range(2, max_filter_size),
            num_filters=num_filters,
            embedding_dim=embedding_dim,
            output_dim=output_dim,
        ),
        dropout=dropout,
        vocab=vocab,
    )
    return model


def objective_fn(
        trial: Trial,
        device: int,
        direction: str,
        target_metric: str,
        base_serialization_dir: str,
):
    embedding_dim = trial.suggest_int("embedding_dim", 128, 256)
    max_filter_size = trial.suggest_int("max_filter_size", 3, 6)
    num_filters = trial.suggest_int("num_filters", 128, 256)
    output_dim = trial.suggest_int("output_dim", 128, 512)
    dropout = trial.suggest_float("dropout", 0, 1.0, log=True)
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)

    train_dataset, valid_dataset, vocab = prepare_data()
    model = create_model(vocab, embedding_dim, max_filter_size, num_filters, output_dim, dropout)

    if device > -1:
        model.to(torch.device("cuda:{}".format(device)))

    optimizer = SGD(model.parameters(), lr=lr)
    data_loader = DataLoader(train_dataset, batch_size=10, collate_fn=allennlp_collate)
    validation_data_loader = DataLoader(valid_dataset, batch_size=64, collate_fn=allennlp_collate)
    serialization_dir = os.path.join(base_serialization_dir, "trial_{}".format(trial.number))
    trainer = GradientDescentTrainer(
        model=model,
        optimizer=optimizer,
        data_loader=data_loader,
        validation_data_loader=validation_data_loader,
        validation_metric=("+" if direction == "MAXIMIZE" else "-") + target_metric,
        patience=None,  # `patience=None` since it could conflict with AllenNLPPruningCallback
        num_epochs=50,
        cuda_device=device,
        serialization_dir=serialization_dir,
        epoch_callbacks=[AllenNLPPruningCallback(trial, f"validation_{target_metric}")],
    )
    vocab.save_to_files(os.path.join(serialization_dir, "vocabulary"))
    return trainer.train()[f"best_validation_{target_metric}"]


@click.command()
@click.option("--device", type=int, default=-1)
@click.option("--target_metric", type=str, default="accuracy")
@click.option("--base_serialization_dir", type=str, default="model")
@click.option("--clear-at-end", is_flag=True)
def main(device, target_metric, base_serialization_dir, clear_at_end):
    os.makedirs(base_serialization_dir, exist_ok=True)
    storage = "sqlite:///" + os.path.join(base_serialization_dir, "optuna.db")
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.HyperbandPruner(),
        storage=storage,  # save results in DB
        study_name="optuna_allennlp",
    )
    objective = functools.partial(
        objective_fn,
        device=device,
        direction=study.direction.name,
        target_metric=target_metric,
        base_serialization_dir=base_serialization_dir,
    )
    study.optimize(objective, n_trials=50)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    if clear_at_end:
        shutil.rmtree(base_serialization_dir)


if __name__ == "__main__":
    if pkg_resources.parse_version(allennlp.__version__) < pkg_resources.parse_version("1.0.0"):
        raise RuntimeError("AllenNLP>=1.0.0 is required for this example.")

    random.seed(41)
    torch.manual_seed(41)
    numpy.random.seed(41)

    main()
