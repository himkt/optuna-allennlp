import functools
import os
import random
import shutil

from allennlp.data import Vocabulary, allennlp_collate
from allennlp.data.dataset_readers import TextClassificationJsonReader
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.token_indexers import TokenCharactersIndexer
from allennlp.data.tokenizers import WhitespaceTokenizer
from allennlp.models import BasicClassifier
from allennlp.modules import Embedding
from allennlp.modules.seq2vec_encoders import CnnEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import TokenCharactersEncoder
from allennlp.modules.seq2vec_encoders import LstmSeq2VecEncoder
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
        token_indexers={
            "tokens": SingleIdTokenIndexer(),
            "token_characters": TokenCharactersIndexer(min_padding_length=1),
        },
        tokenizer=WhitespaceTokenizer(),
    )
    train_dataset = reader.read(
        "https://s3-us-west-2.amazonaws.com/allennlp/datasets/imdb/train.jsonl"
    )  # NOQA
    valid_dataset = reader.read(
        "https://s3-us-west-2.amazonaws.com/allennlp/datasets/imdb/dev.jsonl"
    )  # NOQA
    vocab = Vocabulary.from_instances(train_dataset)
    train_dataset.index_with(vocab)
    valid_dataset.index_with(vocab)
    return train_dataset, valid_dataset, vocab


def create_model(
    vocab: Vocabulary,
    trial: Trial,
):
    embedding_dim = trial.suggest_int("embedding_dim", 16, 128)
    character_embedding_dim = trial.suggest_int("character_embedding_dim", 16, 64)
    max_filter_size = trial.suggest_int("max_filter_size", 3, 6)
    num_filters = trial.suggest_int("num_filters", 16, 128)
    output_dim = trial.suggest_int("output_dim", 64, 128)
    lstm_hidden_size = trial.suggest_int("lstm_hidden_size", 16, 128)
    dropout = trial.suggest_float("dropout", 0, 1.0)

    embedder = BasicTextFieldEmbedder(
        {
            "tokens": Embedding(
                embedding_dim=embedding_dim, trainable=True, vocab=vocab
            ),
            "token_characters": TokenCharactersEncoder(
                embedding=Embedding(
                    embedding_dim=character_embedding_dim, trainable=True, vocab=vocab
                ),
                encoder=CnnEncoder(
                    embedding_dim=character_embedding_dim,
                    num_filters=num_filters,
                    ngram_filter_sizes=range(2, max_filter_size),
                    output_dim=output_dim,
                ),
            ),
        }
    )
    encoder_input_size = embedding_dim + output_dim
    encoder = LstmSeq2VecEncoder(
        input_size=encoder_input_size,
        hidden_size=lstm_hidden_size,
        bidirectional=True,
    )
    model = BasicClassifier(
        text_field_embedder=embedder,
        seq2vec_encoder=encoder,
        dropout=dropout,
        vocab=vocab,
    )
    return model


def objective_fn(trial, device: int, target_metric: str, base_serialization_dir: str):
    train_dataset, valid_dataset, vocab = prepare_data()
    model = create_model(vocab, trial)

    if device > -1:
        model.to(torch.device("cuda:{}".format(device)))

    lr = trial.suggest_float("lr", 1e-3, 1e-1, log=True)
    optimizer = SGD(model.parameters(), lr=lr)
    data_loader = DataLoader(train_dataset, batch_size=32, collate_fn=allennlp_collate)
    validation_data_loader = DataLoader(
        valid_dataset, batch_size=64, collate_fn=allennlp_collate
    )
    serialization_dir = os.path.join(
        base_serialization_dir, "trial_{}".format(trial.number)
    )
    trainer = GradientDescentTrainer(
        model=model,
        optimizer=optimizer,
        data_loader=data_loader,
        validation_data_loader=validation_data_loader,
        validation_metric="+" + target_metric,  # TODO (himkt): use direction
        patience=None,  # `patience=None` since it could conflict with AllenNLPPruningCallback
        num_epochs=100,
        cuda_device=device,
        serialization_dir=serialization_dir,
        epoch_callbacks=[AllenNLPPruningCallback(trial, "validation_" + target_metric)],
    )
    vocab.save_to_files(os.path.join(serialization_dir, "vocabulary"))
    result = trainer.train()
    return result[f"best_validation_{target_metric}"]


@click.command()
@click.option("--device", type=int, default=-1)
@click.option("--target_metric", type=str, default="accuracy")
@click.option("--base_serialization_dir", type=str, default="result/202007251030_tmp11")
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
    if pkg_resources.parse_version(allennlp.__version__) < pkg_resources.parse_version(
        "1.0.0"
    ):
        raise RuntimeError("AllenNLP>=1.0.0 is required for this example.")

    random.seed(41)
    torch.manual_seed(41)
    numpy.random.seed(41)

    main()
