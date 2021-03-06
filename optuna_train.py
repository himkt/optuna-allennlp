"""

An example of hyperparameter optimization using AllenNLPExecutor.
It requires sqlite3 to run this example.

"""

import optuna


def objective(trial: optuna.Trial) -> float:
    trial.suggest_int("embedding_dim", 64, 256)
    trial.suggest_int("max_filter_size", 2, 5)
    trial.suggest_int("num_filters", 64, 256)
    trial.suggest_int("output_dim", 64, 256)
    trial.suggest_float("dropout", 0.0, 0.5)
    trial.suggest_float("lr", 5e-3, 5e-1, log=True)

    executor = optuna.integration.allennlp.AllenNLPExecutor(
        trial=trial,  # trial object
        config_file="./config/imdb_optuna.jsonnet",  # jsonnet path
        serialization_dir=f"./result/optuna/{trial.number}",  # directory for snapshots and logs
        metrics="best_validation_accuracy"
    )
    return executor.run()


if __name__ == '__main__':
    study = optuna.create_study(
        storage="sqlite:///result/trial.db",  # save results in DB
        sampler=optuna.samplers.TPESampler(seed=24),
        study_name="optuna_allennlp",
        direction="maximize",
        pruner=optuna.pruners.HyperbandPruner(),
    )

    timeout = 60 * 60 * 10  # timeout (sec): 60*60*10 sec => 10 hours
    study.optimize(
        objective,
        n_jobs=1,  # number of processes in parallel execution
        n_trials=30,  # number of trials to train a model
        timeout=timeout,  # threshold for executing time (sec)
    )

    optuna.integration.allennlp.dump_best_config("./config/imdb_optuna.jsonnet", "best_imdb_optuna.json", study)
