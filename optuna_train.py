"""

An example of hyperparameter optimization using AllenNLPExecutor.
It requires sqlite3 to run this example.

"""

import optuna


config_file = "./config/imdb.1.jsonnet"
result_path = "result/optuna/trial_{}"


def objective(trial: optuna.Trial) -> float:
    if trial.number == 0:  # start from the same params as imdb.0.jsonnet
        trial.suggest_int("embedding_dim", 128, 128)
        trial.suggest_int("max_filter_size", 4, 4)
        trial.suggest_int("num_filters", 128, 128)
        trial.suggest_int("output_dim", 128, 128)
        trial.suggest_float("dropout", 0.2, 0.2)
        trial.suggest_float("lr", 1e-1, 1e-1)
    else:
        trial.suggest_int("embedding_dim", 32, 256)
        trial.suggest_int("max_filter_size", 2, 6)
        trial.suggest_int("num_filters", 32, 256)
        trial.suggest_int("output_dim", 32, 256)
        trial.suggest_float("dropout", 0.0, 0.8)
        trial.suggest_float("lr", 1e-1, 5e-1)

    executor = optuna.integration.allennlp.AllenNLPExecutor(
        trial,  # trial object
        config_file,  # jsonnet path
        result_path.format(trial.number),  # directory for snapshots and logs
    )
    return executor.run()


if __name__ == '__main__':
    study = optuna.create_study(
        storage="sqlite:///result/trial.db",  # save results in DB
        sampler=optuna.samplers.TPESampler(seed=24),
        study_name="optuna_allennlp",
        direction="maximize",
    )

    timeout = 60 * 60 * 10  # timeout (sec): 60*60*10 sec => 10 hours
    study.optimize(
        objective,
        n_jobs=1,  # number of processes in parallel execution
        n_trials=30,  # number of trials to train a model
        timeout=timeout,  # threshold for executing time (sec)
    )

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))