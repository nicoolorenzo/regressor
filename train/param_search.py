from functools import singledispatch

import keras
import numpy as np
import optuna
from optuna.trial import TrialState
from sklearn.base import clone

from src.training import create_Keras_neural_network
from train.loss import truncated_medae_scorer


def suggest_params(trial):
    params = {
        'number_of_hidden_layers': trial.suggest_int('number_of_hidden_layers', 2, 7),
        'dropout_between_layers': trial.suggest_float('dropout_between_layers', 0, 0.5),
        'number_of_neurons_per_layer': trial.suggest_categorical('number_of_neurons_per_layer', [512, 1024, 2048, 4096]),
        'epochs': trial.suggest_int('epochs', 10, 100),
        'activation': trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'gelu', 'swish']),
        'lr': trial.suggest_float('lr', 10**(-5), 10**(-2), log=True),
        'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32, 64])
    }
    return params


def create_objective(X, y, cv):
    def objective(trial):
        params = suggest_params(trial)
        estimator = create_Keras_neural_network(
            n_features=X.shape[1],
            number_of_hidden_layers=params["number_of_hidden_layers"],
            neurons_per_layer=params["neurons_per_layer"],
            activation=params["activation"],
            dropout_rate=params["dropout_between_layers"]
        )
        scoring = truncated_medae_scorer
        cross_val_scores = []
        for step, (train_index, test_index) in enumerate(cv.split(X, y)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            estimator.compile(
                optimizer=keras.optimizers.Adam(learning_rate=params["lr"]),
                loss=keras.losses.MeanAbsoluteError(),
                metrics=[
                    keras.metrics.MeanSquaredError(),
                    keras.metrics.MeanRelativeError(),
                ],
            )
            estimator.fit(
                x=X_train,
                y=y_train,
                batch_size=params["batch_size"],
                epochs=params["epochs"],
                verbose="auto"
            )
            # TODO: poner breakpoint y extraer de la lista siguiente, MAE, guardalo como score
            test_metrics = estimator.evalute(X_test, y_test)
            # FIXME: score = test_metrics["meas"]
            cross_val_scores.append(score)
            intermediate_value = np.mean(cross_val_scores)
            trial.report(intermediate_value, step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return np.mean(cross_val_scores)

    return objective



def param_search(estimator, X, y, cv, study, n_trials, keep_going=False):
    objective = create_objective(estimator, X, y, cv)
    trials = [trial for trial in study.get_trials() if trial.state in [TrialState.COMPLETE, TrialState.PRUNED]]
    if not keep_going:
        n_trials = n_trials - len(trials)
    if n_trials > 0:
        print(f"Starting {n_trials} trials")
        study.optimize(objective, n_trials=n_trials)

    return load_best_params(estimator, study)


def create_study(model_name, study_prefix, storage):
    return optuna.create_study(
        study_name=f'{study_prefix}-{model_name}',
        direction='minimize' if model_name == 'lgb' else 'maximize',
        storage=storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner()
    )



def load_best_params(estimator, study):
    try:
        return study.best_params
    except Exception as e:
        print(f'Study for {type(estimator)} does not exist')
        raise e

