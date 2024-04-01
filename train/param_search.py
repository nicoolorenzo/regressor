from functools import singledispatch

import numpy as np
import optuna
from optuna.trial import TrialState
from sklearn.base import clone

from train.loss import truncated_medae_scorer


def suggest_params(SkDnn, trial):
    max_number_of_epochs = trial.suggest_int('max_number_of_epochs', 10, 100)
    params = {
        'number_of_hidden_layers': trial.suggest_int('number_of_hidden_layers', 2, 7),
        'dropout_between_layers': trial.suggest_float('dropout_between_layers', 0, 0.5),
        'number_of_neurons_per_layer': trial.suggest_categorical('number_of_neurons_per_layer', [512, 1024, 2048, 4096]),
        'max_number_of_epochs': max_number_of_epochs,
        'activation': trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'gelu', 'swish']),
        'lr': trial.suggest_float('lr', 10**(-5), 10**(-2), log=True),
        'annealing_rounds': trial.suggest_int('annealing_rounds', 2, 5),
        'swa_epochs': trial.suggest_int('swa_epochs', 5, max_number_of_epochs),
        'var_p': trial.suggest_float('var_p', 0.9, 1.0),
        'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32, 64])
    }
    return params


def create_objective(estimator, X, y, cv):
    def estimator_factory():
        return clone(estimator)

    def objective(trial):
        estimator = estimator_factory()
        params = suggest_params(estimator, trial)
        estimator.set_params(**params)
        scoring = truncated_medae_scorer
        cross_val_scores = []
        for step, (train_index, test_index) in enumerate(cv.split(X, y)):
            est = clone(estimator)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            est.fit(X_train, y_train)
            cross_val_scores.append(scoring(est, X_test, y_test))
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

