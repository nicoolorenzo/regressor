import pickle
from functools import singledispatch

import numpy as np
import optuna
from gpytorch.utils.errors import NotPSDError
from optuna.trial import TrialState
from sklearn.base import clone
from sklearn.linear_model import Ridge
from xgboost import XGBClassifier
from xgboost import XGBRegressor

from models.ensemble.Blender import Blender
from models.nn.SkDnn import SkDnn
from train.loss import truncated_medae_scorer


@singledispatch
def suggest_params(estimator, trial):
    raise NotImplementedError



@suggest_params.register
def _(estimator: XGBRegressor, trial):
    return _suggest_xgboost(trial)




@suggest_params.register
def _(estimator: XGBClassifier, trial):
    return _suggest_xgboost(trial)


"""
@suggest_params.register
def _(estimator: SkDnn, trial):
    h1 = trial.suggest_categorical('hidden_1', [512, 1024, 1512, 2048, 4096])
    T0 = trial.suggest_int('T0', 10, 100)
    params = {
        'hidden_1': h1,
        'hidden_2': trial.suggest_int('hidden_2', 32, 512),
        'dropout_1': trial.suggest_float('dropout_1', 0.3, 0.7),
        'dropout_2': trial.suggest_float('dropout_2', 0.0, 0.2),
        'activation': trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'gelu', 'swish']),
        'lr': trial.suggest_float('lr', 1e-4, 1e-3),
        'T0': T0,
        'annealing_rounds': trial.suggest_int('annealing_rounds', 2, 5),
        'swa_epochs': trial.suggest_int('swa_epochs', 5, T0),
        'var_p': trial.suggest_float('var_p', 0.9, 1.0)
    }
    return params
"""


@suggest_params.register
def _(estimator: SkDnn, trial):
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


@suggest_params.register
def _(estimator: Ridge, trial):
    return {'alpha': trial.suggest_float('alpha', 0, 20)}


def create_objective(estimator, X, y, cv):
    def estimator_factory():
        return clone(estimator)

    def objective(trial):
        estimator = estimator_factory()
        params = suggest_params(estimator, trial)
        estimator.set_params(**params)
        scoring = truncated_medae_scorer
        try:
            score = cross_val_score_with_pruning(estimator, X, y, cv=cv, scoring=scoring, trial=trial)
        except NotPSDError:
            print('NotPSDError while cross-validating')
            score = -np.inf
        return score

    return objective


def cross_val_score_with_pruning(estimator, X, y, cv, scoring, trial):
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


@singledispatch
def final_estimator_study_name(estimator):
    raise NotImplementedError

@final_estimator_study_name.register
def _(estimator: Ridge):
    return 'ridge'


@singledispatch
def param_search(estimator, X, y, cv, study, n_trials, keep_going=False):
    objective = create_objective(estimator, X, y, cv)
    trials = [trial for trial in study.get_trials() if trial.state in [TrialState.COMPLETE, TrialState.PRUNED]]
    if not keep_going:
        n_trials = n_trials - len(trials)
    if n_trials > 0:
        print(f"Starting {n_trials} trials")
        study.optimize(objective, n_trials=n_trials)

    return load_best_params(estimator, study)


@param_search.register
def _(estimator: Blender, X, y, cv, study, n_trials, keep_going=False):
    # For the blender, the study is expected to consist of a duple: (storage, study_prefix)
    storage, study_prefix = study
    X_train, X_test, y_train, y_test = estimator._blending_split(X, y)

    models_with_studies = []
    for model_name, model in estimator.estimators:
            model = clone(model)
            study = create_study(model_name, study_prefix, storage)
            models_with_studies.append((model_name, model, study))
            _ = param_search(model, X_train, y_train, cv,
                             study, n_trials, keep_going=keep_going)

    # FIXME
    raise ValueError("Stopping to prevent blender to train")
    estimator.estimators = [
        (n, set_best_params(clone(model), study)) for n, model, study in models_with_studies
    ]

    # Train with best parameters and predict to create the dataset for the blender
    blended_X = []
    fitted_estimators = []
    for model_name, model, study in models_with_studies:
        # train_with_best clones first
        if study is None:
            model = model.fit(X_train, y_train)
        else:
            model = train_with_best_params(model, X_train, y_train, study)
        fitted_estimators.append((model_name, model))
        blended_X.append(model.predict(X_test).reshape(-1, 1))

    blended_dataset = {
        'X': np.concatenate(blended_X, axis=1),
        'y': y_test
    }
    final_estimator = clone(estimator.final_estimator)
    study = create_study(final_estimator_study_name(final_estimator), study_prefix, storage)

    _ = param_search(final_estimator, blended_dataset['X'], blended_dataset['y'],
                     cv, study, n_trials, keep_going=False)
    estimator.final_estimator = set_best_params(clone(estimator.final_estimator), study)
    return estimator


def create_study(model_name, study_prefix, storage):
    return optuna.create_study(
        study_name=f'{study_prefix}-{model_name}',
        direction='minimize' if model_name == 'lgb' else 'maximize',
        storage=storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner()
    )


@singledispatch
def set_best_params(estimator, study):
    if study is not None:
        best_params = load_best_params(estimator, study)
        estimator.set_params(**best_params)
    return estimator


@set_best_params.register
def _(estimator: Blender, study):
    # Again (see param_search), for the blender the study is expected to consist of a duple (storage, study_prefix)
    storage, study_prefix = study
    estimator.estimators = [
        (n, set_best_params(clone(model), create_study(n, study_prefix, storage))) for (n, model) in estimator.estimators
    ]
    estimator.final_estimator = set_best_params(
        clone(estimator.final_estimator),
        create_study(final_estimator_study_name(estimator.final_estimator), study_prefix, storage)
    )
    return estimator


@singledispatch
def train_with_best_params(estimator, X, y, study):
    estimator = clone(estimator)
    best_params = load_best_params(estimator, study)
    estimator.set_params(**best_params)
    return estimator.fit(X, y)


@singledispatch
def load_best_params(estimator, study):
    try:
        return study.best_params
    except Exception as e:
        print(f'Study for {type(estimator)} does not exist')
        raise e

