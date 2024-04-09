from optuna.trial import TrialState
from sklearn.model_selection import RepeatedKFold

import tensorflow.keras as keras

import optuna

from BlackBox.param_search import create_objective
from src.dnn import create_dnn, fit_dnn


def optimize_and_train_dnn(preprocessed_train_split_X, preprocessed_train_split_y, param_search_folds, number_of_trials,
                           fold, features):
    cv = RepeatedKFold(n_splits=param_search_folds, n_repeats=1, random_state=42)
    n_trials = number_of_trials
    keep_going = False

    study = optuna.create_study(study_name=f"foundation_cross_validation-fold-{fold}-{features}",
                                direction='minimize',
                                storage="sqlite:///./results/cv.db",
                                load_if_exists=True,
                                pruner=optuna.pruners.MedianPruner()
                                )

    objective = create_objective(preprocessed_train_split_X, preprocessed_train_split_y, cv)
    trials = [trial for trial in study.get_trials() if trial.state in [TrialState.COMPLETE, TrialState.PRUNED]]
    if not keep_going:
        n_trials = n_trials - len(trials)
    if n_trials > 0:
        print(f"Starting {n_trials} trials")
        study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    estimator = create_dnn(preprocessed_train_split_X.shape[1], best_params)
    estimator = fit_dnn(estimator,
                        preprocessed_train_split_X, 
                        preprocessed_train_split_y,
                        best_params)

    return estimator
