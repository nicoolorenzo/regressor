from optuna.trial import TrialState
from sklearn.model_selection import RepeatedKFold

import numpy as np
import optuna

from src.models.dnn import suggest_params, create_dnn, fit_dnn


#def create_objective(X, y, cv):
def create_objective(X_tr, y_tr, X_te, y_te):
    def objective(trial):
        params = suggest_params(trial)
        estimator = create_dnn(X_tr.shape[1], params)
        cross_val_scores = []
        #for step, (train_index, test_index) in enumerate(cv.split(X, y)):
            # X_train, X_test = X[train_index], X[test_index]
            # y_train, y_test = y[train_index], y[test_index]
            # estimator = fit_dnn(estimator, X_train, y_train, params)
            # test_metrics = estimator.evaluate(
            #     X_test, y_test, return_dict=True, verbose=0
            # )
            # # loss is MAE Score, use it as optuna metric
            # score = test_metrics["loss"]
            # cross_val_scores.append(score)
            # intermediate_value = np.mean(cross_val_scores)
            # #trial.report(intermediate_value, step)
            # trial.report(intermediate_value)
            # if trial.should_prune():
            #     raise optuna.TrialPruned()
        X_train, X_test = X_tr, X_te
        y_train, y_test = y_tr, y_te
        estimator = fit_dnn(estimator, X_train, y_train, params)
        test_metrics = estimator.evaluate(
            X_test, y_test, return_dict=True, verbose=0
        )
        # loss is MAE Score, use it as optuna metric
        score = test_metrics["loss"]
        cross_val_scores.append(score)
        intermediate_value = np.mean(cross_val_scores)
        # trial.report(intermediate_value, step)
        trial.report(intermediate_value, 0)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return np.mean(cross_val_scores)
    return objective


# def optimize_and_train_dnn(preprocessed_train_split_X, preprocessed_train_split_y, param_search_folds, number_of_trials,
#                           fold, features):
def optimize_and_train_dnn(preprocessed_train_split_X, preprocessed_train_split_y, preprocessed_test_split_X,
                           preprocessed_test_split_y, param_search_folds, number_of_trials, fold, features):
    cv = RepeatedKFold(n_splits=param_search_folds, n_repeats=1, random_state=42)
    n_trials = number_of_trials
    keep_going = False

    study = optuna.create_study(study_name=f"foundation_cross_validation-fold-{fold}-{features}",
                                direction='minimize',
                                storage="sqlite:///./results/cv.db",
                                load_if_exists=True,
                                pruner=optuna.pruners.MedianPruner()
                                )

    # objective = create_objective(preprocessed_train_split_X, preprocessed_train_split_y, cv)
    objective = create_objective(preprocessed_train_split_X, preprocessed_train_split_y,
                                 preprocessed_test_split_X, preprocessed_test_split_y)
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






