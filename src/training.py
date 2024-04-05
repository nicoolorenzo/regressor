from models.nn.SkDnn import SkDnn


def create_dnn(features, fingerprints_columns, descriptors_columns, binary_columns):
    if features == "fingerprints":
        return SkDnn(use_col_indices=fingerprints_columns, binary_col_indices=binary_columns, transform_output=True)
    elif features == "descriptors":
        return SkDnn(use_col_indices=descriptors_columns, binary_col_indices=binary_columns, transform_output=True)
    else:
        return SkDnn(use_col_indices='all', binary_col_indices=binary_columns, transform_output=True)

def optimize_parameters(dnn, preprocessed_train_split_X, train_split_y):
    cv = RepeatedKFold(n_splits=param_search_folds, n_repeats=1, random_state=42),
    study = dnn
    n_trials = number_of_trials
    keep_going = False

    study = optuna.create_study(
        study_name=f"cross_validation-fold-{fold}",
        # TODO: elegir uno
        direction='minimize' if model_name == 'lgb' else 'maximize',
        storage="sqlite:///./results/cv.db",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner()
    )






    objective = create_objective(estimator, X, y, cv)
    trials = [trial for trial in study.get_trials() if trial.state in [TrialState.COMPLETE, TrialState.PRUNED]]
    if not keep_going:
        n_trials = n_trials - len(trials)
    if n_trials > 0:
        print(f"Starting {n_trials} trials")
        study.optimize(objective, n_trials=n_trials)

    return load_best_params(estimator, study)

