from optuna.trial import TrialState
from sklearn.model_selection import RepeatedKFold

import tensorflow.keras as keras

import optuna

from keras.models import Sequential
from keras.layers import Dense, Dropout

from BlackBox.param_search import create_objective


def optimize_and_train_dnn(preprocessed_train_split_X, preprocessed_train_split_y, param_search_folds, number_of_trials,
                           fold, features):
    cv = RepeatedKFold(n_splits=param_search_folds, n_repeats=1, random_state=42),
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

    # Neural Network architecture
    # Input layer
    layers = [Dense(best_params["neurons_per_layer"], input_dim=preprocessed_train_split_X.shape[1])]
    # Intermediate hidden layers
    for _ in range(1, best_params["number_of_hidden_layers"]):
        layers.append(Dense(best_params["neurons_per_layer"], activation=best_params["activation"]))
        Dropout(best_params["dropout_between_layers"])
    # Output layer
    layers.append(Dense(1))

    estimator = Sequential(layers)

    estimator.compile(
        optimizer=keras.optimizers.Adam(learning_rate=best_params["lr"]),
        loss=keras.losses.MeanAbsoluteError(),
        metrics=[
            keras.metrics.MeanSquaredError(),
            keras.metrics.MeanRelativeError(),
        ],
    )
    estimator.fit(
        x=preprocessed_train_split_X,
        y=preprocessed_train_split_y,
        batch_size=best_params["batch_size"],
        epochs=best_params["epochs"],
        verbose="auto"
    )
    return estimator
