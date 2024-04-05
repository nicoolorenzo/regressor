from optuna.trial import TrialState
from sklearn.model_selection import RepeatedKFold

import optuna

from keras.models import Sequential
from keras.layers import Dense, Dropout

from train.param_search import create_objective


def create_Keras_neural_network(n_features, number_of_hidden_layers, dropout_rate, activation, neurons_per_layer):
        layers = []
        # Input layer
        layers.append(Dense(neurons_per_layer, input_dim=n_features))
        # Intermediate hidden layers
        for _ in range(1, number_of_hidden_layers):
            layers.append(Dense(neurons_per_layer, activation=activation))
            Dropout(dropout_rate)
        # Output layer
        layers.append(Dense(1))

        return Sequential(layers)


def optimize_and_train_dnn(preprocessed_train_split_X, preprocessed_train_split_y, param_search_folds, number_of_trials, fold, features):
    cv = RepeatedKFold(n_splits=param_search_folds, n_repeats=1, random_state=42),
    n_trials = number_of_trials
    keep_going = False

    study = optuna.create_study(
        study_name=f"foundation_cross_validation-fold-{fold}-{features}",
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

    # TODO: cargar mejores parametros, volver a crear la red y entrenarla con y devolver
    estimator = create_Keras_neural_network(
        n_features=X.shape[1],
        number_of_hidden_layers=best_params["number_of_hidden_layers"],
        neurons_per_layer=best_params["neurons_per_layer"],
        activation=best_params["activation"],
        dropout_rate=best_params["dropout_between_layers"]
    )
    estimator.compile(
        optimizer=keras.optimizers.Adam(learning_rate=best_params["lr"]),
        loss=keras.losses.MeanAbsoluteError(),
        metrics=[
            keras.metrics.MeanSquaredError(),
            keras.metrics.MeanRelativeError(),
        ],
    )
    estimator.fit(
        x=X_train, # ojo con estos
        y=y_train,
        batch_size=best_params["batch_size"],
        epochs=best_params["epochs"],
        verbose="auto"
    )
    return estimator

