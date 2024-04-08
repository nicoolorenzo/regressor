import tensorflow.keras as keras
import numpy as np
import optuna
from keras import Sequential
from keras.layers import Dense, Dropout


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
        # Neural network architecture
        # Input layer
        layers = [Dense(params["number_of_neurons_per_layer"], input_dim=X.shape[1])]
        # Intermediate hidden layers
        for _ in range(1, params["number_of_hidden_layers"]):
            layers.append(Dense(params["number_of_neurons_per_layer"], activation=params["activation"]))
            Dropout(params["dropout_between_layers"])
        # Output layer
        layers.append(Dense(1))

        # The estimator is a neural network
        estimator = Sequential(layers)

        #FIXME: scoring no se usa, se puede borrar o hay que usarlo después?
        # Lo de detrás del igual lo he puesto yo, si se usa hay que refactorizar las y's
        # scoring = median_absolute_error(y_true[:len(y_pred)], y_pred)

        cross_val_scores = []
        # FIXME: no se que le pasa, pero cv no tiene splits, he intentado cambiar los 'repeats', pero nada
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
            # cuando funcione habrá que ver exactamente cómo devuelve las métricas evaluate para añadirlas bien a score
            test_metrics = estimator.evalute(X_test, y_test)
            score = test_metrics["mae", "medae", "mape"]

            cross_val_scores.append(score)
            intermediate_value = np.mean(cross_val_scores)
            trial.report(intermediate_value, step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return np.mean(cross_val_scores)

    return objective
