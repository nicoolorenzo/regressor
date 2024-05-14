import tensorflow.keras as keras
from keras.layers import Dense, Dropout
from keras.models import Sequential


def create_dnn(n_features, optuna_params):
    neurons_per_layer = optuna_params["neurons_per_layer"]
    number_of_hidden_layers = optuna_params["number_of_hidden_layers"]
    activation = optuna_params["activation"]
    dropout_rate = optuna_params["dropout_between_layers"]

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


def suggest_params(trial):
    params = {
        'number_of_hidden_layers': trial.suggest_categorical('number_of_hidden_layers', [2, 6, 10]),
        'dropout_between_layers': trial.suggest_float('dropout_between_layers', 0, 0.5),
        'neurons_per_layer': trial.suggest_categorical('neurons_per_layer', [512, 1024, 2048, 4096]),
       # 'neurons_per_layer': trial.suggest_categorical('neurons_per_layer', [10, 20, 50, 100]),
        'epochs': trial.suggest_categorical('epochs', [1]),
        'activation': trial.suggest_categorical('activation', ['relu']),
        'lr': trial.suggest_float('lr', 10**(-6), 10**(-3), log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16])
    }
    return params


def fit_dnn(dnn, X, y, optuna_params):
    dnn.compile(
        optimizer=keras.optimizers.Adam(learning_rate=optuna_params["lr"]),
        loss=keras.losses.MeanAbsoluteError(),
        metrics=[
            keras.metrics.MeanSquaredError(),
            keras.metrics.MeanAbsolutePercentageError()
        ],
    )
    dnn.fit(
        x=X,
        y=y,
        batch_size=optuna_params["batch_size"],
        epochs=optuna_params["epochs"],
        verbose=0
    )
    return dnn
