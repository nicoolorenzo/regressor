import tensorflow.keras as keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping

def create_dnnNot(n_features, optuna_params):
    neurons_per_layer = optuna_params["neurons_per_layer"]
    number_of_hidden_layers = optuna_params["number_of_hidden_layers"]
    activation = optuna_params["activation"]
    dropout_rate = optuna_params["dropout_between_layers"]

    layers = []
    # Input layer
    layers.append(Dense(neurons_per_layer, input_dim=n_features))
    # Intermediate hidden layers
    for _ in range(1, number_of_hidden_layers+1):
        layers.append(Dense(neurons_per_layer, activation=activation))
        layers.append(Dropout(dropout_rate))
    # Output layer
    layers.append(Dense(1))

    return Sequential(layers)


def suggest_params(trial):
    params = {
        'number_of_hidden_layers': trial.suggest_categorical('number_of_hidden_layers', [10, 14, 18]),
        'dropout_between_layers': trial.suggest_float('dropout_between_layers', 0, 0.5),
        'neurons_per_layer': trial.suggest_categorical('neurons_per_layer', [512, 1024, 2048]),
       # 'neurons_per_layer': trial.suggest_categorical('neurons_per_layer', [10, 20, 50, 100]),
        'epochs': trial.suggest_categorical('epochs', [1]),
        'activation': trial.suggest_categorical('activation', ['relu']),
        'lr': trial.suggest_float('lr', 10**(-6), 10**(-4), log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16])
    }
    return params


def fit_dnnNot(dnn, X, y, optuna_params):
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

def create_dnn(n_features, optuna_params):
    activation1 = keras.activations.swish #optuna_params["activation1"]
    input_deep = keras.layers.Input(shape=(n_features,))
    layer_previous = Dense(2500, activation=activation1)(input_deep)

    for _ in range(1,3):
        new_layer=Dense(2500, activation=activation1)(layer_previous)
        layer_previous=new_layer

    for _ in range(1, 8):
        new_layer=Dense(2500, activation=activation1)(layer_previous)
        layer_previous=new_layer

    layers_deep = layer_previous

    layer_1 = Dense(2500, activation=activation1)(input_deep)
    layer_2 = Dense(2500, activation=activation1)(layer_1)
    layers_wide = layer_2

    concat = keras.layers.concatenate([layers_deep, layers_wide])
    layer_previous = Dense(80, activation=activation1)(concat)

    for _ in range(1, 3):
        new_layer=Dense(80, activation=activation1)(layer_previous)
        layer_previous=new_layer

    layers_deep_and_wide_small = layer_previous
    output = Dense(1)(layers_deep_and_wide_small)
    model = keras.Model(inputs=[input_deep], outputs=[output])

    return model


stop_here_please = EarlyStopping(patience=5)

def fit_dnn(dnn, X, y, optuna_params):
    dnn.compile(
        optimizer=keras.optimizers.Adam(learning_rate=9*10 ** (-6)),
        loss=keras.losses.MeanAbsoluteError(),
        metrics=[
            keras.metrics.MeanSquaredError(),
            keras.metrics.MeanAbsolutePercentageError()
        ],
    )
    dnn.fit(
        x=X,
        y=y,
        batch_size=16,
        epochs=50,
        verbose=1,
        validation_split=0.1
        ,callbacks=[stop_here_please]
    )
    return dnn