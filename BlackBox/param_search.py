import tensorflow.keras as keras
import numpy as np
import optuna

from src.dnn import create_dnn, fit_dnn


def suggest_params(trial):
    params = {
        'number_of_hidden_layers': trial.suggest_int('number_of_hidden_layers', 2, 7),
        'dropout_between_layers': trial.suggest_float('dropout_between_layers', 0, 0.5),
        'neurons_per_layer': trial.suggest_categorical('neurons_per_layer', [512, 1024, 2048, 4096]),
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
        estimator = create_dnn(X.shape[1], params)
        cross_val_scores = []
        for step, (train_index, test_index) in enumerate(cv.split(X, y)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            estimator = fit_dnn(estimator, X_train, y_train, params)
            test_metrics = estimator.evaluate(
                X_test, y_test, return_dict=True, verbose=0
            )
            # loss is MAE Score, use it as optuna metric
            score = test_metrics["loss"] 
            cross_val_scores.append(score)
            intermediate_value = np.mean(cross_val_scores)
            trial.report(intermediate_value, step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return np.mean(cross_val_scores)

    return objective
