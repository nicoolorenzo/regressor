import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from src import preprocessing, training
from utils.data_loading import get_my_data
from src.evaluation import evaluate_model
from utils.stratification import stratify_y
import random

# Parameters
showPlot = False
save_predictions = True
use_chromatography_column = False
keep_all_chromatographic_columns_in_preprocessing = False
split_train_test_by_experiment = False


if __name__ == "__main__":
    # Load data
    print("Loading data")
    X, y, usp_columns, chromatography_columns, descriptors_columns, fingerprints_columns = get_my_data()


    experiments = ['0001']

    if not use_chromatography_column:
        X = X.sort_values("id")
        number_columns = X["id"].str[0:4].drop_duplicates().values
        experiment_data = {}
        number_molecules = 0
        for value in number_columns:
            experiment = int(X[X["id"].str.startswith(value)].shape[0])
            experiment_data[value] = (number_molecules, number_molecules + experiment)
            number_molecules = number_molecules + experiment

    if experiments != ["all"] and not use_chromatography_column:
        experiment_key_values = [(key, experiment_data.get(key)) for key in experiments]
    elif experiments == ["all"] and not use_chromatography_column:
        experiment_key_values = [(key, value) for key, value in experiment_data.items()]
    else:
        experiment_key_values = [("all", 0)]

    for key, values in experiment_key_values:
        if not use_chromatography_column:
            X_ex = X[values[0]:values[1]]
            y_ex = y[values[0]:values[1]]
        else:
            X_ex = X
            y_ex = y
        # Create results directory if it doesn't exist
        if not os.path.exists('./results'):
            os.makedirs('./results')

        fold = 0
        if split_train_test_by_experiment:
            X_ex["id"] = X_ex["id"].str[0:4]
            unique_experiments = X_ex["id"].unique()
            train_experiments = random.sample(list(unique_experiments), k=int(len(unique_experiments) * 0.8))
            train_data = X_ex[X_ex['id'].isin(train_experiments)]
            test_data = X_ex[~X_ex['id'].isin(train_experiments)]
            train_split_X = train_data.drop(["id", "rt"], axis=1).astype('float32')
            test_split_X = test_data.drop(["id", "rt"], axis=1).astype('float32')
            train_split_y = np.array(train_data["rt"]).astype('float32').flatten()
            test_split_y = np.array(test_data["rt"]).astype('float32').flatten()

        else:
            X_ex = X_ex.drop(["id", "rt"], axis=1).astype('float32')
            y = np.array(y).astype('float32').flatten()
            train_split_X, test_split_X, train_split_y, test_split_y = train_test_split(X_ex, y_ex, test_size=0.2,
                                                                                        random_state=42)

        features_list = ["descriptors", "fingerprints", "all"]
        for features in features_list:
            # Preprocess X
            if not keep_all_chromatographic_columns_in_preprocessing and use_chromatography_column:
                (preprocessed_train_split_X, preprocessed_test_split_X) = preprocessing.preprocess_X_except_chromatography(
                     usp_columns=usp_columns,
                     chromatography_columns=chromatography_columns,
                     descriptors_columns=descriptors_columns,
                     fingerprints_columns=fingerprints_columns,
                     train_X=train_split_X,
                     train_y=train_split_y,
                     test_X=test_split_X,
                     test_y=test_split_y,
                     features=features
                )

            elif keep_all_chromatographic_columns_in_preprocessing and use_chromatography_column:
                preprocessed_train_split_X, preprocessed_test_split_X = preprocessing.preprocess_X_except_usp(
                    usp_columns=usp_columns,
                    chromatography_columns=chromatography_columns,
                    descriptors_columns=descriptors_columns,
                    fingerprints_columns=fingerprints_columns,
                    train_X=train_split_X,
                    train_y=train_split_y,
                    test_X=test_split_X,
                    test_y=test_split_y,
                    features=features
                )
            else:
                preprocessed_train_split_X, preprocessed_test_split_X = preprocessing.preprocess_X(
                    descriptors_columns=descriptors_columns,
                    fingerprints_columns=fingerprints_columns,
                    train_X=train_split_X,
                    train_y=train_split_y,
                    test_X=test_split_X,
                    test_y=test_split_y,
                    features=features
                )

            preprocessed_train_split_y, preprocessed_test_split_y, preproc_y = preprocessing.preprocess_y(
                train_y=train_split_y, test_y=test_split_y
            )
            columns_deleted_preprocessing = [column for column in train_split_X.columns if column not in preprocessed_train_split_X]

            print("Param search")
            trained_dnn = training.optimize_and_train_dnn(preprocessed_train_split_X, preprocessed_train_split_y, showPlot, features)

            if save_predictions:
                y_pred_test = trained_dnn.predict(preprocessed_test_split_X)
                prediction_test = pd.DataFrame({'real': preprocessed_test_split_y.flatten(), 'pred': y_pred_test.flatten()})
                prediction_test.to_csv(f"./results/rt_test_predictions-{features}-{key}.csv", sep=",")

            print("Saving dnn used for this fold")
            # trained_dnn.save(f"./results/dnn-{fold}-{key}-{features}.keras")

            print("Evaluation of the model & saving of the results")
            evaluate_model(trained_dnn, preprocessed_test_split_X, preprocessed_test_split_y, preproc_y, fold,
                           features, key)
