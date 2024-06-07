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
number_of_folds = 1
number_of_trials = 2
utilize_chromatography_column = True
keep_all_chromatographic_columns_in_preprocessing = True
split_train_test_by_experiment = True


if __name__ == "__main__":
    # Load data
    print("Loading data")
    X, y, descriptors_columns, fingerprints_columns = get_my_data()

    if not chromatography_column:
        experiment_data, X, desc_cols, fgp_cols = delete_chromatography_columns(X)
    experiment = ["all"]

    if experiment != ["all"]:
        experiment_key_values = [(key, experiment_data.get(key)) for key in experiment]
    else:
        experiment_key_values = [(key, value) for key, value in experiment_data.items()]
    for key, values in experiment_key_values:
        if not utilize_chromatography_column and not is_smrt:
            X_ex = X[values[0]:values[1]]
            y_ex = y[values[0]:values[1]]
        else:
            X_ex = X
            y_ex = y
        # Create results directory if it doesn't exist
        if not os.path.exists('./results'):
            os.makedirs('./results')

        # Do K number of folds for cross validation and save the splits into a variable called splits
        # splitting_function = StratifiedKFold(n_splits=number_of_folds, shuffle=True, random_state=42)

        # Generate the splits dynamically and train with all the splits
        # for fold, (train_indexes, test_indexes) in enumerate(splitting_function.split(X_ex, stratify_y(y_ex))):
            # Use the indexes to actually split the dataset in training and test set.
            # train_split_X = X_ex[train_indexes]
            # train_split_y = y_ex[train_indexes]
            # test_split_X = X_ex[test_indexes]
            # test_split_y = y_ex[test_indexes]
        fold = 0
        if split_train_test_by_experiment:
            X_ex["id"] = X_ex["id"].str[0:4]
            unique_experiments = X_ex["id"].unique()
            train_experiments = random.sample(list(unique_experiments), k=int(len(unique_experiments) * 0.8))
            train_data = X_ex[X_ex['id'].isin(train_experiments)]
            test_data = X_ex[~X_ex['id'].isin(train_experiments)]
            train_split_X = train_data.drop(columns=common_columns, axis=1).astype('float32')
            test_split_X = test_data.drop(columns=common_columns, axis=1).astype('float32')
            train_split_y = np.array(train_data["rt"]).astype('float32').flatten()
            test_split_y = np.array(test_data["rt"]).astype('float32').flatten()

        else:
            X = X.drop(columns=common_columns, axis=1).astype('float32')
            y = np.array(y).astype('float32').flatten()
            train_split_X, test_split_X, train_split_y, test_split_y = train_test_split(X_ex, y_ex, test_size=0.2,
                                                                                        random_state=42)
        if not keep_info_chromatography_processing and utilize_chromatography_column:
            columns_not_processed_train = train_split_X.loc[:, :"flow_rate 17"]
            columns_not_processed_test = test_split_X.loc[:, :"flow_rate 17"]
            descriptors_columns = np.arange(descriptors_columns.shape[0]-columns_not_processed_train.shape[1], dtype="int")
            fingerprints_columns = np.arange(descriptors_columns.shape[0], descriptors_columns.shape[0] +
                                             fingerprints_columns.shape[0], dtype="int")
            train_split_X = train_split_X.loc[:, "MW":]
            test_split_X = test_split_X.loc[:, "MW":]
        features_list = ["fingerprints"] if is_smoke_test else ["descriptors", "fingerprints", "all"]
        for features in features_list:
            # Preprocess X
            if not keep_info_chromatography_processing and utilize_chromatography_column and features != "fingerprints":
                (preprocessed_train_split_X, preprocessed_test_split_X, preprocessed_chromatography_train,
                 preprocessed_chromatography_test) = preprocessing.preprocess_X_chromatography(
                     descriptors_columns=descriptors_columns,
                     fingerprints_columns=fingerprints_columns,
                     train_X=train_split_X,
                     train_y=train_split_y,
                     test_X=test_split_X,
                     test_y=test_split_y,
                     chromatography_train = columns_not_processed_train,
                     chromatography_test = columns_not_processed_test,
                     features=features
                )
                preprocessed_train_split_X = pd.concat([preprocessed_chromatography_train, preprocessed_train_split_X], axis=1)
                preprocessed_test_split_X = pd.concat([preprocessed_chromatography_test, preprocessed_test_split_X], axis=1)

            elif keep_info_chromatography_processing and utilize_chromatography_column:
                preprocessed_train_split_X, preprocessed_test_split_X = preprocessing.preprocess_X_usp(
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
            trained_dnn = training.optimize_and_train_dnn(preprocessed_train_split_X, preprocessed_train_split_y,
                                                          preprocessed_test_split_X, preprocessed_test_split_y,
                                                          param_search_folds, number_of_trials, fold, features)

            print("Saving dnn used for this fold")
            trained_dnn.save(f"./results/dnn-{fold}-{features}.keras")

            print("Evaluation of the model & saving of the results")
            evaluate_model(trained_dnn, preprocessed_test_split_X, preprocessed_test_split_y, preproc_y, fold,
                           features, key)
