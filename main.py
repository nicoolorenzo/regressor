from collections import namedtuple
import numpy as np
import os

from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import StratifiedKFold

from models.nn.SkDnn import SkDnn
from models.preprocessor.Preprocessors import Preprocessor
from src import preprocessing, training
from train.param_search import create_study, param_search
from utils.data_loading import get_my_data
from utils.data_saving import save_dnn
from utils.evaluation import evaluate_model
from utils.stratification import stratify_y

from src import optimize_parameters

# Parameters
is_smoke_test = True

if is_smoke_test:
    print("Running smoke test...")
    number_of_folds = 2
    number_of_trials = 1
    param_search_folds = 2
else:
    number_of_folds = 5
    number_of_trials = 15
    param_search_folds = 5


if __name__ == "__main__":
    # Load data
    print("Loading data")
    # For retention time use common_cols=['pid', 'rt']
    X, y, descriptors_columns, fingerprints_columns = get_my_data(common_cols=['unique_id', 'correct_ccs_avg'],
                                                                  is_smoke_test=is_smoke_test)

    # Create results directory if it doesn't exist
    if not os.path.exists('./results'):
        os.makedirs('./results')

    # Do K number of folds for cross validation and save the splits into a variable called splits
    splitting_function = StratifiedKFold(n_splits=number_of_folds, shuffle=True, random_state=42)
    # Generate the splits dynamically and train with all the splits
    fold = 1
    for train_indexes, test_indexes in splitting_function.split(X, stratify_y(y)):
        # Use the indexes to actually split the dataset in training and test set.
        train_split_X = X[train_indexes]
        train_split_y = y[train_indexes]
        test_split_X = X[test_indexes]
        test_split_y = y[test_indexes]

        for features in ["fingerprints"]:  #, "descriptors", "all"]
            # Preprocess X
            preprocessed_train_split_X, binary_cols = preprocessing.preprocess_X(descriptors_columns=descriptors_columns,
                                                                    fingerprints_columns=fingerprints_columns,
                                                                    train_split_X=train_split_X,
                                                                    train_split_y=train_split_y)

            # Preprocess y
            #TODO:something
            preprocessed_train_split_y = preprocessing.preprocess_y(descriptors_columns=descriptors_columns,
                                                                    fingerprints_columns=fingerprints_columns,
                                                                    train_split_X=train_split_X,
                                                                    train_split_y=train_split_y)  # PipelineWrapper()

            print("Creating DNN")
            dnn = training.create_dnn(features, fingerprints_columns, descriptors_columns, binary_cols)

            print("Param search")
            best_params = training.optimize_parameters(dnn, preprocessed_train_split_X, train_split_y)

            print("Training")
            dnn.set_params(**best_params)
            # TODO: send preprocessed y when ready
            dnn.fit(preprocessed_train_split_X, train_split_y)

            print("Saving dnn used for this fold")
            save_dnn(dnn, fold)

            print("Evaluation of the model & saving of the results")
            # TODO: send preprocessed y when ready
            evaluate_model(dnn, preprocessed_train_split_X, test_split_y, fold)

            # This fold is done, add 1 to the variable fold to keep the count of the number of folds
            fold = fold + 1
