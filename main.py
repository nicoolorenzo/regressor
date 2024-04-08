import os

from sklearn.model_selection import StratifiedKFold

from src import preprocessing, training
from utils.data_loading import get_my_data
from utils.data_saving import save_dnn
from utils.evaluation import evaluate_model
from utils.stratification import stratify_y

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

        # TODO: cuando ya funcione todo, dejar puesto: ["fingerprints", "descriptors", "all"]
        for features in ["descriptors"]:  # "descriptors", "all"]
            # Preprocess X
            preprocessed_train_split_X, preprocessed_test_split_X, preproc = preprocessing.preprocess_X(
                 descriptors_columns=descriptors_columns,
                 fingerprints_columns=fingerprints_columns,
                 train_X=train_split_X,
                 train_y=train_split_y,
                 test_X=test_split_X,
                 test_y=test_split_y,
                 features=features
            )

            preprocessed_train_split_y, preprocessed_test_split_y, preproc_y = preprocessing.preprocess_y(
                train_split_y, test_split_y
            )

            print("Param search")
            trained_dnn = training.optimize_and_train_dnn(preprocessed_train_split_X, train_split_y,
                                                          param_search_folds, number_of_trials, fold, features)

            print("Saving dnn used for this fold")
            save_dnn(trained_dnn, fold, features)

            print("Evaluation of the model & saving of the results")
            evaluate_model(trained_dnn, preprocessed_test_split_X, preprocessed_test_split_y, preproc_y, fold, features)

            # This fold is done, add 1 to the variable fold to keep the count of the number of folds
            fold = fold + 1
