from collections import namedtuple
import numpy as np
import os

from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import StratifiedKFold

from models.nn.SkDnn import SkDnn
from models.preprocessor.Preprocessors import Preprocessor
from train.param_search import create_study, param_search
from utils.data_loading import get_my_data
from utils.data_saving import save_preprocessor_and_dnn
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
    X, y, desc_cols, fgp_cols = get_my_data(common_cols=['unique_id', 'correct_ccs_avg'], is_smoke_test=is_smoke_test)

    # Create results directory if it doesn't exist
    if not os.path.exists('./results'):
        os.makedirs('./results')

    ParamSearchConfig = namedtuple('ParamSearchConfig', ['storage', 'study_prefix', 'param_search_cv', 'n_trials'])
    param_search_config = ParamSearchConfig(
        storage="sqlite:///./results/cv.db",
        study_prefix="dnn",
        param_search_cv=RepeatedKFold(n_splits=param_search_folds, n_repeats=1, random_state=42),
        n_trials=number_of_trials
    )

    DnnConfig = namedtuple('DnnConfig', ['train_size', 'n_strats', 'random_state'])
    dnn_config = DnnConfig(train_size=0.8, n_strats=8, random_state=42)

    cross_validation = StratifiedKFold(n_splits=number_of_folds, shuffle=True, random_state=42)

    for fold, (train_index, test_index) in enumerate(cross_validation.split(X, stratify_y(y))):
        fold = fold + 1  # Start counting the folds at fold=1

        # Split data differently for each fold
        param_search_config = param_search_config._replace(study_prefix=f"cv-fold-{fold}")
        train_split_X = X[train_index]
        train_split_y = y[train_index]
        test_split_X = X[test_index]
        test_split_y = y[test_index]

        # Preprocess X
        preprocessor = Preprocessor(
            desc_cols=desc_cols,
            fgp_cols=fgp_cols
        )
        preprocessed_train_split_X = preprocessor.fit_transform(train_split_X, train_split_y)

        # Preprocess y
        #TODO:something
        preprocessed_train_split_y = None  # PipelineWrapper()

        print("Creating DNN")
        all_cols = np.arange(preprocessed_train_split_X.shape[1])
        dnn = SkDnn(use_col_indices='all', binary_col_indices=all_cols[:-1], transform_output=True)
        """ OPTIONS:
        SkDnn(use_col_indices='all', binary_col_indices=binary_cols, transform_output=True)),
        SkDnn(use_col_indices=desc_cols, binary_col_indices=binary_cols, transform_output=True)),
        SkDnn(use_col_indices=fgp_cols, binary_col_indices=binary_cols, transform_output=True))
        """

        print("Param search")
        study = create_study("dnn", param_search_config.study_prefix, param_search_config.storage)
        best_params = param_search(
            dnn,
            preprocessed_train_split_X, train_split_y,
            cv=param_search_config.param_search_cv,
            study=study,
            n_trials=param_search_config.n_trials,
            keep_going=False
        )
        print("Training")
        dnn.set_params(**best_params)
        dnn.fit(preprocessed_train_split_X, train_split_y)

        save_preprocessor_and_dnn(preprocessor, dnn, fold)
        evaluate_model(dnn, preprocessor, test_split_X, test_split_y, fold)
