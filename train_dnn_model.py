import numpy as np

from models.nn.SkDnn import SkDnn
from models.preprocessor.Preprocessors import FgpPreprocessor, DescriptorsPreprocessor, Preprocessor
from train.param_search import param_search
from train.param_search import create_study


def create_dnn(fgp_cols, binary_cols):
    return SkDnn(use_col_indices=fgp_cols, binary_col_indices=binary_cols, transform_output=True)


def tune_and_fit(X, y, desc_cols, fgp_cols, param_search_config, features):
    """
    features: should be one of "fingerprints", "descriptors" or "all"
    """
    print(f"Starting tune_and_fit with data with dim ({X.shape[0]},{X.shape[1]})")
    print("Preprocessing...")
    if features == "fingerprints":
        print("Training fingerprints")
        preprocessor = FgpPreprocessor(
            fgp_cols=fgp_cols
        )

    elif features == "descriptors":
        print("Training with descriptors")
        preprocessor = DescriptorsPreprocessor(
            desc_cols=desc_cols,
            adduct_cols=fgp_cols[-3:]
        )
    elif features == "all":
        print("Training with descriptors + fingerprints")
        preprocessor = Preprocessor(
            desc_cols=desc_cols,
            fgp_cols=fgp_cols
        )
    else:
        raise ValueError('features: should be one of "fingerprints", "descriptors" or "all"')

    X_train = preprocessor.fit_transform(X, y)

    print("Creating DNN")
    all_cols = np.arange(X_train.shape[1])
    dnn = SkDnn(use_col_indices=all_cols,
                binary_col_indices=preprocessor.transformed_binary_cols,
                transform_output=True)

    print("Param search")
    study = create_study("dnn", param_search_config.study_prefix, param_search_config.storage)
    best_params = param_search(
        dnn,
        X_train, y,
        cv=param_search_config.param_search_cv,
        study=study,
        n_trials=param_search_config.n_trials,
        keep_going=False
    )
    print("Training")
    dnn.set_params(**best_params)
    dnn.fit(X_train, y)

    return preprocessor, dnn

