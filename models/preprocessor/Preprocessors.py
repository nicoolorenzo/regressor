import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.preprocessor.ThresholdSelectors import CorThreshold


def is_binary_feature(x):
    """Indicates if the given feature is binary (0 and 1).

    :param x: feature to be checked.
    :return: Boolean indicating if the given feature is binary.
    """
    ux = np.unique(x)
    if len(ux) == 1:
        return ux == 0 or ux == 1
    if len(ux) == 2:
        return np.all(np.sort(ux) == np.array([0, 1]))
    else:
        return False


def binary_features_cols(X):
    """Get column indices of binary features.

    :param X: numpy array of features.
    :return: numpy array of column indices of binary features.
    """
    return np.where(np.apply_along_axis(is_binary_feature, 0, X))[0]


class Preprocessor(BaseEstimator):
    def __init__(self, desc_cols, fgp_cols, p=0, cor_th=0.9, k='all'):
        """ We assume that the adducts indicators are part of the fgp_cols"""
        self.desc_cols = desc_cols
        self.fgp_cols = fgp_cols
        self.p = p
        self.cor_th = cor_th
        self.k = k

    def _init_hidden_models(self):
        self._desc_preprocessor = DescriptorsPreprocessor(
            desc_cols=self.desc_cols,
            adduct_cols=None,
            k=self.k,
            cor_th=self.cor_th
        )
        self._fgp_preprocessor = FgpPreprocessor(fgp_cols=self.fgp_cols, p=self.p)

    def fit_transform(self, X, y=None):
        if y is None:
            # fit method of arity 1 (unsupervised transformation)
            return self.fit(X).transform(X)
        else:
            # fit method of arity 2 (supervised transformation)
            return self.fit(X, y).transform(X)

    def fit(self, X, y=None):
        self._init_hidden_models()
        self._desc_preprocessor.fit(X, y)
        self._fgp_preprocessor.fit(X, y)
        return self

    def transform(self, X, y=None):
        X_desc_proc = self._desc_preprocessor.transform(X)
        X_fgp_proc = self._fgp_preprocessor.transform(X)
        new_X = np.concatenate([X_desc_proc, X_fgp_proc], axis=1)
        # Annotate which columns are related to descriptors an fingerprints after transformation. Also, annotate which
        # columns can be considered binary
        self.transformed_desc_cols = np.concatenate([
            np.arange(X_desc_proc.shape[1]),
            np.arange(new_X.shape[1]-3, new_X.shape[1])
        ])
        self.transformed_fgp_cols = np.arange(X_desc_proc.shape[1], new_X.shape[1], dtype='int')
        self.transformed_binary_cols = binary_features_cols(new_X)
        return new_X

    def describe_transformed_features(self):
        return {
            'n_descriptors': len(self.transformed_desc_cols),
            'n_fgp': len(self.transformed_fgp_cols),
            'binary_cols': self.transformed_binary_cols,
            'desc_cols': self.transformed_desc_cols,
            'fgp_cols': self.transformed_fgp_cols
        }


class DescriptorsPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, desc_cols, adduct_cols, cor_th=0.9, k='all'):
        self.desc_cols = desc_cols
        self.adduct_cols = adduct_cols
        self.cor_th = cor_th
        self.k = k

    def _init_hidden_models(self):
        self._desc_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('imputation', SimpleImputer(missing_values=np.nan, strategy='median', add_indicator=True)),
            ('var_threshold', VarianceThreshold()),
            ('cor_selector', CorThreshold(threshold=self.cor_th)),
            ('f_selector', SelectKBest(score_func=f_regression, k=self.k))
        ])

    def fit(self, X, y=None):
        self._init_hidden_models()
        X_desc = X[:, self.desc_cols]
        self._desc_pipeline.fit(X_desc, y)
        return self

    def transform(self, X, y=None):
        X_desc = X[:, self.desc_cols]
        X_desc_proc = self._desc_pipeline.transform(X_desc)
        if self.adduct_cols is not None:
            X_adduct_indicator = X[:, self.adduct_cols]
            new_X = np.concatenate([X_desc_proc, X_adduct_indicator], axis=1)
        else:
            new_X = X_desc_proc
        # Annotate which columns are related to descriptors an fingerprints after transformation. Also, annotate which
        # columns can be considered binary
        self.transformed_binary_cols = binary_features_cols(new_X)
        return new_X


class FgpPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, fgp_cols, p=0.9):
        self.fgp_cols = fgp_cols
        self.p = p

    def _init_hidden_models(self):
        self._fgp_vs = VarianceThreshold(threshold=self.p * (1 - self.p))

    def fit(self, X, y=None):
        self._init_hidden_models()
        X_fgp = X[:, self.fgp_cols]
        _ = self._fgp_vs.fit_transform(X_fgp)
        return self

    def transform(self, X, y=None):
        X_fgp = X[:, self.fgp_cols]
        self.transformed_binary_cols = binary_features_cols(X_fgp)
        return X_fgp