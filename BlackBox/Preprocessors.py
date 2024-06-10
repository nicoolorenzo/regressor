import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted


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


class Preprocessor(BaseEstimator, TransformerMixin):
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
            k=self.k,
            cor_th=self.cor_th
        )
        self._fgp_preprocessor = FgpPreprocessor(fgp_cols=self.fgp_cols, p=self.p)

    def fit(self, X, y=None):
        self._init_hidden_models()
        self._desc_preprocessor.fit(X, y)
        self._fgp_preprocessor.fit(X, y)
        return self

    def transform(self, X, y=None):
        X_desc_proc = self._desc_preprocessor.transform(X)
        X_fgp_proc = self._fgp_preprocessor.transform(X)
        if X_fgp_proc.shape[1] == 11:
            new_X = pd.concat([X_fgp_proc, X_desc_proc], axis=1)
        else:
            new_X = pd.concat([X_desc_proc, X_fgp_proc], axis=1)
        # Annotate which columns are related to descriptors an fingerprints after transformation. Also, annotate which
        # columns can be considered binary
        self.transformed_desc_cols = np.concatenate([np.arange(X_desc_proc.shape[1]),np.arange(new_X.shape[1]-3, new_X.shape[1])])
        self.transformed_fgp_cols = np.arange(X_desc_proc.shape[1], new_X.shape[1], dtype='int')
        self.transformed_binary_cols = np.where(np.apply_along_axis(is_binary_feature, 0, new_X))[0]
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
    def __init__(self, desc_cols, cor_th=0.99, k='all'):
        self.desc_cols = desc_cols
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
        X_desc = X.iloc[:, self.desc_cols]
        self._desc_pipeline.set_output(transform="pandas")
        self._desc_pipeline.fit(X_desc, y)
        return self

    def transform(self, X, y=None):
        X_desc = X.iloc[:, self.desc_cols]
        self._desc_pipeline.set_output(transform="pandas")
        X_desc_proc = self._desc_pipeline.transform(X_desc)
        new_X = X_desc_proc
        # Annotate which columns are related to descriptors an fingerprints after transformation. Also, annotate which
        # columns can be considered binary
        self.transformed_binary_cols = np.where(np.apply_along_axis(is_binary_feature, 0, new_X))[0]
        return new_X

class FgpPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, fgp_cols, p=0.9):
        self.fgp_cols = fgp_cols
        self.p = p

    def _init_hidden_models(self):
        self._fgp_vs = VarianceThreshold(threshold=self.p * (1 - self.p))

    def fit(self, X, y=None):
        self._init_hidden_models()
        X_fgp = X.iloc[:, self.fgp_cols]
        self._fgp_vs.set_output(transform="pandas")
        _ = self._fgp_vs.fit_transform(X_fgp)
        return self

    def transform(self, X, y=None):
        X_fgp = X.iloc[:, self.fgp_cols]
        self.transformed_binary_cols = np.where(np.apply_along_axis(is_binary_feature, 0, X_fgp))[0]
        return X_fgp


class CorThreshold(SelectorMixin, BaseEstimator):
    def __init__(self, threshold=0.):
            self.threshold = threshold

    def fit(self, X, y=None):
        cor_matrix = np.abs(np.corrcoef(X, rowvar=False))
        self.upper_tri_ = np.triu(cor_matrix, k=1)
        return self

    def _get_support_mask(self):
        check_is_fitted(self)
        n = self.upper_tri_.shape[1]
        return np.array([all(self.upper_tri_[:column, column] < self.threshold) for column in range(n)])
