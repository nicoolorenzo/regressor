from sklearn.preprocessing import QuantileTransformer
from BlackBox.Preprocessors import FgpPreprocessor, DescriptorsPreprocessor, Preprocessor
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_X_except_usp(usp_columns, chromatography_columns, descriptors_columns,fingerprints_columns,
                     train_X, train_y, test_X, test_y, features):
    if features == "fingerprints":
        preproc_chromatography = Preprocessor(desc_cols=chromatography_columns, fgp_cols=usp_columns)
        preproc_des_fgp = FgpPreprocessor(fgp_cols=fingerprints_columns)
    elif features == "descriptors":
        preproc_chromatography = Preprocessor(desc_cols=chromatography_columns, fgp_cols=usp_columns)
        preproc_des_fgp = DescriptorsPreprocessor(desc_cols=descriptors_columns)
    else:
        preproc_des_fgp = Preprocessor(desc_cols=descriptors_columns, fgp_cols=fingerprints_columns)
        preproc_chromatography = Preprocessor(desc_cols=chromatography_columns, fgp_cols=usp_columns)
    preproc_des_fgp_train_X = preproc_des_fgp.fit_transform(train_X, train_y)
    preproc_des_fgp_test_X = preproc_des_fgp.transform(test_X, test_y)
    preproc_chrom_train_X = preproc_chromatography.fit_transform(train_X, train_y)
    preproc_chrom_test_X = preproc_chromatography.transform(test_X, test_y)
    preproc_train_X = pd.concat([preproc_chrom_train_X, preproc_des_fgp_train_X], axis=1)
    preproc_test_X = pd.concat([preproc_chrom_test_X, preproc_des_fgp_test_X], axis=1)
    return preproc_train_X, preproc_test_X


def preprocess_X_except_chromatography(usp_columns, chromatography_columns, descriptors_columns,fingerprints_columns,
                                       train_X, train_y, test_X, test_y, features):
    if features == "fingerprints":
        preproc = FgpPreprocessor(fgp_cols=fingerprints_columns)
    elif features == "descriptors":
        preproc = DescriptorsPreprocessor(desc_cols=descriptors_columns)
    else:
        preproc = Preprocessor(desc_cols=descriptors_columns, fgp_cols=fingerprints_columns)
    scaler = StandardScaler().set_output(transform="pandas")
    chromatography_train = scaler.fit_transform(train_X.iloc[:, chromatography_columns])
    chromatography_test = scaler.fit_transform(test_X.iloc[:, chromatography_columns])
    chromatography_train = pd.concat([train_X.iloc[:, usp_columns], chromatography_train], axis=1)
    chromatography_test = pd.concat([test_X.iloc[:, usp_columns], chromatography_test], axis=1)
    preproc_train_X = preproc.fit_transform(train_X, train_y)
    preproc_test_X = preproc.transform(test_X, test_y)
    preproc_train_X = pd.concat([chromatography_train, preproc_train_X], axis=1)
    preproc_test_X = pd.concat([chromatography_test, preproc_test_X], axis=1)
    return preproc_train_X, preproc_test_X


def preprocess_X(fingerprints_columns, descriptors_columns, train_X, train_y, test_X, test_y, features):
    if features == "fingerprints":
        preproc = FgpPreprocessor(fgp_cols=fingerprints_columns)
    elif features == "descriptors":
        preproc = DescriptorsPreprocessor(desc_cols=descriptors_columns)
    else:
        preproc = Preprocessor(desc_cols=descriptors_columns, fgp_cols=fingerprints_columns)
    preproc_train_X = preproc.fit_transform(train_X, train_y)
    preproc_test_X = preproc.transform(test_X, test_y)
    return preproc_train_X, preproc_test_X

def preprocess_y(train_y, test_y):
    preproc_y = QuantileTransformer(n_quantiles=1000, output_distribution='normal')
    train_preproc_y = preproc_y.fit_transform(train_y.reshape(-1, 1))
    test_preproc_y = preproc_y.transform(test_y.reshape(-1, 1))
    return train_preproc_y.flatten(), test_preproc_y.flatten(), preproc_y
