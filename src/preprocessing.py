from sklearn.preprocessing import QuantileTransformer
from BlackBox.Preprocessors import FgpPreprocessor, DescriptorsPreprocessor, Preprocessor
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_X_usp(fingerprints_columns, descriptors_columns, train_X, train_y, test_X, test_y, features):
    if features == "fingerprints":
        preproc = FgpPreprocessor(fgp_cols=fingerprints_columns)
        preproc_train_X = preproc.fit_transform(train_X, train_y)
        preproc_test_X = preproc.transform(test_X, test_y)
    elif features == "descriptors":
        usp_columns = descriptors_columns[:train_X.loc[:, "column.usp.code_0":"column.usp.code_L7"].shape[1]]
        des_columns = descriptors_columns[train_X.loc[:, "column.usp.code_0":"column.usp.code_L7"].shape[1]:]
        preproc = Preprocessor(desc_cols=des_columns, fgp_cols=usp_columns)
        preproc_train_X = preproc.fit_transform(train_X, train_y)
        preproc_test_X = preproc.transform(test_X, test_y)
    else:
        usp_columns = descriptors_columns[:train_X.loc[:, "column.usp.code_0":"column.usp.code_L7"].shape[1]]
        des_columns = descriptors_columns[train_X.loc[:, "column.usp.code_0":"column.usp.code_L7"].shape[1]:]
        preproc_fgp = FgpPreprocessor(fgp_cols=fingerprints_columns)
        preproc_des = Preprocessor(desc_cols=des_columns, fgp_cols=usp_columns)
        preproc_des_train_X = preproc_des.fit_transform(train_X, train_y)
        preproc_des_test_X = preproc_des.transform(test_X, test_y)
        preproc_fgp_train_X = preproc_fgp.fit_transform(train_X, train_y)
        preproc_fgp_test_X = preproc_fgp.transform(test_X, test_y)
        preproc_train_X = pd.concat([preproc_des_train_X, preproc_fgp_train_X], axis=1)
        preproc_test_X = pd.concat([preproc_des_test_X, preproc_fgp_test_X], axis=1)
    return preproc_train_X, preproc_test_X


def preprocess_X_chromatography(fingerprints_columns, descriptors_columns, train_X, train_y, test_X, test_y,
                 chromatography_train, chromatography_test, features):
    if features == "descriptors":
        preproc = DescriptorsPreprocessor(desc_cols=descriptors_columns)
    else:
        preproc = Preprocessor(desc_cols=descriptors_columns, fgp_cols=fingerprints_columns)
    scaler = StandardScaler().set_output(transform="pandas")
    chromatography_train_sel = scaler.fit_transform(chromatography_train.iloc[:, 11:])
    chromatography_test_sel = scaler.fit_transform(chromatography_test.iloc[:, 11:])
    chromatography_train = pd.concat([chromatography_train.iloc[:, 0:11], chromatography_train_sel], axis=1)
    chromatography_test = pd.concat([chromatography_test.iloc[:, 0:11], chromatography_test_sel], axis=1)
    preproc_train_X = preproc.fit_transform(train_X, train_y)
    preproc_test_X = preproc.transform(test_X, test_y)
    return preproc_train_X, preproc_test_X, chromatography_train, chromatography_test


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
