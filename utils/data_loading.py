import bz2
import os
import pickle
import numpy as np
import pandas as pd
from utils.cure_descriptors_and_fingerprints import cure


def get_my_data(common_columns, is_smoke_test, is_smrt, chromatography_column):
    """
    Load or merge Alvadesk files containing descriptors and fingerprints, returning the necessary data for training.

    Args:
        common_columns (list): List of common columns used to merge descriptors and fingerprints.
        is_smoke_test (bool): Argument to create or to load a smaller dataset
        is_smrt (bool): Argument to include SMRT dataset
        chromatography_column (bool): Argument to include chromatography column data and separate in different experiments

    Returns:
        tuple: A tuple containing:
            - X (numpy.ndarray): The merged dataset consisting of descriptors and fingerprints.
            - y (numpy.ndarray): The target values (correct ccs averages).
            - desc_cols (numpy.ndarray): Indices of columns corresponding to descriptors in the merged dataset.
            - fgp_cols (numpy.ndarray): Indices of columns corresponding to fingerprints in the merged dataset.
            - experiment_data (dict): Position of each experiment in X if chromatography_column is True
    """
    experiment_data = {"_": (0, -1)}
    # If we are running a smoke test, and we've already created the complete dataset then:
    if is_smoke_test and os.path.exists("./resources/descriptors_and_fingerprints_RepoRT.pklz"):
            # If we have created the "smoke dataset", load it
        if os.path.exists("./resources/smoke_dataset.pklz"):
            with bz2.BZ2File("./resources/smoke_dataset.pklz", "rb") as f:
                X, y, desc_cols, fgp_cols = pickle.load(f)
        # If we haven't, create it
        if not os.path.exists("./resources/smoke_dataset.pklz"):
            # Load the complete dataset
            with bz2.BZ2File("./resources/descriptors_and_fingerprints_RepoRT.pklz", "rb") as f:
                X, y, desc_cols, fgp_cols = pickle.load(f)
            # Drop most of the dataset
            X = X[:1500]
            y = y[:1500]
            # Save the slimmed down data to a file called "smoke_dataset.pklz" for future smoke tests
            with bz2.BZ2File("./resources/smoke_dataset.pklz", "wb") as f:
                pickle.dump([X, y, desc_cols, fgp_cols], f)

        if not chromatography_column and not is_smrt:
            experiment_data, X, desc_cols, fgp_cols = delete_chromatography_columns(X)
        # Do this necessary preformatting step
        X = X.drop(columns=common_columns, axis=1).astype('float32')
        y = np.array(y).astype('float32').flatten()

        # Return the smoke dataset
        return X, y, desc_cols, fgp_cols, experiment_data


    # Check if we have the file with both databases already merged, and if not, merge them
    if os.path.exists("./resources/descriptors_and_fingerprints_RepoRT.pklz"):
        with bz2.BZ2File("./resources/descriptors_and_fingerprints_RepoRT.pklz", "rb") as f:
            X, y, desc_cols, fgp_cols = pickle.load(f)
    else:
        # Load the original files created with Alvadesk
        descriptors = pd.read_csv("./resources/des_no_SMRT.tsv", sep="\t")
        fingerprints = pd.read_csv("./resources/fgp_no_SMRT.tsv", sep="\t")

        # Create the file that will be used for training
        print('Merging')
        descriptors = descriptors.drop("inchi.std", axis=1)
        fingerprints = fingerprints.drop("inchi.std", axis=1)

        descriptors_and_fingerprints = pd.merge(descriptors, fingerprints, on=common_columns)
        descriptors_and_fingerprints = descriptors_and_fingerprints.fillna(0)
        descriptors_and_fingerprints["rt"] = descriptors_and_fingerprints["rt"]*60
        X_desc = descriptors_and_fingerprints[descriptors.columns]
        X_fgp = descriptors_and_fingerprints[fingerprints.drop(common_columns, axis=1).columns]

        X = pd.concat([X_desc, X_fgp], axis=1)
        labels_column = common_columns[1]
        y = descriptors_and_fingerprints[labels_column].values.flatten()

        desc_cols = np.arange(X_desc.drop(common_columns, axis=1).shape[1], dtype='int')
        fgp_cols = np.arange(X_desc.drop(common_columns, axis=1).shape[1], X.drop(common_columns, axis=1).shape[1], dtype='int')

        # Save the file that will be use for training
        with bz2.BZ2File("./resources/descriptors_and_fingerprints_RepoRT.pklz", "wb") as f:
            pickle.dump([X, y, desc_cols, fgp_cols], f)

    if is_smoke_test:
        # Drop most of the dataset
        X = X[:1500]
        y = y[:1500]
        # Save the slimmed down data to a file called "smoke_dataset.pklz" for future smoke tests
        with bz2.BZ2File("./resources/smoke_dataset.pklz", "wb") as f:
            pickle.dump([X, y, desc_cols, fgp_cols], f)

    if not chromatography_column and not is_smrt:
        experiment_data, X, desc_cols, fgp_cols = delete_chromatography_columns(X)

    X = X.drop(columns=common_columns, axis=1).astype('float32')
    y = np.array(y).astype('float32').flatten()

    if is_smrt:
        if os.path.exists("./resources/descriptors_and_fingerprints_SMRT.pklz"):
            with bz2.BZ2File("./resources/descriptors_and_fingerprints_SMRT.pklz", "rb") as smrt:
                X_smrt, y_smrt, desc_cols_smrt, fgp_cols_smrt = pickle.load(smrt)
                if (desc_cols_smrt == desc_cols) and (fgp_cols_smrt == fgp_cols):
                    X = np.concatenate([X, X_smrt], axis=0)
                    y = np.concatenate([y, y_smrt], axis=0)

    return X, y, desc_cols, fgp_cols, experiment_data


def delete_chromatography_columns(X):
    X = X.drop(columns=X.loc[:, "column.usp.code_0":"flow_rate 17"].columns, axis=1)
    desc_cols = np.arange(X.loc[:, "MW":"chiralPhMoment"].shape[1], dtype='int')
    fgp_cols = np.arange(desc_cols.shape[0], desc_cols.shape[0] + X.loc[:,"V1":].shape[1], dtype='int')
    number_columns = X["id"].str[0:4].drop_duplicates().values
    experiment_data = {}
    number_molecules = 0
    for value in number_columns:
        experiment = int(X[X["id"].str.startswith(value)].shape[0])
        experiment_data[value] = (number_molecules, number_molecules + experiment)
        number_molecules = number_molecules + experiment
    return experiment_data, X, desc_cols, fgp_cols

