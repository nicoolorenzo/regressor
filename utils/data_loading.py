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

    Returns:
        tuple: A tuple containing:
            - X (numpy.ndarray): The merged dataset consisting of descriptors and fingerprints.
            - y (numpy.ndarray): The target values (correct ccs averages).
            - desc_cols (numpy.ndarray): Indices of columns corresponding to descriptors in the merged dataset.
            - fgp_cols (numpy.ndarray): Indices of columns corresponding to fingerprints in the merged dataset.
    """
    experiment_data = {}
    # If we are running a smoke test, and we've already created the complete dataset then:
    if is_smoke_test and os.path.exists("./resources/descriptors_and_fingerprints.pklz"):
            # If we have created the "smoke dataset", load it
        if os.path.exists("./resources/smoke_dataset.pklz"):
            with bz2.BZ2File("./resources/smoke_dataset.pklz", "rb") as f:
                X, y, desc_cols, fgp_cols = pickle.load(f)
        # If we haven't, create it
        if not os.path.exists("./resources/smoke_dataset.pklz"):
            # Load the complete dataset
            with bz2.BZ2File("./resources/descriptors_and_fingerprints.pklz", "rb") as f:
                X, y, desc_cols, fgp_cols = pickle.load(f)
            # Drop most of the dataset
            X = X[:1500]
            y = y[:1500]
            # Save the slimmed down data to a file called "smoke_dataset.pklz" for future smoke tests
            with bz2.BZ2File("./resources/smoke_dataset.pklz", "wb") as f:
                pickle.dump([X, y, desc_cols, fgp_cols], f)

        # Do this necessary preformatting step
        X = X.astype('float32')
        y = np.array(y).astype('float32').flatten()

        # Return the smoke dataset
        return X, y, desc_cols, fgp_cols


    # Check if we have the file with both databases already merged, and if not, merge them
    if os.path.exists("./resources/descriptors_and_fingerprints.pklz"):
        with bz2.BZ2File("./resources/descriptors_and_fingerprints.pklz", "rb") as f:
            X, y, desc_cols, fgp_cols = pickle.load(f)
            experiment_data[0] = (0, 0)
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
        if not chromatography_column and not is_smrt:
            descriptors_and_fingerprints = descriptors_and_fingerprints.drop(columns=descriptors.loc[:, "column.usp.code_0":"flow_rate 17"].columns, axis=1)
            descriptors = descriptors.drop(columns=descriptors.loc[:, "column.usp.code_0":"flow_rate 17"].columns, axis=1)
            number_columns = descriptors_and_fingerprints["id"].str[0:4].drop_duplicates().values
            number_molecules = 0
            for value in number_columns:
                experiment = int(descriptors_and_fingerprints[descriptors_and_fingerprints["id"].str.startswith(value)].shape[0])
                experiment_data[value] = (number_molecules, number_molecules + experiment)
                number_molecules = number_molecules + experiment
        else:
            experiment_data[0] = (0, 0)

        X_desc = descriptors_and_fingerprints[descriptors.drop(common_columns, axis=1).columns].values
        X_fgp = descriptors_and_fingerprints[fingerprints.drop(common_columns, axis=1).columns].values

        X = np.concatenate([X_desc, X_fgp], axis=1)
        labels_column = common_columns[1]
        y = descriptors_and_fingerprints[labels_column].values.flatten()

        desc_cols = np.arange(X_desc.shape[1], dtype='int')
        fgp_cols = np.arange(X_desc.shape[1], X.shape[1], dtype='int')
        if is_smrt:
            if os.path.exists("./resources/descriptors_and_fingerprints_SMRT.pklz"):
                with bz2.BZ2File("./resources/descriptors_and_fingerprints_SMRT.pklz", "rb") as smrt:
                    X_smrt, y_smrt, desc_cols_smrt, fgp_cols_smrt = pickle.load(smrt)
                    if (desc_cols_smrt == desc_cols) and (fgp_cols_smrt == fgp_cols):
                        X = np.concatenate([X, X_smrt], axis=0)
                        y = np.concatenate([y, y_smrt], axis=0)
        # Save the file that will be use for training
        with bz2.BZ2File("./resources/descriptors_and_fingerprints.pklz", "wb") as f:
            pickle.dump([X, y, desc_cols, fgp_cols], f)

    if is_smoke_test:
        # Drop most of the dataset
        X = X[:1500]
        y = y[:1500]
        # Save the slimmed down data to a file called "smoke_dataset.pklz" for future smoke tests
        with bz2.BZ2File("./resources/smoke_dataset.pklz", "wb") as f:
            pickle.dump([X, y, desc_cols, fgp_cols], f)

    X = X.astype('float32')
    y = np.array(y).astype('float32').flatten()

    return X, y, desc_cols, fgp_cols, experiment_data






