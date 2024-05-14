import pandas as pd
import os
import bz2
import numpy as np
import pickle


def des_and_fgp():
    data = pd.read_csv("../resources/RepoRT_classified_CCinformation.tsv", sep='\t', header=0, encoding='utf-8')
    descriptors = pd.read_csv("../resources/des_and_fgp_data/report_unique_inchis_descriptors.csv",
                                           sep=',', header=0, encoding='utf-8')
    fingerprints = pd.read_csv("../resources/des_and_fgp_data/report_unique_inchis_vectorfingerprintsVectorized.csv",
                                           sep=',', header=0, encoding='utf-8')
    columns_in_data = data.drop(columns="inchi.std").columns
    columns_in_des_and_fgn = [i for i in columns_in_data if i in fingerprints]
    descriptors = descriptors.drop(columns=columns_in_des_and_fgn).drop(columns="column.usp.code")
    fingerprints = fingerprints.drop(columns=columns_in_des_and_fgn).drop(columns="column.usp.code")
    columns_to_drop = ["name", "formula", "smiles.std", "inchikey.std", "classyfire.kingdom", "classyfire.superclass",
                       "classyfire.class", "classyfire.subclass", "classyfire.level5", "classyfire.level6",
                       "alternative_parents", "comment", "column.name"]
    data = data.drop(columns=columns_to_drop)
    descriptors = descriptors[~descriptors["MW"].isnull()]
    des_no_SMRT = pd.merge(data, descriptors, on="inchi.std")
    fng_no_SMRT = pd.merge(data.iloc[:, :3], fingerprints, on="inchi.std")
    des_no_SMRT.to_csv("../resources/des_no_SMRT.tsv", sep='\t', index=False)
    fng_no_SMRT.to_csv("../resources/fgp_no_SMRT.tsv", sep='\t', index=False)


def get_smrt():
    data = pd.read_csv("../resources/RepoRT_classified_CCinformation.tsv", sep='\t', header=0, encoding='utf-8')
    smrt_data = data[data["id"].str.startswith(r'0186_')]
    smrt_column_data = smrt_data.loc[:, "column.usp.code_0":]
    number_columns_column_data = smrt_column_data.shape[1]
    if os.path.exists("../resources/descriptors_and_fingerprints.pklz"):
        with bz2.BZ2File("../resources/descriptors_and_fingerprints.pklz", "rb") as f:
            X, y, desc_cols, fgp_cols = pickle.load(f)
        X = pd.DataFrame(X)
        smrt_column_data = smrt_column_data.set_index(X.index)
        X = pd.concat([smrt_column_data, X], axis=1).values
        desc_cols = np.arange(desc_cols.shape[0] + number_columns_column_data, dtype='int')
        fgp_cols = np.arange(desc_cols.shape[0], X.shape[1], dtype='int')
        with bz2.BZ2File("../resources/descriptors_and_fingerprints_SMRT.pklz", "wb") as f:
            pickle.dump([X, y, desc_cols, fgp_cols], f)
