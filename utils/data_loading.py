import bz2
import os
import pickle
import pandas as pd
from glob import glob
import numpy as np
from scripts import Gradient_data
import re
from formula_validation.Formula import Formula
import zipfile
import io



def get_my_data():
    """
        Accesses RepoRT chromatography data based on a specific molecule and column pattern and merges it with alvaDesc
        descriptors and fingerprints

    This function filters RepoRT data based on the provided molecule pattern and column location,
    and merges it with an alternative parents dataset. Then it loads and merges alvaDesc files containing descriptors and
    fingerprints, returning the necessary chromatography data for training.

    Returns:
        tuple: A tuple containing:
            - X (DataFrame): The merged dataset consisting of chromatographic data, descriptors and fingerprints.
            - y (DataFrame): The target values (correct rt averages).
            - usp_cols (numpy.ndarray): Indices of columns corresponding to usp columns in the merged dataset.
            - chromatography_columns (numpy.ndarray): Indices of columns corresponding to chromatographic columns in the merged dataset.
            - desc_cols (numpy.ndarray): Indices of columns corresponding to descriptors in the merged dataset.
            - fgp_cols (numpy.ndarray): Indices of columns corresponding to fingerprints in the merged dataset.
    """

    # Obtain chromatographic data from RepoRT
    if (not os.path.exists("./resources/RepoRT_extracted_data.zip")
            and not os.path.exists("./resources/report_unique_inchis_descriptorsAndFingerprintsVectorized.zip")
            and not os.path.exists("./resources/chromatography_descriptors_and_fingerprints_RepoRT.pklz")):
        try:
            location = ""
            pattern = ".*"
            imputation = True
            RepoRT_directory = glob("./resources/RepoRT_data/*/*.tsv")
            rt_alt_par_list = []
            column_filter = None
            with zipfile.ZipFile('./resources/RepoRT_alternative_parents_classified.zip', 'r') as zip_ref:
                with zip_ref.open("RepoRT_classified.tsv") as tsv_file:
                    alt_parents_data = pd.read_csv(tsv_file, sep='\t', header=0, encoding='utf-8',
                                           dtype=object)
            for file in RepoRT_directory:
                if re.search(r"_rtdata_canonical_success", file):
                    RepoRT_rtdata = pd.read_csv(file, sep='\t', header=0, encoding='utf-8')
                    column_filter = RepoRT_rtdata.filter(regex=f'{location}', axis=1)
                    column_filter = column_filter.select_dtypes(include=['object'])
                    for column in column_filter.columns:
                        matching_pattern = RepoRT_rtdata[
                            column_filter[column].str.lower().str.contains(pattern.lower(), na=False)]
                        if not matching_pattern.empty:
                            merge_rt_alt_parents = matching_pattern.merge(
                                alt_parents_data.drop(columns=[col for col in matching_pattern.columns[1:]] + ["0"]),
                                left_on="id", right_on="id", how="left")
                            rt_alt_par_list.append(merge_rt_alt_parents)
                            break
            if column_filter is not None and column_filter.size == 0:
                print(f"{location} not found")
            elif rt_alt_par_list:
                molecule_data = pd.concat(rt_alt_par_list, axis=0, ignore_index=True)
                molecule_data["alternative_parents"] = (molecule_data.iloc[:, 14:].astype(str)
                                                        .apply(lambda x: ", ".join(x.drop_duplicates()), axis=1))
                molecule_data = (molecule_data.drop(columns=molecule_data.columns[14:287]).replace("NA (NA)", np.nan)
                                 .set_index(molecule_data["id"].str[0:4].astype(int)))
                for pos, inchi in enumerate(molecule_data["inchi.std"]):
                    try:
                        formula = Formula.formula_from_inchi(inchi, None)
                    except Exception as e:
                        smiles = molecule_data.loc[pos, "smiles.std"]
                        formula = Formula.formula_from_smiles(smiles, None)
                    molecule_data.loc[pos, "new_formula"] = str(formula)
                formula_column = molecule_data.pop("new_formula")
                molecule_data.insert(3, "new_formula", formula_column)
                column_data = Gradient_data.gradient_data(imputation)
                molecule_column_data = pd.merge(molecule_data, column_data, left_index=True, right_index=True,
                                                how="inner")
                molecule_column_data = molecule_column_data.fillna(0)
                tsv_buffer = io.StringIO()
                molecule_column_data.to_csv(tsv_buffer, sep='\t', index=False)
                tsv_data = tsv_buffer.getvalue()
                with zipfile.ZipFile("./resources/RepoRT_extracted_data.zip", 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    zip_file.writestr("RepoRT_extracted_data.tsv", tsv_data)
            else:
                print(f'No matches found with {pattern}')
        except Exception as e:
            print(f"Error:{e}")

    # Load alvaDesc files containing descriptors and fingerprints.
    if (not os.path.exists("./resources/report_unique_inchis_descriptorsAndFingerprintsVectorized.zip")
            and not os.path.exists("./resources/chromatography_descriptors_and_fingerprints_RepoRT.pklz")):
        print("Download report_unique_inchis_descriptorsAndFingerprintsVectorized.zip file in "
              "https://upm365-my.sharepoint.com/:u:/g/personal/ana_amil_alumnos_upm_es/EZgWDOgcGCxGjg2nH9I3Z4cBduu1SHlbEQMgS9pjAnTVaA?e=1K6pmG")
        pass

    # Merge alvaDesc files with chromatography data for training.
    if not os.path.exists("./resources/chromatography_descriptors_and_fingerprints_RepoRT.pklz"):

        with zipfile.ZipFile('./resources/RepoRT_extracted_data.zip', 'r') as zip_ref:
            with zip_ref.open("RepoRT_classified_CCinformation.tsv") as tsv_file:
                chromatography_data = pd.read_csv(tsv_file, sep='\t', header=0, encoding='utf-8', dtype=object)
        with zipfile.ZipFile('./resources/report_unique_inchis_descriptorsAndFingerprintsVectorized.zip', 'r') as zip_ref:
            with zip_ref.open("report_unique_inchis_descriptorsAndFingerprintsVectorized.csv") as tsv_file:
                descriptors_fingerprints = pd.read_csv(tsv_file, sep=',', header=0, encoding='utf-8', dtype=object)

        columns_in_data = chromatography_data.drop(columns="inchi.std").columns
        columns_in_des_and_fgn = [i for i in columns_in_data if i in descriptors_fingerprints]
        descriptors_fingerprints = descriptors_fingerprints.drop(columns=columns_in_des_and_fgn).drop(columns="column.usp.code")
        columns_to_drop = ["name", "formula", "smiles.std", "inchikey.std", "classyfire.kingdom",
                           "classyfire.superclass",
                           "classyfire.class", "classyfire.subclass", "classyfire.level5", "classyfire.level6",
                           "alternative_parents", "comment", "column.name"]
        chromatography_data = chromatography_data.drop(columns=columns_to_drop)
        descriptors_fingerprints = descriptors_fingerprints[~descriptors_fingerprints["MW"].isnull()]
        print('Merging')
        merge_des_and_fgp = pd.merge(chromatography_data, descriptors_fingerprints, on="inchi.std")
        merge_des_and_fgp = merge_des_and_fgp.fillna(0)
        merge_des_and_fgp["rt"] = merge_des_and_fgp["rt"].astype("float32") * 60

        X_usp = merge_des_and_fgp.loc[:, "column.usp.code_0":"column.usp.code_L7"].columns
        X_chromatography = merge_des_and_fgp.loc[:, "column.length":"flow_rate 17"].columns
        X_descriptors = merge_des_and_fgp.loc[:, "MW":"chiralPhMoment"].columns

        X = merge_des_and_fgp.drop(columns="inchi.std")
        y = X["rt"].values.flatten()

        usp_columns = np.arange(X_usp.shape[0], dtype='int')
        chromatography_columns = np.arange(usp_columns[-1]+1,usp_columns[-1]+1+X_chromatography.shape[0], dtype='int')
        descriptors_columns = np.arange(chromatography_columns[-1]+1, chromatography_columns[-1]+1+X_descriptors.shape[0], dtype='int')
        fingerprints_columns = np.arange(descriptors_columns[-1]+1, X.drop(["id", "rt"], axis=1).shape[1], dtype='int')

        # Save the file that will be use for training
        with bz2.BZ2File("./resources/chromatography_descriptors_and_fingerprints_RepoRT.pklz", "wb") as f:
            pickle.dump([X, y, usp_columns, chromatography_columns, descriptors_columns, fingerprints_columns], f)

    else:
        with bz2.BZ2File("./resources/chromatography_descriptors_and_fingerprints_RepoRT.pklz", "rb") as f:
            X, y, usp_columns, chromatography_columns, descriptors_columns, fingerprints_columns = pickle.load(f)


    return X, y, usp_columns, chromatography_columns, descriptors_columns, fingerprints_columns



