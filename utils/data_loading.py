import bz2
import os
import pickle
import numpy as np
import pandas as pd
from utils.cure_descriptors_and_fingerprints import cure


def get_my_data():
    #FIXME: Ana docstring
    """
        Accesses RepoRT data based on a specified molecule pattern and column.

    This function searches for files in the '../data/*/' RepoRT_directory containing molecule and retention time data.
    It reads the data from these files, filters it based on the provided molecule pattern and column location,
    and merges it with an alternative parents dataset.

    Args:
        pattern (str, optional): Molecule pattern or name to search for in the data. Default value "", representing
        all types of molecules.
        location (str, optional): Column name to search for the pattern. Default value ".*", representing all columns.
        imputation (bool, optional): Indicates whether training data processing is performed. Default value True.

    Returns:
        DataFrame: Processed DataFrame containing the merged data with its chromatographic information.

    Load or merge Alvadesk files containing descriptors and fingerprints, returning the necessary data for training.

    Args:
        common_columns (list): List of common columns used to merge descriptors and fingerprints.
        is_smoke_test (bool): Argument to create or to load a smaller dataset
        is_smrt (bool): Argument to include SMRT dataset
        chromatography_column (bool): Argument to include chromatography column data and separate in different experiments

    Returns:
        tuple: A tuple containing:
            - X (DataFrame): The merged dataset consisting of descriptors and fingerprints.
            - y (DataFrame): The target values (correct rt averages).
            - desc_cols (numpy.ndarray): Indices of columns corresponding to descriptors in the merged dataset.
            - fgp_cols (numpy.ndarray): Indices of columns corresponding to fingerprints in the merged dataset.
            - experiment_data (dict): Position of each experiment in X if chromatography_column is True
    """

    # Ana añade comentario corto
    if (not os.path.exists("./resources/RepoRT_extracted_data.zip")
            and not os.path.exists("./resources/descriptors_and_fingerprints.zip")):
        try:
            RepoRT_directory = glob("./resources/RepoRT_data/*/*.tsv")
            rt_alt_par_list = []
            column_filter = None
            alt_parents_data = pd.read_csv('./resources/RepoRT_classified.tsv', sep='\t', header=0, encoding='utf-8',
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
                return molecule_column_data
            else:
                print(f'No matches found with {pattern}')
        except Exception as e:
            print(f"Error:{e}")

    # Ana añade comentario corto
    if not os.path.exists("./resources/descriptors_and_fingerprints_RepoRT.zip"):
        # TODO: Ana añade funcion de alberto y ligera modificacion final y quitar pass
        pass

    # Ana añade comentario corto
    if not os.path.exists("./resources/descriptors_and_fingerprints_RepoRT_prepared.pklz"):
        data = pd.read_csv("./resources/RepoRT_extracted_data.zip", sep='\t', header=0, encoding='utf-8')
        # TODO: extraer zip
        """
        del archivo  de alberto cojemos descriptors de los standares
        " fingerprints de los standares
        
        dataframe de ./resources/RepoRT_extracted_data.zip coger el inchi, rt e info column y quitar los compuestos de smrt artificiales
        
        le añadimos las columnas de descriptors and fingerprints
        
        y salvamos y devolvemos X, y, descriptor_columns, fingerprints_columns, chromatography_columns = pickle.load(f)
        
        """


        # TODO: eliminar esto cuando hagas lo anterior
        descriptors = pd.read_csv("../resources/des_and_fgp_data/report_unique_inchis_descriptors.csv",
                                  sep=',', header=0, encoding='utf-8')
        fingerprints = pd.read_csv(
            "../resources/des_and_fgp_data/report_unique_inchis_vectorfingerprintsVectorized.csv",
            sep=',', header=0, encoding='utf-8')
        columns_in_data = data.drop(columns="inchi.std").columns
        columns_in_des_and_fgn = [i for i in columns_in_data if i in fingerprints]
        descriptors = descriptors.drop(columns=columns_in_des_and_fgn).drop(columns="column.usp.code")
        fingerprints = fingerprints.drop(columns=columns_in_des_and_fgn).drop(columns="column.usp.code")
        columns_to_drop = ["name", "formula", "smiles.std", "inchikey.std", "classyfire.kingdom",
                           "classyfire.superclass",
                           "classyfire.class", "classyfire.subclass", "classyfire.level5", "classyfire.level6",
                           "alternative_parents", "comment", "column.name"]
        data = data.drop(columns=columns_to_drop)
        descriptors = descriptors[~descriptors["MW"].isnull()]
        des_no_SMRT = pd.merge(data, descriptors, on="inchi.std")
        fgp_no_SMRT = pd.merge(data.iloc[:, :3], fingerprints, on="inchi.std")
        des_no_SMRT.to_csv("../resources/des_no_SMRT.tsv", sep='\t', index=False)
        fgp_no_SMRT.to_csv("../resources/fgp_no_SMRT.tsv", sep='\t', index=False)


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
        with bz2.BZ2File("./resources/descriptors_and_fingerprints_RepoRT_prepared.pklz", "wb") as f:
            pickle.dump([X, y, desc_cols, fgp_cols], f)

    else:
        with bz2.BZ2File("./resources/descriptors_and_fingerprints_RepoRT_prepared.pklz", "rb") as f:
            X, y, descriptor_columns, fingerprints_columns, chromatography_columns = pickle.load(f)


    return X, y, desc_cols, fgp_cols



