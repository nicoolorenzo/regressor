import pandas as pd
from glob import glob
import numpy as np
from scripts import Gradient_data
import re
from formula_validation.Formula import Formula


def access_data(pattern="", location=".*", imputation=True):
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
    """
    try:
        RepoRT_directory = glob("./resources/RepoRT_data/*/*.tsv")
        rt_alt_par_list = []
        column_filter = None
        alt_parents_data = pd.read_csv('./resources/RepoRT_classified.tsv', sep='\t', header=0, encoding='utf-8', dtype=object)
        for file in RepoRT_directory:
            if re.search(r"_rtdata_canonical_success", file):
                RepoRT_rtdata = pd.read_csv(file, sep='\t', header=0, encoding='utf-8')
                column_filter = RepoRT_rtdata.filter(regex=f'{location}', axis=1)
                column_filter = column_filter.select_dtypes(include=['object'])
                for column in column_filter.columns:
                    matching_pattern = RepoRT_rtdata[column_filter[column].str.lower().str.contains(pattern.lower(), na=False)]
                    if not matching_pattern.empty:
                        merge_rt_alt_parents = matching_pattern.merge(alt_parents_data.drop(columns=[col for col in matching_pattern.columns[1:]] + ["0"]),
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
            molecule_column_data = pd.merge(molecule_data, column_data, left_index=True, right_index=True, how="inner")
            molecule_column_data = molecule_column_data.fillna(0)
            return molecule_column_data
        else:
            print(f'No matches found with {pattern}')
    except Exception as e:
        print(f"Error:{e}")
