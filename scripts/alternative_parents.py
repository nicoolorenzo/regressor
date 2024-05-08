import pandas as pd
import re
from glob import glob


def alternative_parents():
    """
    Obtains alternative parents data of the molecules in RepoRTs processed data

    This function searches alternative parents data from "all_classified.tsv" that contains molecules from RepoRT
    processed data. It reads the data and matches the InChIKey values in the processed data with the InChIKey
    values in the 'all_classified.tsv' alt_parents_file. It creates a DataFrame with the matched records and saves it as
    'RepoRT_classified.tsv'.

    Returns:
        DataFrame: DataFrame containing the matched records.
    """
    try:
        RepoRT_directory_alt_parents = glob("./resources/RepoRT_data/*/*.tsv")
        rt_data_list = []
        merge_data = []
        for files in RepoRT_directory_alt_parents:
            if re.search(r"_rtdata_canonical_success.tsv", files):
                rt_data_df = pd.read_csv(files, sep='\t', header=0, encoding='utf-8')
                rt_data_list.append(rt_data_df)
        if rt_data_list:
            rt_data_RepoRT = pd.concat(rt_data_list, axis=0, ignore_index=True)
            alt_parents_file = open("../../all_classified.tsv", 'r')
            for i, line in enumerate(alt_parents_file):
                alt_par_lines = line.strip("\n").split("\t")
                matching_query = rt_data_RepoRT[rt_data_RepoRT["inchikey.std"].str.contains(alt_par_lines[0])]
                if not matching_query.empty:
                    matching_query.loc[0:, "inchikey.std"] = alt_par_lines[0]
                    alt_parents_df = (pd.DataFrame(alt_par_lines)).transpose()
                    merge_data.append(pd.merge(matching_query, alt_parents_df, right_on=0, left_on="inchikey.std"))
            alt_parents_RepoRT = pd.concat(merge_data, ignore_index=True)
            alt_parents_RepoRT.to_csv("../resources/RepoRT_classified.tsv", sep="\t", index=False)
            return alt_parents_RepoRT
    except Exception as e:
        print(f"Error: {e}")
