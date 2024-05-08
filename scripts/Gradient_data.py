import re
import pandas as pd
from glob import glob
import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def gradient_data(imputation):
    """
    Access to data related to gradient used in chromatography

    This function reads gradient data from TSV files in the '../data/*/' RepoRT_directory,
    concatenates them into a single DataFrame, and merges them with chromatographic column metadata.
    It obtains the maximum and minimum gradients, time intervals, and files to exclude if training is enabled.

    Args:
        imputation(bool): Indicates whether to perform training data processing

    Returns:
        DataFrame: A DataFrame containing processed gradient data merged with chromatographic column metadata
    """
    try:
        RepoRT_directory = glob("./resources/RepoRT_data/*/*.tsv")
        excluded_files = []
        gradient_data_list = []
        drop_file = []
        gradient_time_data = {}
        flowrate_null = {}
        column_data, eluent_data = metadata()
        for file in RepoRT_directory:
            if re.search(r"_gradient.tsv", file):
                gradient_RepoRT_data = pd.read_csv(file, sep='\t', header=0, encoding='utf-8')
                file_name = int(os.path.basename(file)[0:4])
                if gradient_RepoRT_data["t [min]"].isnull().values.any() or gradient_RepoRT_data["t [min]"].values.size == 0:
                    excluded_files.append(f'experiment nº {file_name}')
                    drop_file.append(file_name)
                else:
                    gradient_time_data[file_name] = gradient_RepoRT_data["t [min]"].values.max(), gradient_RepoRT_data.values.shape[0]
                    gradient_RepoRT_data["file"] = file_name
                    gradient_rearrangement = delete_eluent(gradient_RepoRT_data, eluent_data)
                    gradient_processing = pd.DataFrame(pd.concat(gradient_rearrangement)).transpose()
                    column_flowrate = gradient_processing.filter(regex="flow_rate *", axis=1)
                    flowrate_null[gradient_processing.index[0]] = gradient_processing[column_flowrate.columns].isnull().columns
                    gradient_data_list.append(gradient_processing)
        gradient_final_data = pd.concat(gradient_data_list)
        chromatographic_data = pd.merge(column_data, gradient_final_data, left_index=True, right_index=True, how="left")
        data_drop = chromatographic_data.iloc[:, 0:8].isnull().sum(axis=1) > 5
        drop_file.extend(chromatographic_data[data_drop].index.tolist())
        excluded_files.extend([f'experiment nº {i}' for i in chromatographic_data[data_drop].index.tolist()])
        excluded_files = pd.Series(excluded_files).drop_duplicates()
        excluded_files.name = "nº experiments"
        if imputation is True:
            chromatographic_data = processing_data(chromatographic_data, drop_file, flowrate_null)
        df_gradient_time = pd.DataFrame(data=gradient_time_data, index=["t_max", "num"])
        df_gradient_time = df_gradient_time.transpose()
        excluded_files.to_csv("../../excluded_files.tsv", index=False)
        df_gradient_time.to_csv("../../gradient_time_data.tsv", sep="\t", index=True)
        return chromatographic_data
    except Exception as e:
        print(e)

def metadata():
    """
    Access to chromatographic column data

    This function reads chromatographic column metadata from TSV files in the '../data/*/' RepoRT_directory,
    concatenates them into a single DataFrame, and processes the data to ensure that all eluents are in
    the same units (%) and to generate a new column with the number of missing values.

    Returns:
        tuple: A tuple containing two DataFrames:
            - `column_data`: DataFrame containing metadata related to chromatographic columns,
              including column inner diameter, name, length, temperature, etc., and a column indicating
              the number of missing values.
            - `eluent_data`: DataFrame containing metadata related to the eluent used in chromatography.
              This DataFrame excludes unit-related columns and columns related to gradient data
    """
    try:
        RepoRT_directory_met = glob("./resources/RepoRT_data/*/*.tsv")
        chromatography_metadata = []
        for file in RepoRT_directory_met:
            if re.search(r"_metadata.tsv", file):
                metadata_file = pd.read_csv(file, sep='\t', header=0, encoding='utf-8')
                chromatography_metadata.append(metadata_file)
        chromatography_data = pd.concat(chromatography_metadata, ignore_index=True)
        chromatography_data = chromatography_data.set_index("id")
        unit_columns_position = [pos for pos, col in enumerate(chromatography_data.columns) if "unit" in col]
        for position in unit_columns_position:
            if chromatography_data.iloc[:, position].notna().any() and chromatography_data.iloc[:, position].str.contains("mM").any():
                if "nh4ac" in chromatography_data.columns[position - 1]:
                    chromatography_data.iloc[:, position - 1] *= 0.007
                elif "nh4form" in chromatography_data.columns[position - 1]:
                    chromatography_data.iloc[:, position - 1] *= 0.005
                elif "nh4carb" in chromatography_data.columns[position - 1]:
                    chromatography_data.iloc[:, position - 1] *= 0.006
                elif "nh4bicarb" in chromatography_data.columns[position - 1]:
                    chromatography_data.iloc[:, position - 1] *= 0.005
                elif "nh4form" in chromatography_data.columns[position - 1]:
                    chromatography_data.iloc[:, position - 1] *= 0.004
                elif "nh4oh" in chromatography_data.columns[position - 1]:
                    chromatography_data.iloc[:, position - 1] *= 0.004
            elif chromatography_data.iloc[:, position].notna().any() and chromatography_data.iloc[:, position].str.contains("µM").any():
                if "phosphor" in chromatography_data.columns[position - 1]:
                    chromatography_data.iloc[:, position - 1] *= 5.21 / (10 ** 6)
                elif "medronic" in chromatography_data.columns[position - 1]:
                    chromatography_data.iloc[:, position - 1] *= 8.38 / (10 ** 6)
        unit_gradient_columns = [col for col in chromatography_data.columns if '.unit' in col or "gradient." in col]
        column_data = chromatography_data.loc[:, :"eluent.A.h2o":]
        column_data["missing_values"] = column_data.isnull().sum(axis=1)
        eluent_data = chromatography_data.loc[:, "eluent.A.h2o":].drop(columns=unit_gradient_columns)
        return column_data, eluent_data
    except Exception as e:
        print(f"Error metadata:{e}")


def delete_eluent(gradient_data, eluent_data):
    """
    Processes gradient and eluent data.

    This function retains the data for the two most concentrated eluents and removes the remaining ones.

    Args:
        gradient_data (DataFrame): A DataFrame containing gradient data
        eluent_data (DataFrame): A DataFrame containing eluent data

    Returns:
        list: A list containing processed gradient and eluent data from each experiment
    """
    try:
        gradient_data = gradient_data.set_index("file")
        gradient_rearrangement = []
        for position in range(gradient_data.shape[0]):
            gra_row = gradient_data.iloc[position, :]
            sort_gra_row = gra_row.iloc[1:5].sort_values(ascending=False)
            drop_gra_columns = sort_gra_row[2:].index
            gradient = gra_row.drop(drop_gra_columns)
            concat_column_data = pd.concat([eluent_data.loc[gradient_data.index[0]], gradient])
            drop_elu_columns = [i for i in concat_column_data.index if drop_gra_columns[0][0] in i or drop_gra_columns[1][0] in i]
            eluent_gradient_data = concat_column_data.drop(drop_elu_columns)
            for col in eluent_gradient_data.index:
                if f'{gradient.index[1][0]}' in col:
                    eluent_gradient_data = eluent_gradient_data.rename(index={col: f'eluent.1{col[8:]} {position}'})
                elif f'{gradient.index[2][0]}' in col:
                    eluent_gradient_data = eluent_gradient_data.rename(index={col: f'eluent.2{col[8:]} {position}'})
            eluent_gradient_data = eluent_gradient_data.rename(index={f't [min]': f't {position}', f'flow rate [ml/min]': f'flow_rate {position}'})
            gradient_rearrangement.append(eluent_gradient_data)
        return gradient_rearrangement
    except Exception as e:
        print(f"Error delete_eluents: {e}")


def processing_data(chromatographic_data, drop_file, flowrate_null):
    """
    Processes training data.

    This function fills missing values based on related columns means.
    It calculates dead time (t0) value with "column.id", "column.length" and "column.flowrate"
    in those columns where t0 is missing. Finally, it drops specified rows from the DataFrame.

    Args:
        chromatographic_data (DataFrame): DataFrame containing the data.
        drop_file (list): List of index to drop from the DataFrame.
        flowrate_null(dictionary): Dictionary containing index(key) and flow_rate columns(values)
        from each experiment

    Returns:
        DataFrame: Processed DataFrame after filling missing values and updating "column.t0".
    """
    try:
        for column in chromatographic_data.columns[2:8]:
            lines_null = chromatographic_data[chromatographic_data[column].isnull()]
            for column_name in lines_null["column.name"]:
                if pd.notnull(column_name):
                    same_lines = chromatographic_data[chromatographic_data['column.name'] == column_name]
                    mean = same_lines[column].mean()
                    if pd.isnull(mean):
                        same_pattern = chromatographic_data[chromatographic_data['column.name'].fillna('').str.contains(column_name[0:15])]
                        mean = same_pattern[column].mean()
                        if pd.isnull(mean):
                            mean = chromatographic_data[column].mean()
                    chromatographic_data.loc[(chromatographic_data[column].isnull()) & (chromatographic_data['column.name'] == column_name), column] = mean
        for key, values in flowrate_null.items():
            chromatographic_data.loc[key, values] = chromatographic_data.loc[key, "column.flowrate"]
        t0_lines = chromatographic_data[chromatographic_data["column.t0"] == 0]
        new_t0 = 0.66*np.pi*((t0_lines["column.id"]/2)**2)*t0_lines["column.length"]/(t0_lines["column.flowrate"]*10**3)
        chromatographic_data.loc[new_t0.index, "column.t0"] = new_t0
        chromatographic_data = chromatographic_data.drop(index=[i for i in pd.Series(drop_file)])
        encoder = OneHotEncoder()
        one_hot_data = encoder.fit_transform(chromatographic_data[["column.usp.code"]])
        one_hot_df = pd.DataFrame(one_hot_data.toarray(),
                                  columns=encoder.get_feature_names_out(['column.usp.code']))
        position_column_name = chromatographic_data.columns.get_loc("column.name")
        chromatographic_data = pd.concat([chromatographic_data.iloc[:, 0:position_column_name], one_hot_df,
                                          chromatographic_data.iloc[:, position_column_name + 1:]], axis=1)
        return chromatographic_data
    except Exception as e:
        print(f"Error processing_data:{e}")
