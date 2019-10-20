import os
import glob
import math
import datetime as dt
import numpy as np
import pandas as pd
from netCDF4 import Dataset
import matplotlib.pyplot as plt


def mount_patches(data, num_patches=180):
    n_row, n_col = data.shape
    # assuming patches are even for columns and rows...
    div_num = math.floor(math.sqrt(num_patches))
    num_data_rows = math.floor(n_row / div_num)
    num_data_cols = math.floor(n_col / div_num)
    df = pd.DataFrame(data)
    data_patches = []
    for i in range(div_num):
        for j in range(div_num):
            # Get the maximum number of rows and columns
            max_row = min((i*num_data_rows) + num_data_rows, n_row)
            max_col = min((j*num_data_cols) + num_data_cols, n_col)
            # Split the maximum number into patches
            new_df = df.iloc[
                (i*num_data_rows):max_row,
                (j*num_data_cols):max_col
            ]
            data_patches.append(new_df)

    return data_patches


def load_var(var_name=None, file_name=None):
    f = Dataset(file_name, 'r')
    if var_name is not None:
        var = np.ma.filled(f.variables[var_name])
        return var
    else:
        # print(f.variables.keys())
        return list(f.variables.keys())[0]


def subtract_vars(actual_var, past_var):
    actual_var[actual_var <= 0] = 0
    past_var[past_var <= 0] = 0
    return actual_var - past_var


def read_folders(root_folder, output_folder):
    input_folders = glob.glob(os.path.join(
        root_folder, "**/*.nc"), recursive=True)
    output_folders = glob.glob(os.path.join(
        output_folder, "**/*.nc"), recursive=True)
    # Get the filename
    data = load_data({"feat": input_folders})
    out = load_data({"feat": output_folders})
    # Concatenate input and output variables
    full_df = pd.merge(
        data, out,
        on=['date', 'block'],
        how='outer')
    return full_df


def load_data(list_files, lag=False):
    df = pd.DataFrame()

    def parse_date(x):
        return get_fname_date(
            os.path.basename(x).split(".")[0])

    def variable_name(x):
        return os.path.dirname(x).split("/")[-2]

    for _, files in list_files.items():
        sorted_files = sorted(files,
                              key=lambda kv: (variable_name(kv),
                                              parse_date(kv)))
        print(sorted_files)
        for i, day in enumerate(sorted_files):
            if lag:
                prev = sorted_files[i-1]
                if variable_name(day) == variable_name(prev):
                    #print(variable_name(day), variable_name(prev))
                    feat = load_var(file_name=day)
                    # print(day)
                    actual_day = load_var(feat, day)
                    prev_day = load_var(feat, prev)

                    diff = subtract_vars(actual_day, prev_day)
                    data = mount_patches(diff)
                    data = calc_stat(data, prefix=variable_name(day))
                    data["date"] = parse_date(day)
                    df = df.append(data, ignore_index=True)
            else:
                feat = load_var(file_name=day)
                # print(day)
                actual_day = load_var(feat, day)
                actual_day[actual_day <= 0] = 0
                data = mount_patches(actual_day)
                data = calc_stat(data, prefix=variable_name(day))
                data["date"] = parse_date(day)
                df = df.append(data, ignore_index=True)

    return df


def get_fname_date(fname):
    return dt.datetime.strptime(fname, "%B_%Y")


def calc_stat(df_list,
              cmd={
        "mean": np.mean,
        "median": np.median,
        "std": np.std,
        "min": np.min,
        "max": np.max},
        prefix=""):
    df_result = pd.DataFrame(
        columns=["{0}_{1}".format(prefix, c) for c in cmd.keys()])
    for i, sub_df in enumerate(df_list):
        aux = sub_df.values
        values_dict = {}
        for k, v in cmd.items():
            aux01 = v(aux)
            values_dict["{0}_{1}".format(prefix, k)] = aux01
        values_dict["block"] = i
        df_result = df_result.append(values_dict, ignore_index=True)
    return df_result


def main(root_folder, output_folders=[]):
    compiled_df = read_folders(root_folder, output_folders)
    # Interpolate df
    compiled_df = compiled_df.interpolate(method="polynomial", order=2)
    compiled_df.dropna(inplace=True)
    print(compiled_df.head())
    compiled_df.to_csv("aggregated_data_without_lag.csv")


if __name__ == "__main__":
    import locale
    # use German locale; name might vary with platform
    locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')

    main("/home/gctucci/Codes/NASA/ARQUIVOS .NC-20191020T043716Z-001/ARQUIVOS .NC",
         "/home/gctucci/Codes/NASA/ARQUIVOS .NC-20191020T043716Z-001/clorofila")
