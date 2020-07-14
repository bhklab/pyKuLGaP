import numpy as np
import pandas as pd

from . import pdxat


def str_to_array(string):
    """
    Converts a string representation of an array back to the array
    :param string the string representation:
    :return [ndarray] the array from the string representation:
    """
    return float(string.replace("[", "").replace("]", ""))


def read_anonymised(filename):
    """
    Reads in data from a file containing anonymised PDX data
    :param filename The name of the file:
    :return [list] A list of Patient objects: 
    """
    patients_list = []
    df = pd.read_csv(filename, index_col=0)
    for pname in df.patient.unique():
        df_pat = df[df.patient == pname]
        new_patient = pdxat.Patient(pname, tumour_type="no_tumour_type",
                                    start_date=None, drug_start_day=df_pat.drug_start_day.iloc[0],
                                    end_date=None)

        for cname in df_pat.category.unique():
            df_cat = df_pat[df_pat.category == cname]
            x_array = np.array([str_to_array(x) for x in df_cat.day.unique()])
            y_list = []
            for x in df_cat.day.unique():
                df_day = df_cat[df_cat.day == x]
                y_list.append(df_day.volume)
            y_array = np.array(y_list)
            new_cat = pdxat.Category(cname, phlc_id=pname, x=x_array, y=y_array,
                                     replicates=range(y_array.shape[1]),
                                     drug_start_day=df_cat.drug_start_day.iloc[0],
                                     is_control=df_cat.control.iloc[0] == 1)

            new_cat.measurement_start = df_cat.measurement_start.iloc[0]
            new_cat.measurement_end = df_cat.measurement_end.iloc[0]
            new_cat.x_cut = new_cat.x[new_cat.measurement_start:new_cat.measurement_end + 1]

            new_patient.categories[cname] = new_cat

        patients_list.append(new_patient)

    return patients_list
