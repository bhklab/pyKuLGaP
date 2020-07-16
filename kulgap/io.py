import numpy as np
import pandas as pd

from .classes import TreatmentResponseExperiment, CancerModel, TreatmentCondition

def read_pdx_data(file_path):
    """
    Reads in data from a file containing anonymized PDX data

    :param file_path: [string] The path to the data file
    :return [TreatmentResponseExperiment] Containing a CancerModel object for each unique cancer model in the data file
        (e.g., PDX for a single patient, cultures from a single CCL).
    """
    patients_list = []
    df = pd.read_csv(file_path, index_col=0)
    for pname in df.patient.unique()[1:5]:
        df_pat = df[df.patient == pname]
        new_patient = CancerModel(pname, tumour_type="no_tumour_type",
                                  start_date=None, drug_start_day=df_pat.drug_start_day.iloc[0],
                                  end_date=None)

        for cname in df_pat.category.unique():
            df_cat = df_pat[df_pat.category == cname]
            x_array = np.array([parse_string_to_ndarray(x) for x in df_cat.day.unique()])
            y_list = []
            for x in df_cat.day.unique():
                df_day = df_cat[df_cat.day == x]
                y_list.append(df_day.volume)
            y_array = np.array(y_list)
            new_cat = TreatmentCondition(cname, source_id=pname, x=x_array, y=y_array,
                                         replicates=range(y_array.shape[1]),
                                         drug_start_day=df_cat.drug_start_day.iloc[0],
                                         is_control=df_cat.control.iloc[0] == 1)

            new_cat.measurement_start = df_cat.measurement_start.iloc[0]
            new_cat.measurement_end = df_cat.measurement_end.iloc[0]
            new_cat.x_cut = new_cat.x[new_cat.measurement_start:new_cat.measurement_end + 1]

            new_patient.categories[cname] = new_cat
        new_patient.normalize_all_categories()
        patients_list.append(new_patient)
        pdx_experiment = TreatmentResponseExperiment(patients_list)

    return pdx_experiment


## TODO:: Build experiment object using pandas groupby statements
def read_to_treat_resp_exp(file_path):
     df = pd.read_csv(file_path, index_col=0)
     def _group_by_category(df):
         return dict(zip(df.category.unique(), [group for _, group in df.groupby(['category'])]))
     cancer_models = dict(zip(df.patient.unique(), df.groupby(['patient']).apply(_group_by_category)))



## -- Local helper methods

def parse_string_to_ndarray(array_as_str):
    """
    Parse a string representation of a numeric array to an ndarray

    :param string the string representation:
    :return [ndarray] the array from the string representation:
    """
    return float(array_as_str.replace("[", "").replace("]", ""))