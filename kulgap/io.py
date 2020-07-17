import numpy as np
import pandas as pd
import re

from .classes import TreatmentResponseExperiment, CancerModel, TreatmentCondition

def read_pdx_data(file_path):
    """
    Reads in data from a file containing anonymized PDX data

    :param file_path: [string] The path to the data file
    :return [TreatmentResponseExperiment] Containing a CancerModel object for each unique cancer model in the data file
        (e.g., PDX for a single patient, cultures from a single CCL).
    """
    pdx_model_list = []
    df = pd.read_csv(file_path, index_col=0)
    for pname in df.patient.unique()[1:5]:
        new_pdx_model = None
        print(pname)
        df_pat = df[df.patient == pname]
        new_pdx_model = CancerModel(name=pname, tumour_type="no_tumour_type",
                                    start_date=None, treatment_start_date=df_pat.drug_start_day.iloc[0],
                                    end_date=None, treatment_condition_dict={})

        for cname in df_pat.category.unique():
            new_cond = None
            print(cname)
            df_cat = df_pat[df_pat.category == cname]
            level_array = np.array([parse_string_to_ndarray(x) for x in df_cat.day.unique()])
            response_list = []
            for day in df_cat.day.unique():
                df_day = df_cat[df_cat.day == day]
                response_list.append(df_day.volume)
            response_array = np.array(response_list)
            new_cond = TreatmentCondition(cname, source_id=pname, level=level_array,
                                          response=response_array,
                                          replicates=range(response_array.shape[1]),
                                          treatment_start_date=df_cat.drug_start_day.iloc[0],
                                          is_control=df_cat.control.iloc[0] == 1)
            new_cond.measurement_start = df_cat.measurement_start.iloc[0]
            new_cond.measurement_end = df_cat.measurement_end.iloc[0]
            new_cond.x_cut = new_cond.response[new_cond.measurement_start:new_cond.measurement_end + 1]
            new_pdx_model.add_treatment_condition(new_cond)
            del new_cond
        new_pdx_model.normalize_treatment_conditions()
        pdx_model_list.append(new_pdx_model)
        del new_pdx_model
    pdx_experiment = TreatmentResponseExperiment(pdx_model_list)
    return pdx_experiment

def read_cancer_model_from_csv_buffer(file_buffer):
    df = pd.read_csv(file_buffer)

    # -- build the TreatmentCondition objects from the df
    control_response = df.iloc[:, [bool(re.match('Control.*', col)) for col in df.columns]].to_numpy()
    control = TreatmentCondition('Control', source_id='from_webapp',
                                 level=df.Time.to_numpy(), response=control_response,
                                 treatment_start_date=min(df.Time), is_control=True)

    treatment_response = df.iloc[:, [bool(re.match('Control.*', col)) for col in df.columns]].to_numpy()
    treatment = TreatmentCondition('Control', source_id='from_webapp',
                                   level=df.Time.to_numpy(), response=treatment_response,
                                   treatment_start_date=min(df.Time), is_control=False)

    # -- build the CancerModel object from the TreatmentConditions
    treatment_condition_dict = {'Control': control, 'Treatment': treatment}
    cancer_model = CancerModel(name="from_webapp",
                               treatment_condition_dict=treatment_condition_dict,
                               model_type="PDX",
                               tumour_type="unknown",
                               start_date=min(df.Time),
                               treatment_start_date=min(df.Time),
                               end_date=max(df.Time))

    treatment_response_experiment = TreatmentResponseExperiment(cancer_model_list=[cancer_model])
    patient_json = treatment_response_experiment.to_dict(recursive=True)
    stats_json = treatment_response.






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