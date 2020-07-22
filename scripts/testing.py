import numpy as np
import pandas as pd
import json
import re
import os

from kulgap.classes import TreatmentResponseExperiment, CancerModel, TreatmentCondition
from kulgap.io import read_pdx_data, parse_string_to_ndarray
from kulgap.plotting import plot_everything, create_and_plot_agreements, get_classification_df, \
    plot_category, plot_histogram, create_and_plot_FDR, create_and_save_KT, plot_histograms_2c,\
    create_measurement_dict, create_measurement_df

if ('scripts' in os.getcwd()):
    os.chdir('..')
results_path = os.path.join(os.getcwd(), "results")
data_path = os.path.join(os.getcwd(), "data")

file_path = os.path.join(data_path, 'alldata_new.csv')

file_buffer = open(os.path.join(data_path, 'kulgap_webapp_data.csv'), 'r')

df = pd.read_csv(file_buffer)

# -- build the TreatmentCondition objects from the df
control_response = df.iloc[:, [bool(re.match('Control.*', col)) for col in df.columns]].to_numpy()
variable = df.Time.to_numpy()
treatment_response = df.iloc[:, [bool(re.match('Treatment.*', col)) for col in df.columns]].to_numpy()

if control_response.shape != treatment_response.shape:
    raise ValueError("Oh no! We can't seem to parse your control and treatment columns. Please ensure the correct"
                     "formatting has been used for you .csv file.")

# -- subset to death of first mouse
# Determine index of first mouse death to remove all NaNs before fitting the model
first_death_idx = min(min(np.sum(~np.isnan(control_response), axis=0)),
                      min(np.sum(~np.isnan(treatment_response), axis=0)))

control_response = control_response[0:first_death_idx, :]
treatment_response = treatment_response[0:first_death_idx, :]
variable = variable[0:first_death_idx]

control = TreatmentCondition('Control', source_id='from_webapp',
                             variable=variable, response=control_response,
                             replicates=list(range(control_response.shape[1])),
                             variable_treatment_start=min(variable), is_control=True)

treatment = TreatmentCondition('Treatment', source_id='from_webapp',
                               replicates=list(range(treatment_response.shape[1])),
                               variable=variable, response=treatment_response,
                               variable_treatment_start=min(variable), is_control=False)

# -- build the CancerModel object from the TreatmentConditions
treatment_condition_dict = {'Control': control, 'Treatment': treatment}
cancer_model = CancerModel(name="from_webapp",
                           treatment_condition_dict=treatment_condition_dict,
                           model_type="PDX",
                           tumour_type="unknown",
                           variable_start=min(df.Time),
                           variable_treatment_start=min(df.Time),
                           variable_end=max(df.Time))

# -- build the TreatmentResponseExperiment object from the CancerModel
treatment_response_experiment = TreatmentResponseExperiment(cancer_model_list=[cancer_model])

# -- fit gaussian process models and calculate statistics
for model_name, cancer_model in treatment_response_experiment:
    cancer_model.normalize_treatment_conditions()
    cancer_model.fit_all_gps()
    cancer_model.compute_other_measures(fit_gps=True)

# -- extract summary statistics and dump to json
#stats_json = pd.DataFrame.from_dict(
create_measurement_dict(treatment_response_experiment.cancer_models)
#).transpose().to_json()