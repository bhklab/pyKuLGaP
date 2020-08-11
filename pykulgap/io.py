import numpy as np
import pandas as pd
import re
import io

from pykulgap.classes import TreatmentResponseExperiment, CancerModel, ExperimentalCondition


def read_pdx_data(file_path):
    """
    Reads in data from a file containing anonymized PDX data

    :param file_path: [string] The path to the data file
    :return [TreatmentResponseExperiment] Containing a CancerModel object for each unique cancer model in the data file
        (e.g., PDX for a single patient, cultures from a single CCL).
    """
    pdx_model_list = []
    df = pd.read_csv(file_path, index_col=0)
    # TODO:: Remove from production
    for pname in df.patient.unique():
        new_pdx_model = None
        print(pname)
        df_pat = df[df.patient == pname]
        new_pdx_model = CancerModel(name=pname, tumour_type="no_tumour_type",
                                    variable_start=None, variable_treatment_start=df_pat.drug_start_day.iloc[0],
                                    variable_end=None, experimental_condition_dict={})

        for cname in df_pat.category.unique():
            new_cond = None
            print(cname)
            df_cat = df_pat[df_pat.category == cname]
            variable_array = np.array([parse_string_to_ndarray(x) for x in df_cat.day.unique()])
            response_list = []
            for day in df_cat.day.unique():
                df_day = df_cat[df_cat.day == day]
                response_list.append(df_day.volume)
            response_array = np.array(response_list)
            new_cond = ExperimentalCondition(cname, source_id=pname, variable=variable_array,
                                             response=response_array,
                                             replicates=range(response_array.shape[1]),
                                             variable_treatment_start=df_cat.drug_start_day.iloc[0],
                                             is_control=df_cat.control.iloc[0] == 1)
            new_cond.variable_treatment_start_index = df_cat.measurement_start.iloc[0]
            new_cond.variable_treatment_end_index = df_cat.measurement_end.iloc[0]
            new_pdx_model.add_experimental_condition(new_cond)
            del new_cond
        new_pdx_model.normalize_experimental_conditions()
        pdx_model_list.append(new_pdx_model)
        del new_pdx_model
    pdx_experiment = TreatmentResponseExperiment(pdx_model_list)
    return pdx_experiment


def read_pdx_from_byte_stream(csv_byte_stream):
    """
    Read in a .csv file from the KuLGaP web application as a byte stream and parses that .csv into a
    `TreatmentResponseExperiment` object. Then computes summary statistics, builds a DataFrame and JSONizes
    the DataFrame, returning the JSON string.

    :param csv_byte_stream [bytes] A byte stream passed from the web application containing the .csv data for
        a PDX experiment.
    :return [TreatmentResponseExperiment] object containing the data from the .csv byte stream
    """
    stream = io.StringIO(csv_byte_stream.decode('utf-8'))
    df = pd.read_csv(stream)

    # -- parse the control and treatment columns to NumPy arrays
    control_response = df.iloc[:, [bool(re.match('Control.*', col)) for col in df.columns]].to_numpy()
    variable = df.Time.to_numpy()
    treatment_response = df.iloc[:, [bool(re.match('Treatment.*', col)) for col in df.columns]].to_numpy()

    if control_response.shape != treatment_response.shape:
        raise ValueError("Oh no! We can't seem to parse your control and treatment columns. Please ensure the correct"
                         "formatting has been used for your .csv file.")

    # -- subset to death of first mouse
    # Determine index of first mouse death to remove all NaNs before fitting the model
    first_death_idx = min(min(np.sum(~np.isnan(control_response), axis=0)),
                          min(np.sum(~np.isnan(treatment_response), axis=0)))

    # Subset the relevant data to first_death_idx
    control_response = control_response[0:first_death_idx, :]
    treatment_response = treatment_response[0:first_death_idx, :]
    variable = variable[0:first_death_idx]

    # -- build the ExperimentalCondition objects from the df
    control = ExperimentalCondition('Control', source_id='from_webapp',
                                    variable=variable, response=control_response,
                                    replicates=list(range(control_response.shape[1])),
                                    variable_treatment_start=min(variable), is_control=True)

    treatment = ExperimentalCondition('Treatment', source_id='from_webapp',
                                      replicates=list(range(treatment_response.shape[1])),
                                      variable=variable, response=treatment_response,
                                      variable_treatment_start=min(variable), is_control=False)

    # -- build the CancerModel object from the TreatmentConditions
    experimental_condition_dict = {'Control': control, 'Treatment': treatment}
    cancer_model = CancerModel(name="from_webapp",
                               experimental_condition_dict=experimental_condition_dict,
                               model_type="PDX",
                               tumour_type="unknown",
                               variable_start=min(df.Time),
                               variable_treatment_start=min(df.Time),
                               variable_end=max(df.Time))

    # -- build the TreatmentResponseExperiment object from the CancerModel
    treatment_response_experiment = TreatmentResponseExperiment(cancer_model_list=[cancer_model])

    # -- fit gaussian process model and calculate statistics
    for model_name, cancer_model in treatment_response_experiment:
        cancer_model.normalize_experimental_conditions()
        cancer_model.fit_all_gps()
        cancer_model.compute_summary_statistics(fit_gp=True)

    return treatment_response_experiment


def byte_stream_to_stats_json(csv_byte_stream, orient='records'):
    """
    Accepts a byte stream of a .csv file where the first column is Time, followed by N columns named Control, then a
    blank column, then N columns named Treatment. Control contains tumour volume (mm^3) values for each control PDX
    mouse model, Treatment contains the same for each PDX mouse model given a specific treatment.

    :param csv_byte_stream: [bytes] A byte stream containing the string representation of a .csv file. The format of
        this .csv file is specified above.
    :param orient: [string] A JSON orientation passed to pandas.DataFrame.to_json as the orient argument. Defaults to
        'records', see Pandas documentation for other options.
    :return [string] A JSON string containing the summary_stats_df data from a TreatmentResponseExperiment built from
        the .csv byte stream.
    """
    treatment_response_experiment = read_pdx_from_byte_stream(csv_byte_stream)
    return treatment_response_experiment.summary_stats_df.to_json(orient=orient)


## -- Local helper methods

def parse_string_to_ndarray(array_as_str):
    """
    Parse a string representation of a numeric array to an ndarray

    :param string the string representation:
    :return [ndarray] the array from the string representation:
    """
    return float(array_as_str.replace("[", "").replace("]", ""))