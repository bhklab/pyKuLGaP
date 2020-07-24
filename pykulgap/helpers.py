import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.integrate import quad


def p_value(l1, l2):
    """
    returns p-value for each variable in l1, based on l2
    :param l1: The list  of variable for which the p-value is to be computed
    :param l2: The list of values on which the p-value calculation is based
    :return: The calculated list of p-values
    """
    pval_list = []
    for y in l1:
        pval_list.append((len([x for x in l2 if x >= y]) + 1) / (len(l2) + 1))
    return pval_list


def get_all_experimental_conditions(treatment_response_experiment):
    """
    Takes a list of patients and returns a dictionary of all categories in the list of patients,

    :param treatment_response_experiment: [TreatmentResponseExperiment] A container holding the set of
        `CancerModel` objects associated with a treatment response experiment.
    :return: [dict] A dictionary of the form {name: experimental_condition}
    """
    experimental_condition_dict = {}
    for _, cancer_model in treatment_response_experiment:
        control = cancer_model.experimental_conditions.get("Control")
        for condition_name, experimental_condition in cancer_model.experimental_conditions:
            if condition_name != "Control":
                experimental_condition_dict[str(cancer_model.name) + "-" + str(condition_name)] = \
                    {"case": experimental_condition, "control": control}
    return experimental_condition_dict


def calculate_AUC(variable, response):
    """
    Calculates the area under the curve of a set of observations 

    :param variable [ndarray] the time points:
    :param response [ndarray] the observations:
    :return [float] The area under the curve:    
    """
    AUC = 0
    min_length = min(len(variable), len(response))
    variable = variable.ravel()
    for j in range(min_length - 1):
        AUC += (response[j + 1] - response[j]) / (variable[j + 1] - variable[j])
    return AUC


def kl_divergence(case, control):
    """
    Calcluates KL divergence between case and control
    :param case: The treatment ExperimentalCondition object
    :param control: The control ExperimentalCondition object
    :return: [float] The KL value
    """

    def kl_integrand(t):
        mean_control, var_control = control.gp.predict(np.asarray([[t]]))
        mean_case, var_case = case.gp.predict(np.asarray([[t]]))
        return ((var_control + (mean_control - mean_case) ** 2) / (2 * var_case)) + (
                (var_case + (mean_case - mean_control) ** 2) / (2 * var_control)) - 1

    max_x_index = min(case.variable_treatment_end_index, control.variable_treatment_end_index)
    if control.response.shape[1] > case.response.shape[1]:
        kl_divergence = abs(1 / (case.variable[max_x_index] - case.variable_treatment_start) *
                            quad(kl_integrand, case.variable_treatment_start, case.variable[max_x_index],
                                 limit=100)[0])[0] / 11
    else:
        kl_divergence = abs(1 / (control.variable[max_x_index] - case.variable_treatment_start) *
                            quad(kl_integrand, case.variable_treatment_start, control.variable[max_x_index],
                                 limit=100)[0])[0] / 11
    return kl_divergence


def cross_kl_divergences(experimental_condition_list):
    """
    takes a list of categories and computes KL(variable,response) for all variable and response in the list
    :param experimental_condition_list: A list of ExperimentalCondition objects
    :return: The list of all KL(variable,response) as variable, response range over cat_list
    """
    kl_list = []
    cl = len(experimental_condition_list)

    for i in range(cl):
        print(f"done {i+1} out of {cl}")
        for j in range(i):
            new_kl = kl_divergence(experimental_condition_list[i], experimental_condition_list[j])
            if (new_kl is not None) and (str(new_kl) != "nan") and str(new_kl) != "inf":
                kl_list.append(new_kl)
    return kl_list


def cv_smoothing(list_to_be_smoothed):
    """
    Computes kernel smoothing for list_to_be_smoothed
    :param list_to_be_smoothed: the list to be smoothed. Needs to be of type numeric.
    
    :return: a KDEMultivariate object, smoothed using leave-one-out cross-validation
    """
    return sm.nonparametric.KDEMultivariate(data=list_to_be_smoothed, var_type="c", bw="cv_ml")


def calculate_null_kl(experimental_condition_list=None, filename=None):
    """
    Calculates the smoothed null KL distribution. One of the two parameters must be non-null
    :param experimental_condition_list: [list] The list of treatment condition from which the null kl is to be calculated.
    :param filename: If None, calculate from category_list. Else read in from file_path
    :return: [list] the list of values and the smoothed object
    """
    if filename is None and experimental_condition_list is not None:
        null_kl_data = cross_kl_divergences(experimental_condition_list)
    elif filename is not None and experimental_condition_list is None:
        null_kl_data = list(pd.read_csv(filename, header=None)[0])
    else:
        raise ValueError("Only one of `filename` or `experimental_condition_list` can be passed as a parameter!")
    if len(null_kl_data) > 1:
        smoothed_null_kl = cv_smoothing(null_kl_data)
    else:
        smoothed_null_kl = None
    return {"list": null_kl_data, "smoothed": smoothed_null_kl}


def dict_to_string(dictionary):
    """
    Write the input dictionary to a string of the form {key:entry,...}
    :param dictionary: A dictionary
    :return: The converted string
    """
    return "_".join([str(key) + ":" + str(value) for key, value in dictionary.items()])


def remove_extremal_nas(response, replacement_value):
    """
    Replace leading and trailing n/a values in the rows of response by replacement_value
    Return the modified response, the start (first measurement) and the end (last measurement) dates

    :param response [ndarray] the array to be modified:
    :param replacement_value [int] The value with which the nas will be replaced:
    :return [tuple] a tuple containing the items:
        - response the modified ndarray
        - first the last occasion of a leading na
        - last the first occasion of a trailing na
    """
    firsts = []
    lasts = []
    for j, response_slice in enumerate(response):
        not_nan_idx = np.where(~np.isnan(response_slice))[0]
        firsts.append(not_nan_idx[0])
        lasts.append(not_nan_idx[-1])

        response[j, :firsts[-1]] = replacement_value
        response[j, lasts[-1] + 1:] = replacement_value
    first = max(firsts)
    last = min(lasts)
    return response, first, last


def forward_fill_nas(arr):
    """
    forward-fills na values in numpy array arr: replaces it by previous valid choice

    :param arr [ndarray] the array to be modified:
    :return [ndarray] the modified array:
    """
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = arr[np.arange(idx.shape[0])[:, None], idx]
    return out


def relativize(response, start):
    """
    Normalises a numpy array to a given start index.
    :param response [ndarray] the array to be normalised:
    :param start [int] the start index:
    :return [ndarray] the modified array:
    """
    return (response / response[start]) - 1


def centre(response, start):
    """
    Subtracts the value at index start from a numpy array
    :param response [ndarray] the array to be modified:
    :param start [int] the index to centre on:
    :return [ndarray] the modified array
    """
    return response - response[start]


def compute_response_angle(variable: object, response: object, start: object) -> object:
    """
    Calculates the response angle for observations response, given time points variable and start point start
    :param variable [ndarray] the time points:
    :param response [ndarray] the observations:
    :param start [umpy array] the start point for the angle computation:
    :return [float] the angle:
    """
    min_length = min(len(variable), len(response))
    model = sm.OLS(response[start:min_length], variable[start:min_length], missing='drop')
    results = model.fit()
    return np.arctan(results.params[0])
