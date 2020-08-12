import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from math import atan
from math import pi
import statsmodels.formula.api as smf
import statsmodels.api as sm
from matplotlib import pylab as pl, patches as mp
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from scipy.stats import mannwhitneyu
from sklearn.metrics import accuracy_score

from .helpers import dict_to_string, calculate_null_kl

sns.set(style="ticks")


def create_measurement_dict(all_models, kl_null_filename=None):
    """
    Creates a dictionary of measurements from a list of CancerModel objects.
    The keys of the measurement dictionary are the experiments, the corresponding value
    is a dictionary whose keys are the names of the measurements and ve values the corresponding

    values of the measurement for that experiment.
    :param all_models: A list of CancerModel objects
    :param kl_null_filename: Name of the file from which the KL null distribution is read
    :return:  [dict] The dictionary of measurements
    """
    stats_dict = {}
    if kl_null_filename is not None:
        kl_control_vs_control = calculate_null_kl(filename=kl_null_filename)
    else:
        kl_control_vs_control = calculate_null_kl(experimental_condition_list=[treatment_cond for _, model in all_models
                                                                            for _, treatment_cond in model])

    for name, cancer_model in all_models:
        control = cancer_model._CancerModel__experimental_conditions.get('Control')
        control.calculate_mrecist()
        control.fit_linear_models()

        for experimental_condition in cancer_model.condition_names:
            if 'Control' not in experimental_condition:
                cur_case = cancer_model._CancerModel__experimental_conditions.get(experimental_condition)
                key = str(cur_case.source_id) + "*" + str(experimental_condition)
                stats_dict[key] = {'tumour_type': cancer_model.tumour_type,
                                   'mRECIST': None,
                                   'mRECIST_control': None,
                                   'best_avg_response': None,
                                   'best_avg_response_control': None,
                                   'lm_slopes': None,
                                   'num_mCR': None,
                                   'num_mPR': None,
                                   'num_mSD': None, 'num_mPD': None,
                                   'perc_mCR': None, 'perc_mPR': None,
                                   'perc_mSD': None, 'perc_mPD': None,
                                   'drug': None,
                                   'response_angle': None, 'response_angle_control': None,
                                   'perc_true_credible_intervals': None,
                                   'delta_log_likelihood': None,
                                   'kl': None, 'kl_p_value': None, 'kl_p_cvsc': None, 'gp_deriv': None,
                                   'gp_deriv_control': None, 'auc': None,
                                   'auc_control_norm': None, 'auc_norm': None, 'auc_control': None, 'auc_gp': None,
                                   'auc_gp_control': None,
                                   'number_replicates': len(cur_case.replicates),
                                   'number_replicates_control': len(control.replicates),
                                   "tgi": cur_case.tgi}
                stats_dict[key]['drug'] = experimental_condition

                try:
                    cur_case.calculate_mrecist()
                    cur_case.enumerate_mrecist()
                    cur_case.fit_linear_models()
                except Exception as e:
                    print(e)
                    continue

                if cur_case.kl_divergence is not None:
                    cur_case.kl_p_value = (len([x for x in kl_control_vs_control["list"] if
                                                x >= cur_case.kl_divergence]) + 1) / (
                                                  len(kl_control_vs_control["list"]) + 1)

                    if kl_control_vs_control["smoothed"] is not None:
                        cur_case.kl_p_cvsc = 1 - kl_control_vs_control["smoothed"].cdf([cur_case.kl_divergence])
                    else:
                        cur_case.kl_p_cvsc = None

                num_replicates = len(cur_case.replicates)
                stats_dict[key]['mRECIST'] = cur_case.mrecist
                stats_dict[key]['mRECIST_control'] = control.mrecist
                stats_dict[key]['best_avg_response'] = cur_case.best_avg_response
                stats_dict[key]['best_avg_response_control'] = control.best_avg_response
                stats_dict[key]['lm_slopes'] = cur_case.calculate_lm_slopes()
                stats_dict[key]['lm_slopes_control'] = control.calculate_lm_slopes()

                stats_dict[key]['num_mCR'] = cur_case.mrecist_counts['mCR']
                stats_dict[key]['num_mPR'] = cur_case.mrecist_counts['mPR']
                stats_dict[key]['num_mSD'] = cur_case.mrecist_counts['mSD']
                stats_dict[key]['num_mPD'] = cur_case.mrecist_counts['mPD']
                stats_dict[key]['perc_mCR'] = cur_case.mrecist_counts['mCR'] / num_replicates
                stats_dict[key]['perc_mPR'] = cur_case.mrecist_counts['mPR'] / num_replicates
                stats_dict[key]['perc_mSD'] = cur_case.mrecist_counts['mSD'] / num_replicates
                stats_dict[key]['perc_mPD'] = cur_case.mrecist_counts['mPD'] / num_replicates

                stats_dict[key]['perc_true_credible_intervals'] = cur_case.percent_credible_intervals
                stats_dict[key]['delta_log_likelihood'] = cur_case.delta_log_likelihood_h0_h1
                stats_dict[key]['kl'] = cur_case.kl_divergence
                stats_dict[key]['kl_p_value'] = cur_case.kl_p_value
                stats_dict[key]['kl_p_cvsc'] = cur_case.kl_p_cvsc
                stats_dict[key]['gp_deriv'] = np.nanmean(cur_case.rates_list)
                stats_dict[key]['gp_deriv_control'] = np.nanmean(cur_case.rates_list_control)

                stats_dict[key]['auc'] = cur_case.auc
                stats_dict[key]['auc_norm'] = dict_to_string(cur_case.auc_norm)
                stats_dict[key]['auc_control'] = dict_to_string(cur_case.auc_control)
                stats_dict[key]['auc_control_norm'] = dict_to_string(cur_case.auc_control_norm)
                try:
                    stats_dict[key]['auc_gp'] = cur_case.auc_gp[0]
                    stats_dict[key]['auc_gp_control'] = cur_case.auc_gp_control[0]
                except TypeError:
                    stats_dict[key]['auc_gp'] = ""
                    stats_dict[key]['auc_gp_control'] = ""

                stats_dict[key]['response_angle'] = dict_to_string(cur_case.response_angle)
                stats_dict[key]['response_angle_rel'] = dict_to_string(cur_case.response_angle_rel)
                stats_dict[key]['response_angle_control'] = dict_to_string(cur_case.response_angle_control)
                stats_dict[key]['response_angle_rel_control'] = dict_to_string(cur_case.response_angle_rel_control)

                stats_dict[key]['average_angle'] = cur_case.average_angle
                stats_dict[key]['average_angle_rel'] = cur_case.average_angle_rel
                stats_dict[key]['average_angle_control'] = cur_case.average_angle_control
                stats_dict[key]['average_angle_rel_control'] = cur_case.average_angle_rel_control
    return stats_dict


def create_measurement_df(all_cancer_models):
    """
    Creates a DataFrame of measurements from a list of CancerModel objects.
    One row per experiment, one column per measurement.
    Wraps the response of create_measurement as a DataFrame
    :param all_cancer_models: The list of CancerModel objects
    :return:  [DataFrame] The DataFrame of measurements
    """
    stats_dict = create_measurement_dict(all_cancer_models)
    return pd.DataFrame.from_dict(stats_dict).transpose()


def plusnone(a, b):
    """
    Add a and b, returning None if either of them is None
    :param a: The first summand
    :param b: The second summand
    :return: The sum
    """
    if (a is None) or (b is None):
        return None
    return a + b


def dictvals(dictionary):
    """
    Returns the list of elements in a dictionary, unpacking them if they are inside a list
    :param dictionary: the dictionary to be unpacked
    :returns :[list]
    """
    try:
        return [x[0] for x in dictionary.values()]
    except IndexError:
        return list(dictionary.values())
    except TypeError:
        return list(dictionary.values())


def bts(boolean, y="Y", n="N"):
    """
    Converts a boolean value to a string
    :param boolean: The boolean to be converted
    :param y: [string] the value to be returned if boolean is True
    :param n: [string] the value to be returned if boolean is False
    :return [string]:    
    """
    if boolean:
        return y
    return n


def tsmaller(v1, v2, y="Y", n="N", na="N/a"):
    """
    Compares v1 with v2. Returns the value of response if v1 is smaller than v2 and the value of n
    otherwise. Returns na if either of v1 or v2 is None
    :param v1: the first value of the comparison
    :param v2: the first value of the comparison 
    :param y: the value to be returned if v1<v2
    :param n: the value to be returned if v1>=v2
    :param na: the value to be returned if either v1 or v2 is None.
    """
    if (v1 is not None) and (v2 is not None):
        return bts(v1 < v2, y=y, n=n)
    return na


def mw_letter(d1, d2, pval=0.05, y="Y", n="N", na=None):
    """
    Mann-Whitney U test on d1, d2.
    :param d1: The first list or dict-object to be compared
    :param d2: The second list or dict-object to be compared  
    :param pval: The p-value to be used
    :param y: Returned if the test is significant
    :param n: Returned if the test is not significant
    :param na: Returned if the test fails
    """
    l1 = dictvals(d1)
    l2 = dictvals(d2)
    try:
        return bts(mannwhitneyu(l1, l2).pvalue < pval, y=y, n=n)
    except ValueError as e:
        if na is None:
            return str(e)
        return na


def mw_letter_from_strings(s1, s2, pval=0.05, y="Y", n="N", na=None):
    """
    Turn strings s1, s2 into dictionaries, then apply Mann-Whitney test as in mw_letter
    :param s1: The first string to be compared
    :param s2: The second string to be compared  
    :param pval: The p-value to be used
    :param y: Returned if the test is significant
    :param n: Returned if the test is not significant
    :param na: Returned if the test fails
    """
    ## TODO:: Determine why there are empty strings being returned in ...angle_rel_control
    if ("nan" == str(s1)) or ("nan" == str(s2)) or not s1 or not s2: # Add conditions to deal with empty strings
        if na is None:
            return "no value"
        return na
    return mw_letter(dict_from_string(s1), dict_from_string(s2), pval, y, n, na)


def dict_from_string(s):
    """
    Inverse of dict_to_string. Takes the string representation of a dictionary and returns
    the original dictionary.
    :param s: The string representation of the dictionary
    :return: [dict] the dictionary
    """
    l = s.replace("[", "").replace("]", "").split("_")
    d = {x.split(":")[0]: float(x.split(":")[1]) for x in l}
    return d


def pointwise_kl(case, control, t):
    """
    Calculates the point-wise KL divergence between case and control at time t
    :param case: The treatment ExperimentalCondition
    :param controL: The control ExperimentalCondition
    :param t: The time point
    :return: [float] The KL value.
    """
    mean_control, var_control = control.gp.predict(np.asarray([[t]]))
    mean_case, var_case = case.gp.predict(np.asarray([[t]]))
    return ((var_control + (mean_control - mean_case) ** 2) / (2 * var_case)) + (
            (var_case + (mean_case - mean_control) ** 2) / (2 * var_control))


def p_value(y, l2):
    """
    returns p-value for each response based on l2
    :param y: The value for which the p-value is to be computed
    :param l2: The list of values on which the p-value calculation is based
    :return: The calculated p-value
    """
    return (len([x for x in l2 if x >= y]) + 1) / (len(l2) + 1)


def find_start_end(case, control):
    """
    Find the measurement start and end of a control, treatment pair.
    :param case: The treatment ExperimentalCondition
    :param controL: The control ExperimentalCondition
    :return a [tuple]:
        - the start index point
        - the end index point
    """
    if control is None:
        start = case.find_start_date_index()
        end = case.measurement_end
    else:
        start = max(case.find_start_date_index(), control.measurement_start)
        end = min(case.measurement_end, control.measurement_end)

    return start, end


def logna(x):
    """
    Calcluate the log of variable except return 0 if variable is None
    :param x: the input value
    :return: the log or 0.
    """
    if x is None:
        return 0
    return np.log(x)


def plot_gp(case, control, savename):
    """
    Plots a GP fitted to a treatment and control pair.
    :param case: The treatment ExperimentalCondition
    :param controL: The control ExperimentalCondition
    :param savename: name under which the plot will be saved.
    """
    start, end = find_start_end(case, control)
    plot_limits = [case.variable[start][0], case.variable[end - 1][0] + 1]
    fig, ax = plt.subplots()

    plt.title("GP fits")
    plt.xlim(*plot_limits)
    plt.ylim([0, 3.75])

    plt.xlabel("Time since start of experiment (days)")
    plt.ylabel("Log-normalized tumor size")

    control.gp.plot_data(ax=ax, color="blue")
    control.gp.plot_mean(ax=ax, color="blue", plot_limits=plot_limits, label="Control mean")
    control.gp.plot_confidence(ax=ax, color="blue", plot_limits=plot_limits, label="Control confidence")

    case.gp.plot_data(ax=ax, color="red")
    case.gp.plot_mean(ax=ax, color="red", plot_limits=plot_limits, label="Treatment mean")
    case.gp.plot_confidence(ax=ax, color="red", plot_limits=plot_limits, label="Treatment confidence")
    plt.savefig(savename)


def plot_experimental_condition(case, control, means=None, savename="figure.pdf", normalised=True):
    """
    Fully plot a category
    :param case: the category to be plotted. Not allowed to be None
    :param control : the corresponding control to be plotted. Can be None
    :paran mean:  whether the mean values across replicates are also plotted. Can be None
        (mean will not be plotted), "both" (mean is overlayed) or "only" 
        (only mean is plotted)
    :param savename: The file name under which the figure will be saved.
    :param normalised: If true, plots the normalised versions (case.response_norm). Otherwise case.response
    :return [Figure]: The figure showing the plot
    """
    case_y = case.response_norm if normalised else case.y

    if means not in [None, "only", "both"]:
        raise ValueError("means must be None, 'only', or 'both'")

    start, end = find_start_end(case, control)
    if control is None:
        #        start,end = case.find_start_date_index()
        #        end = case.measurement_end
        high = case_y[:, start:end].max()
    else:
        control_y = control.response_norm if normalised else control.y
        high = max(case_y[:, start:end].max(), control_y[:, start:end].max())
    low = min(case_y[:, start:end].min() * 10, 0)
    fig = plt.figure()
    plt.ylim(low, high * 1.05)
    plt.xlabel("Time since start of experiment (days)")
    if normalised:
        plt.ylabel("Log-normalized tumor size")
    else:
        plt.ylabel("Tumor size (mm3)")
    if means is None:
        plt.title("Replicates")
    elif means == "both":
        plt.title("Replicates and mean")
    else:
        plt.title("Means")
    if means != "only":
        if case is not None:
            for (j, y_slice) in enumerate(case_y):
                if j == 1:
                    s = "treatment"
                else:
                    s = "_treatment"
                plt.plot(case.variable[start:end], y_slice[start:end], '.r-', label=s)
        if control is not None:
            for j, y_slice in enumerate(control_y):
                if j == 1:
                    s = "control"
                else:
                    s = "_control"
                plt.plot(control.variable[start:end], y_slice[start:end], '.b-', label=s)
    if means is not None:
        if means == "both":
            scase = ".k-"
            scontrol = ".k-"
        else:
            scase = ".r-"
            scontrol = ".b-"
        plt.plot(case.variable[start:end], case_y.mean(axis=0)[start:end], scase, label="treatment")
        plt.plot(control.variable[start:end], control_y.mean(axis=0)[start:end], scontrol, label="control")
    fig.legend(loc='upper left', bbox_to_anchor=(0.125, .875))  # loc="upperleft"
    #    fig.legend(loc=(0,0),ncol=2)#"upper left")
    fig.savefig(savename)
    return fig


def plot_everything(outname, treatment_response_expt, ag_df, kl_null_filename, fit_gp=True, p_val=0.05, p_val_kl=0.05,
                    tgi_thresh=0.6):
    """
    Plot a long PDF, one page per cancer_model in all_cancer_models
    :param outname: The name under which the PDF will be saved
    :param treatment_response_expt: list of CancerModel objects to be plotted
    :param ag_df: corresponding DataFrame of binary classifiers
    :param kl_null_filename: Filename from which the KL null is read
    :param fit_gp: whether a GP was fitted
    :param p_val: the p-value
    :param p_val_kl: The p-value for the KuLGaP calculation
    :param tgi_thresh: The threshold for calling a TGI response.
    """
    stats_df = treatment_response_expt.summary_stats_df.copy()
    with PdfPages(outname) as pdf:
        for model_name, cancer_model in treatment_response_expt:
            control = cancer_model["Control"]
            for condition_name, treatment_cond in cancer_model:
                if condition_name != "Control":
                    # TO ADD: SHOULD START ALSO CONTAIN control.measurement_start?!?
                    start = max(treatment_cond.find_variable_start_index(), treatment_cond.variable_start_index)
                    end = min(treatment_cond.variable_end_index, control.variable_end_index)
                    name = str(cancer_model.name) + "*" + str(condition_name)

                    fig, axes = plt.subplots(4, 2, figsize=(32, 18))
                    fig.suptitle(name, fontsize="x-large")
                    axes[0, 0].set_title("Replicates")

                    print("Now plotting cancer_model", name)
                    for response_slice in treatment_cond.response_norm:
                        axes[0, 0].plot(treatment_cond.variable[start:end], response_slice[start:end], '.r-')

                    if control.response_norm is None:
                        print(f"No control for cancer_model {cancer_model.name}, category {condition_name}")
                        print(cancer_model)
                        print('----')
                    else:
                        for response_slice in control.response_norm:
                            axes[0, 0].plot(control.variable[start:end], response_slice[start:end], '.b-')

                    axes[1, 0].set_title("Means")
                    axes[1, 0].plot(treatment_cond.variable[start:end],
                                    treatment_cond.response_norm.mean(axis=0)[start:end], '.r-')
                    if control.response_norm is not None:
                        axes[1, 0].plot(control.variable[start:end],
                                        control.response_norm.mean(axis=0)[start:end], '.b-')

                    axes[1, 1].set_title("Pointwise KL divergence")

                    if fit_gp:
                        axes[1, 1].plot(treatment_cond.variable[start:end + 1].ravel(),
                                        [pointwise_kl(treatment_cond, control, t).ravel()[0] for t in
                                         treatment_cond.variable[start:end + 1].ravel()], 'ro')
                    else:
                        axes[1, 1].axis("off")
                        axes[1, 1].text(0.05, 0.3, "no GP fitting, hence no KL values")
                    axes[2, 0].set_title("GP plot: case")
                    axes[2, 1].set_title("GP plot: control")
                    if fit_gp:
                        treatment_cond.gp.plot(ax=axes[2, 0])
                        pl.show()
                        control.gp.plot(ax=axes[2, 1])
                        pl.show()
                    else:
                        for axis in [axes[2, 0], axes[2, 1]]:
                            axis.text(0.05, 0.3, "not currently plotting GP fits")

                    axes[3, 0].axis("off")
                    axes[3, 1].axis('off')
                    txt = []
                    mrlist = [str(stats_df.loc[name, mr]) for mr in ["num_mCR", "num_mPR", "num_mSD", "num_mPD"]]
                    txt.append("mRECIST: (" + ",".join(mrlist))
                    for col in ["kl", "response_angle_rel", "response_angle_rel_control", "auc_norm",
                                "auc_control_norm", "tgi"]:
                        txt.append(col + ": " + str(stats_df.loc[name, col]))

                    # TO ADD: MAYBE BETTER AGGREGATE DATA?
                    txt.append("red = treatment,       blue=control")
                    axes[3, 0].text(0.05, 0.3, '\n'.join(txt))

                    axes[0, 1].axis("off")
                    rtl = ["KuLGaP: " + bts(treatment_cond.kl_p_cvsc < p_val),
                           "mRECIST (Novartis): " + tsmaller(stats_df.loc[name, "perc_mPD"], 0.5),
                           "mRECIST (ours): " + tsmaller(
                               plusnone(stats_df.loc[name, "perc_mPD"], stats_df.loc[name, "perc_mSD"]), 0.5),
                           "Angle: " + mw_letter(treatment_cond.response_angle_rel,
                                                 treatment_cond.response_angle_rel_control,
                                                 pval=p_val),
                           "AUC: " + mw_letter(treatment_cond.auc_norm, treatment_cond.auc_control_norm, pval=p_val),
                           "TGI: " + tsmaller(tgi_thresh, treatment_cond.tgi)]

                    #                    not yet implemented" )
                    # TO ADD: TGI
                    resp_text = "\n".join(rtl)
                    axes[0, 1].text(0.05, 0.3, resp_text, fontsize=20)

                    pdf.savefig(fig)


def get_classification_df(stats_df, p_val=0.05, p_val_kl=0.05, tgi_thresh=0.6):
    """
    Computes the DF of classifications (which measures call a Responder) from the continuous statistics
    :param stats_df: corresponding DataFrame of continuous statisitics
    :param p_val: the p-value for the angle and AUC tests
    :param p_val_kl: The p-value for the KuLGaP calculation
    :param tgi_thresh: The threshold for calling a TGI response.    
    :return:
    """
    responses = stats_df[["kl"]].copy()

    responses["pykulgap"] = stats_df.kl_p_cvsc.apply(tsmaller, v2=p_val, y=1, n=-1, na=0)
    responses["mRECIST-Novartis"] = stats_df.perc_mPD.apply(tsmaller, v2=0.5, y=1, n=-1, na=0)

    responses["Angle"] = stats_df.apply(
        lambda row: mw_letter_from_strings(row["response_angle_rel"], row["response_angle_rel_control"], pval=p_val,
                                           y=1, n=-1, na=0), axis=1)
    responses["AUC"] = stats_df.apply(
        lambda row: mw_letter_from_strings(row["auc_norm"], row["auc_control_norm"], pval=p_val, y=1, n=-1, na=0),
        axis=1)
    responses["TGI"] = stats_df.tgi.apply(lambda x: tsmaller(tgi_thresh, x, y=1, n=-1, na=0))
    responses.drop("kl", axis=1, inplace=True)
    return responses


def get_classification_dict_with_patients(all_cancer_models, stats_df, p_val, all_kl, p_val_kl, tgi_thresh):
    """
    Return the responses (responder/non-responder calls) as a dictionary, using the list of patients
    rather than the DataFrame input
    :param all_cancer_models: list of CancerModel objects
    :param stats_df: corresponding DataFrame of continuous statistics
    :param p_val: the p-value
    :param all_kl: The list of KL null values
    :param p_val_kl: The p-value for the KuLGaP calculation
    :param tgi_thresh: The threshold for calling a TGI response.    

    :return: a dictionary of lists of calls (values) for each classifier (keys)
    """
    predict = {"pykulgap": [], "AUC": [], "Angle": [], "mRECIST_Novartis": [], "mRECIST_ours": [],
               "TGI": []}
    for model_name, cancer_model in all_cancer_models:
        for condition_name, treatment_cond in cancer_model.experimental_conditions:
            if condition_name != "Control":
                name = str(cancer_model.name) + "*" + str(condition_name)
                predict["pykulgap"].append(tsmaller(p_value(treatment_cond.kl_divergence, all_kl), p_val_kl, y=1, n=-1, na=0))
                predict["mRECIST_Novartis"].append(tsmaller(stats_df.loc[name, "perc_mPD"], 0.5, y=1, n=-1, na=0))
                predict["mRECIST_ours"].append(
                    tsmaller(plusnone(stats_df.loc[name, "perc_mPD"], stats_df.loc[name, "perc_mSD"]), 0.5, y=1, n=-1,
                             na=0))
                predict["Angle"].append(
                    mw_letter(treatment_cond.response_angle_rel, treatment_cond.response_angle_rel_control, pval=p_val, y=1, n=-1,
                              na=0))
                predict["AUC"].append(
                    mw_letter(treatment_cond.auc_norm, treatment_cond.auc_control_norm, pval=p_val, y=1, n=-1, na=0))
                predict["TGI"].append(tsmaller(tgi_thresh, treatment_cond.tgi, y=1, n=0, na=2))
    return predict


def create_and_plot_agreements(classifiers_df, agreements_outfigname, agreements_outname):
    """
    Creates and plots the agreement matrix between measures
    :param classifiers_df: The DataFrame of responder calls
    :param agreements_outfigname: Name under which the figure will be saved
    :param agreements_outname: Name under which the data will be saved.
    """
    agreements = create_agreements(classifiers_df)
    agreements.to_csv(agreements_outname)
    paper_list = ["pykulgap", "TGI", "mRECIST", "AUC", "Angle"]
    ag2 = agreements[paper_list].reindex(paper_list)
    print(ag2)
    plt.figure()
    sns.heatmap(ag2, vmin=0, vmax=1, center=.5, square=True, annot=ag2, cbar=False, linewidths=.3, linecolor="k",
                cmap="Greens")
    #    sns.heatmap(agreements, vmin=0, vmax=1, center=0,square=True,annot=agreements,cbar=False)
    plt.savefig(agreements_outfigname)


def create_and_plot_FDR(classifiers_df, FDR_outfigname, FDR_outname):
    """
    Creates the false discovery matrix and then plots it
    :param classifiers_df: The DataFrame of responder calls
    :param FDR_outfigname: Name under which the figure will be saved
    :param FDR_outname: Name under which the data will be saved.
    """
    FDR = create_FDR(classifiers_df)
    FDR.to_csv(FDR_outname)
    paper_list = ["pykulgap", "TGI", "mRECIST", "AUC", "Angle"]
    FDR = FDR[paper_list].reindex(paper_list)
    plt.figure()
    sns.heatmap(FDR, cmap="coolwarm", square=True, annot=FDR,
                cbar=False, linewidths=.3, linecolor="k", vmin=-.8, vmax=.8, center=-0.1)
    plt.savefig(FDR_outfigname)


def create_and_save_KT(classifiers_df, KT_outname):
    """
    Creates and saves the matrix of Kendall Tau tests between the responder calls
    :param classifiers_df: The DataFrame of responder calls
    :param KT_outname: The name under which the data will be saved
    """
    kts = create_KT(classifiers_df)
    print(kts)
    kts.to_csv(KT_outname)


def plot_histogram(list_to_be_plotted, varname, marked=None, savename=None, smoothed=None, x_min=None, x_max=None,
                   dashed=None,
                   solid=None):
    """
    Plots the histogram of list_to_be_plotted, with an asterix and an arrow at marked
    Labels the variable axis according to varname
    :param list_to_be_plotted: The list to be plotted
    :param varname: The label for the variable-axis
    :param marked: Where the arrow is to appear
    :param savename: Filename under which the figure will be saved
    :param smoothed: Either none or a smoothed object
    :param x_min: The left end point of the range of variable-values
    :param x_max: The right end point of the range of variable-values
    :param dashed: Where to draw a vertical dashed line
    :param solid: Where to draw a vertical solid line
    :return:
    """
    fig = plt.figure()
    var = pd.Series(l)
    var.dropna().hist(bins=30, grid=False, density=True)
    if smoothed is not None:
        x = np.linspace(x_min, x_max, 1000)
        plt.plot(x, smoothed(x), "-r")
    plt.xlabel(varname)
    plt.ylabel("frequency")
    if marked is not None:
        plt.plot(marked, .02, marker="*", c="r")
        style = "Simple,tail_width=0.5,head_width=4,head_length=8"
        kw = dict(arrowstyle=style, color="k")
        plt.text(11, .2, "critical value")
        arrow = mp.FancyArrowPatch(posA=[11, .2], posB=[marked + .25, 0.035], connectionstyle="arc3,rad=-.25", **kw)
        plt.gca().add_patch(arrow)
    if dashed is not None:
        for val in dashed:
            ax = plt.gca()
            ax.axvline(x=val, color='black', linestyle="--")
    if solid is not None:
        for val in solid:
            ax = plt.gca()
            ax.axvline(x=val, color='black', linestyle="-")

    plt.savefig(savename)
    return fig


def create_scatterplot(stats_df, classifiers_df, savename):
    """
    Creates a scatterplot of all experiments, plotting the number of measures agreeing on 
    a responder label against the logarithm of the KL divergenc.
    Not used in the paper
    :param stats_df: [DataFrame] The raw values of the statistics
    :param classifiers_df: [DataFrame] The binary values (1/0) of the measures
    :param savename: The name under which the figure is saved.
    """

    df = stats_df[["kl"]]
    df.loc[:, "kl_p"] = stats_df.kl_p_cvsc
    df.loc[:, "Ys"] = classifiers_df.drop("pykulgap", axis=1).apply(lambda row: row[row == 1].count(), axis=1)

    plt.figure()
    plt.ylim(0, 5)
    plt.plot(df.kl.apply(logna), df.Ys, 'r', marker=".", markersize=2, linestyle="")
    c = np.log(7.97)
    plt.plot([c, c], [0, 5], 'k-', lw=1)
    c = np.log(5.61)
    plt.plot([c, c], [0, 5], 'k--', lw=1)
    c = np.log(13.9)
    plt.plot([c, c], [0, 5], 'k--', lw=1)
    plt.xlabel("Log(KL)")
    plt.ylabel('Number of measures that agree on a "responder" label')
    plt.ylim(-0.2, 4.2)
    plt.yticks(ticks=[0, 1, 2, 3, 4])
    plt.savefig(savename)


def plot_histograms_2c(stats_df, classifiers_df, savename):
    """
    Plots Figure 2C in the paper.
    :param stats_df: [DataFrame] The raw values of the statistics
    :param classifiers_df: [DataFrame] The binary values (1/0) of the measures
    :param savename: The name under which the figure is saved.
    """
    data = stats_df[["kl"]].copy()
    data.loc[:, "klval"] = stats_df.kl.apply(logna)
    data.loc[:, "count"] = classifiers_df.drop("pykulgap", axis=1).apply(lambda row: row[row == 1].count(), axis=1)

    ordering = list(data['count'].value_counts().index)
    ordering.sort(reverse=True)
    g = sns.FacetGrid(data, row="count", hue="count", row_order=ordering,
                      height=1.5, aspect=4, margin_titles=False)

    # Draw the densities
    g.map(plt.axhline, y=0, lw=1, clip_on=False, color='black')
    g.map(sns.distplot, "klval", hist=True, rug=True, rug_kws={'height': 0.1})

    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)
        ax.axvline(x=np.log(7.97), color='black', linestyle="-")  # critical value for p-val=0.05
        ax.axvline(x=np.log(5.61), color='black', linestyle="--")  # critical value for p-val=0.1
        ax.axvline(x=np.log(13.9), color='black', linestyle="--")  # critical value for p-val=0.001 

    g.map(label, "klval")

    # Set the subplots to have no spacing
    g.fig.subplots_adjust(hspace=0.01)

    # Remove axes details
    g.set_titles("")
    g.set(yticks=[])

    # Set labels
    g.set_axis_labels(x_var='log(KL)')
    plt.ylabel('Number of measures that agree on a "responder" label', horizontalalignment='left')
    g.despine(bottom=True, left=True)
    plt.savefig("{}.pdf".format(savename))


## -- create_heatmaps.py

# TODO THIS function is to be removed
# def conservative_score(l1, l2, n, response):
#     """

#     :param l1:
#     :param l2:
#     :param n:
#     :param response:
#     :return:
#     """
#     assert len(l1) == len(l2)

#     def convert(variable, n, response):  # Convert what to what? Variable names shadow enclosing scope
#         """
#         Add brief description of function here.

#         :param variable:
#         :param n:
#         :param response:
#         :return:
#         """
#         if variable == n:
#             return -1
#         if variable == response:
#             return 1
#         return 0

#     return (l1.map(lambda variable: convert(variable, n, response)).sum() - l2.map(lambda variable: convert(variable, n, response)).sum()) / 2 / len(l1)


def create_agreements(responders_df):
    """
    Creates the agreement matrix (percentage of same calls) between the different measures.
    :param responders_df: [DataFrame] The dataframe of responders: one column per measure, one row per experiment
    :return: [DataFrame] agreements the agreement matrix
    """
    agreements = pd.DataFrame([[accuracy_score(responders_df[x], responders_df[y]) for x in
                                responders_df.columns] for y in responders_df.columns])
    responders_df.rename(columns={"mRECIST-Novartis": "mRECIST"}, inplace=True)
    agreements.columns = responders_df.columns
    agreements.index = responders_df.columns
    return agreements


# TODO ThIS function is to be removed
## TODO:: Consider more informative function name. Suggestion - find_conservative_aggreements

# def create_conservative(agreements_df):
#     """

#     :param agreements_df: [DataFrame]
#     :return: []
#     """
#     conservative_agreements = pd.DataFrame([[conservative_score(agreements_df[variable], agreements_df[response], -1, 1) for variable in
#                                              agreements_df.columns] for response in agreements_df.columns])
#     agreements_df.rename(columns={"mRECIST-Novartis": "mRECIST"}, inplace=True)
#     conservative_agreements.columns = agreements_df.columns
#     conservative_agreements.index = agreements_df.columns
#     return conservative_agreements


def create_FDR(responders_df):
    """
    Creates the false discovery rate (FDR) matrix from the responders
    :param responders_df: [DataFrame] The dataframe of responders: one column per measure, one row per experiment
    :return [DataFrame]: The FDR matrix
    """
    n = responders_df.shape[1]
    FDR_df = pd.DataFrame(np.zeros((n, n)))
    for row in range(n):
        for col in range(n):
            if responders_df[responders_df.iloc[:, col] == 1].shape[0] != 0:
                FDR_df.iloc[row, col] = \
                responders_df[(responders_df.iloc[:, row] == -1) & (responders_df.iloc[:, col] == 1)].shape[0] / \
                responders_df[responders_df.iloc[:, col] == 1].shape[0]
            else:
                FDR_df.iloc[row, col] = np.nan

    FDR_df = FDR_df.T  # transpose
    FDR_df.columns = responders_df.columns
    FDR_df.index = responders_df.columns
    return FDR_df


def create_KT(responders_df):
    """
    Creates the matrix of Kendall tau tests between the different responders
    :param responders_df: [DataFrame] The dataframe of responders: one column per measure, one row per experiment
    :return [DataFrame]: The matrix of Kendall tau results
    """
    kts_df = pd.DataFrame(
        [[stats.kendalltau(responders_df[x], responders_df[y])[0] for x in responders_df.columns] for y in
         responders_df.columns])
    kts_df.columns = responders_df.columns
    kts_df.index = responders_df.columns
    return kts_df
