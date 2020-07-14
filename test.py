#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 09:07:30 2020

@author: ortmann_j
"""


from collections import defaultdict

import numpy as np
import pandas as pd
import statsmodels.api as sm

from kulgap.allplot import plot_everything, create_and_plot_agreements, get_classification_df, \
    get_classification_df_from_df, \
    plot_category, plot_gp, plot_histogram, \
    create_and_plot_FDR, create_and_save_KT, plot_histograms_2c
from kulgap.aux_functions import get_all_cats, calculate_AUC, calculate_null_kl, dict_to_string, \
    relativize, centre, compute_response_angle

from kulgap.read_data_from_anonymous import read_anonymised


results_folder = "results/test-run/"
data_folder = "data/"



############WHERE THE INPUT DATA IS SAVED##########################################################
anon_filename = data_folder + "alldata_new.csv"
filename_crown = data_folder + "20180402_sheng_results.csv"

kl_null_filename = data_folder + "kl_control_vs_control.csv"
############WHERE THE REPORT (THINGS THAT FAILED) IS SAVED#########################################
out_report = results_folder + 'report_all.txt'  # 'report_LPDX.txt'

############WHERE THE OUTPUT STATISTICS ARE SAVED##################################################
stats_outname = results_folder + "statistics_all.csv"  # "statistics_safeside.csv"
classifiers_outname = results_folder + "classifiers.csv"
agreements_outname = results_folder + "Fig2_agreements.csv"
agreements_outfigname = results_folder + "Fig2_agreements.pdf"
conservative_outname = results_folder + "Fig2_conservative.csv"
conservative_outfigname = results_folder + "Fig2_conservative.pdf"
scatterplot_outfigname = results_folder + "Fig2c"



fig3a_figname = results_folder + "fig3a.pdf"
fig3b_figname = results_folder + "fig3b.pdf"

fig4a_figname = results_folder + "fig4a.pdf"
fig4b_figname = results_folder + "fig4b.pdf"
fig4c_figname = results_folder + "fig4c.pdf"
fig4d_figname = results_folder + "fig4d.pdf"

fig5a_figname = results_folder + "fig5a.pdf"
fig5b_figname = results_folder + "fig5b.pdf"
fig5c_figname = results_folder + "fig5c.pdf"
fig5d_figname = results_folder + "fig5d.pdf"

supfig1_figname = results_folder + "sup-fig1.pdf"

supfig2a_figname = results_folder + "sup-fig2a.pdf"
supfig2b_figname = results_folder + "sup-fig2b.pdf"

supfig3a_figname = results_folder + "sup-fig3a.pdf"
supfig3b_figname = results_folder + "sup-fig3b.pdf"

supfig4_figname = results_folder + "sup-fig4.pdf"

supfig5a_figname = results_folder + "sup-fig5a.pdf"
supfig5b_figname = results_folder + "sup-fig5b.pdf"

supfig6a_figname = results_folder + "sup-fig6a.pdf"
supfig6b_figname = results_folder + "sup-fig6b.pdf"

histograms_out = results_folder + "KLDivergenceHistograms/"

histograms_outfile = results_folder + "kl_histograms.csv"

KT_outname = results_folder + "Kendalls_tau.csv"

allplot_figname = results_folder + "allplot.pdf"

###################################################################################################




anon_filename = data_folder + "alldata_small.csv"

failed_plot = []
failed_p_value = []
failed_mrecist = []
failed_gp = []
failed_response_angle = []

allowed_list = []

P_VAL = 0.05
fit_gp=True
rerun_kl_null=False

all_patients = read_anonymised(anon_filename)

for i,patient in enumerate(all_patients):
    print("Now dealing with patient %d of %d" % (i + 1, len(all_patients)))

    print("Num failed mRECISTS: " + str(len(failed_mrecist)))
    print("Num failed plots: " + str(len(failed_plot)))
    print("Num failed p values: " + str(len(failed_p_value)))
    print("Patient: " + str(patient.name))

    # need to ensure that we've found and processed the control.
    control = patient.categories['Control']
    control.normalize_data()

    if fit_gp:
        control.fit_gaussian_processes()
        assert (control.name is not None)
        assert (control.x is not None)
        assert (control.y is not None)
        assert (control.y_norm is not None)
        assert (control.drug_start_day is not None)
        assert (control.replicates is not None)
        assert (control.gp is not None)
        assert (control.gp_kernel is not None)

    for category in patient.categories.keys():
        if category != 'Control':

            cur_case = patient.categories[category]
            cur_case.normalize_data()
            cur_case.start = max(cur_case.find_start_date_index(), control.measurement_start)
            cur_case.end = min(control.measurement_end, cur_case.measurement_end)

            cur_case.create_full_data(control)
            assert (cur_case.full_data != [])

            # DELTA LOG LIKELIHOOD
            if fit_gp:
                try:
                    cur_case.fit_gaussian_processes(control=control)
                    assert (cur_case.gp_h0 is not None)
                    assert (cur_case.gp_h0_kernel is not None)
                    assert (cur_case.gp_h1 is not None)
                    assert (cur_case.gp_h1_kernel is not None)
                    assert (cur_case.delta_log_likelihood_h0_h1 is not None)

                    # KL DIVERGENCE
                    cur_case.calculate_kl_divergence(control)
                    assert (cur_case.kl_divergence is not None)
                except Exception as e:
                    # NEED TO FIGURE OUT HOW TO REFER TO GENERIC ERROR
                    failed_gp.append((cur_case.phlc_id, e))

            # MRECIST
            try:
                cur_case.calculate_mrecist()
                assert (cur_case.mrecist is not None)
            except ValueError as e:
                failed_mrecist.append((cur_case.phlc_id, e))
                print(e)
                continue

            # angle

            try:
                cur_case.calculate_response_angles(control)
                assert (cur_case.response_angle is not None)
                cur_case.response_angle_control = {}
                for i in range(len(control.replicates)):
                    # cur_case.response_angle_control[control.replicates[i]] = compute_response_angle(control.x.ravel(),control.y[i],control.find_start_date_index())
                    start = control.find_start_date_index() - control.measurement_start
                    if start is None:
                        raise TypeError("The 'start' parameter is None")
                    else:
                        cur_case.response_angle_control[control.replicates[i]] = compute_response_angle(
                            control.x_cut.ravel(),
                            centre(control.y[i, control.measurement_start:control.measurement_end + 1], start),
                            start)
                        cur_case.response_angle_rel_control[control.replicates[i]] = compute_response_angle(
                            control.x_cut.ravel(),
                            relativize(control.y[i, control.measurement_start:control.measurement_end + 1],
                                       start), start)

            except ValueError as e:
                failed_response_angle.append((cur_case.phlc_id, e))
                print(e)
                continue
            # compute AUC
            try:
                cur_case.calculate_auc(control)
                cur_case.calculate_auc_norm(control)
                if fit_gp:
                    cur_case.calculate_gp_auc()
                    cur_case.auc_gp_control = calculate_AUC(control.x_cut, control.gp.predict(control.x_cut)[0])
                cur_case.auc_control = {}
                start = max(cur_case.find_start_date_index(), control.measurement_start)
                end = min(cur_case.measurement_end, control.measurement_end)
                for i in range(len(control.replicates)):
                    cur_case.auc_control[control.replicates[i]] = calculate_AUC(control.x[start:end],
                                                                                control.y[i, start:end])
                    cur_case.auc_control_norm[control.replicates[i]] = calculate_AUC(control.x[start:end],
                                                                                     control.y_norm[i,
                                                                                     start:end])
            except ValueError as e:
                print(e)

            try:
                cur_case.calculate_tgi(control)
            except ValueError as e:
                print(e)

            # PERCENT CREDIBLE INTERVALS
            if fit_gp:
                cur_case.calculate_credible_intervals(control)
                assert (cur_case.credible_intervals != [])
                cur_case.calculate_credible_intervals_percentage()
                assert (cur_case.percent_credible_intervals is not None)

                # compute GP derivatives:
                cur_case.compute_all_gp_derivatives(control)

# COMPUTATION OF P-VALUES IN SEPARATE ITERATION: WE FIRST NEED TO HAVE FIT ALL THE GPs

# NOW CYCLE AGAIN THROUGH all_patients TO COMPUTE kl p-values:

categories_by_drug = defaultdict(list)
failed_by_drug = defaultdict(list)
for patient in all_patients:
    for key in patient.categories.keys():
        if patient.categories[key].gp:
            categories_by_drug[key].append(patient.categories[key])
        else:
            failed_by_drug[key].append(patient.categories[key].name)

fig_count = 0
cur_case.kl_p_cvsc = None

print("Now computing KL divergences between controls for kl_control_vs_control - this may take a moment")
controls = [patient.categories["Control"] for patient in all_patients]
if rerun_kl_null:
    kl_control_vs_control = calculate_null_kl(controls, None)
else:
    kl_control_vs_control = calculate_null_kl(controls, kl_null_filename)

print("Done computing KL divergences between controls for kl_control_vs_control")

if fit_gp:

    # The following  plots the KL histgrams
    kl_histograms = defaultdict(list)
    print("Now computing KL p-values")
    for i in range(0, len(all_patients)):
        if (allowed_list == []) or (all_patients[i].name in allowed_list):

            patient = all_patients[i]
            print("Patient: ", patient.name, "(", i + 1, "of", len(all_patients), ")")

            for category in patient.categories.keys():
                if category != 'Control':
                    print("Category: ", category)
                    cur_case = patient.categories[category]

                    # IF FIRST OCCURRENCE OF DRUG: COMPUTE HISTOGRAM OF KL DIVERGENCES
                    # if cur_case.name not in kl_histograms:

                    ###SOMETHING BAD GOING ON HERE:
                    # kl_histograms[cur_case.name] = [kl_divergence(x,y) for x in categories_by_drug[cur_case.name] for y in categories_by_drug['Control']]

                    try:
                        if cur_case.kl_divergence is not None:
                            ####COMPUTE KL DIVERGENCE PVALUES HERE!!

                            ##The old way of computing kl_p_value (by comparing against 
                            ## [kl(x,y) for x in same_drug for y in controls]) doesn't really
                            ## make sense in the aanonymised setting (the `drug' will be simply C1, C2, etc.)
                            ## therefore replace by same calculation as kl_p_cvsc
                            ## cur_case.kl_p_value= (len([x for x in kl_histograms[cur_case.name] if x >= cur_case.kl_divergence]) + 1) / (len(kl_histograms[cur_case.name]) + 1)                                    
                            cur_case.kl_p_value = (len([x for x in kl_control_vs_control["list"] if
                                                        x >= cur_case.kl_divergence]) + 1) / (
                                                          len(kl_control_vs_control["list"]) + 1)

                            cur_case.kl_p_cvsc = 1 - kl_control_vs_control["smoothed"].cdf(
                                [cur_case.kl_divergence])
                            #                                    print(cur_case.kl_p_value,cur_case.kl_p_cvsc, (cur_case.kl_p_cvsc-cur_case.kl_p_value)/cur_case.kl_p_cvsc)

                            assert (cur_case.kl_p_value is not None)
                    except Exception as e:
                        failed_p_value.append((cur_case.phlc_id, e))
                        print(e)
                        raise
if fit_gp:
    with open(histograms_outfile, 'w') as outfile:
        for key, value in kl_histograms.items():
            outfile.write(str(key) + "\n")
            outfile.write(",".join(map(str, value)))
            outfile.write("\n")
print("Done computing KL p-values, saved to {}".format(histograms_outfile))
all_kl = [x["case"].kl_divergence for x in get_all_cats(all_patients).values() if
          str(x["case"].kl_divergence) != "nan"]

with open(out_report, 'w') as f:
    print("Errors during plotting:", file=f)
    print(failed_plot, file=f)
    print("\n\n\n", file=f)
    print("failed p-values:", file=f)
    print(failed_p_value, file=f)
    print("\n\n\n", file=f)
    print(failed_mrecist, file=f)
    print("\n\n\n", file=f)
    print("Errors during GP fitting:", file=f)
    print(failed_gp, file=f)

# =============================================================================
# COMPILATION OF STATISTICS
# =============================================================================

stats_dict = {}
for i in range(0, len(all_patients)):
    if (allowed_list == []) or (all_patients[i].name in allowed_list):
        patient = all_patients[i]

        control = patient.categories['Control']
        #     control.normalize_data()
        #     control.fit_gaussian_processes()

        for category in patient.categories.keys():
            if 'Control' not in category:
                cur_case = patient.categories[category]
                key = str(cur_case.phlc_id) + "*" + str(category)
                stats_dict[key] = {'tumour_type': patient.tumour_type, 'mRECIST': None, 'num_mCR': None,
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
                stats_dict[key]['drug'] = category

                try:
                    cur_case.calculate_mrecist()
                    cur_case.enumerate_mrecist()
                except Exception as e:
                    print(e)
                    continue

                num_replicates = len(cur_case.replicates)
                stats_dict[key]['mRECIST'] = dict_to_string(cur_case.mrecist)
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

                stats_dict[key]['auc'] = dict_to_string(cur_case.auc)
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

stats_df = pd.DataFrame.from_dict(stats_dict).transpose()


full_stats_df = stats_df


classifiers_df = get_classification_df(all_patients, stats_df, .05, kl_control_vs_control["list"], 0.05, .6)
classifiers_df.rename(columns={"mRECIST_ours": "mRECIST-ours", "mRECIST_Novartis": "mRECIST-Novartis"},
                           inplace=True)

classifiers_df.drop(["mRECIST-ours", "kulgap-prev"], axis=1, inplace=True)

# =============================================================================
#     Finally we save all our files to the disk and create the figures:
# =============================================================================

stats_df.to_csv(stats_outname)
classifiers_df.to_csv(classifiers_outname)


# Figure 2: heatmaps and scatterplot
create_and_plot_agreements(classifiers_df, agreements_outfigname, agreements_outname)
create_and_plot_FDR(classifiers_df, conservative_outfigname, conservative_outname)
plot_histograms_2c(full_stats_df, classifiers_df, scatterplot_outfigname)




create_and_save_KT(classifiers_df, KT_outname)

plot_everything(allplot_figname, all_patients, stats_df, classifiers_df, True, 0.05, 0.05,
                kl_control_vs_control["list"], .6)
