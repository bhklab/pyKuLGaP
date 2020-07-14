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
from kulgap.aux_functions import get_all_cats, calculate_null_kl, dict_to_string

from kulgap.read_data_from_anonymous import read_anonymised


results_folder = "results/test-run/"
data_folder = "data/"



############WHERE THE INPUT DATA IS SAVED##########################################################
anon_filename = data_folder + "alldata_new.csv"
filename_crown = data_folder + "20180402_sheng_results.csv"

kl_null_filename = data_folder + "kl_control_vs_control.csv" #set to None to re-compute
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
failed_gp=[]
failed_p_value = []
failed_mrecist = []



allowed_list = []

P_VAL = 0.05
fit_gp=True


all_patients = read_anonymised(anon_filename)

for i,patient in enumerate(all_patients):
    print("Now dealing with patient %d of %d" % (i + 1, len(all_patients)))
    print("Num failed mRECISTS: " + str(len(failed_mrecist)))
    print("Num failed plots: " + str(len(failed_plot)))
    print("Num failed p values: " + str(len(failed_p_value)))
    print("Patient: " + str(patient.name))
    patient.normalize_all_categories()
    
    if fit_gp:
        patient.fit_all_gps()
        
    patient.compute_other_measures(fit_gp,report_name = out_report)
        
        
    control=patient.categories["Control"]  
    
    for category in patient.categories.keys():
        if category != 'Control':

            cur_case = patient.categories[category]
            

            

            
            
            

            




# =============================================================================
# CALCULATE KL P-VALUES
# =============================================================================

# categories_by_drug = defaultdict(list)
# failed_by_drug = defaultdict(list)
# for patient in all_patients:
#     for key in patient.categories.keys():
#         if patient.categories[key].gp:
#             categories_by_drug[key].append(patient.categories[key])
#         else:
#             failed_by_drug[key].append(patient.categories[key].name)


print("Now computing KL divergences between controls for kl_control_vs_control - this may take a moment")
controls = [patient.categories["Control"] for patient in all_patients]
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

full_stats_df =pd.DataFrame.from_dict(stats_dict).transpose()





classifiers_df = get_classification_df(all_patients, full_stats_df, .05, kl_control_vs_control["list"], 0.05, .6)
classifiers_df.rename(columns={"mRECIST_ours": "mRECIST-ours", "mRECIST_Novartis": "mRECIST-Novartis"},
                           inplace=True)

classifiers_df.drop(["mRECIST-ours", "kulgap-prev"], axis=1, inplace=True)

# =============================================================================
#     Finally we save all our files to the disk and create the figures:
# =============================================================================

full_stats_df.to_csv(stats_outname)
classifiers_df.to_csv(classifiers_outname)


# Figure 2: heatmaps and scatterplot
create_and_plot_agreements(classifiers_df, agreements_outfigname, agreements_outname)
create_and_plot_FDR(classifiers_df, conservative_outfigname, conservative_outname)
plot_histograms_2c(full_stats_df, classifiers_df, scatterplot_outfigname)




create_and_save_KT(classifiers_df, KT_outname)

plot_everything(allplot_figname, all_patients, full_stats_df, classifiers_df, True, 0.05, 0.05,
                kl_control_vs_control["list"], .6)




#quick verification:

statistics_old=pd.read_csv("results/test-run-old/statistics_all.csv",index_col=0)
if np.all(full_stats_df == statistics_old):
    print("OK!")
else:
    
    for i,col in enumerate(full_stats_df.columns):
        if full_stats_df.iloc[0,i]!=statistics_old.iloc[0,i]:
            print(col)
            if np.abs(full_stats_df.iloc[0,i]-statistics_old.iloc[0,i])>0.00001:
                print(col)
                print(full_stats_df.iloc[0,i]-statistics_old.iloc[0,i])
    else:
        print("Only rounding errors")
            
        
