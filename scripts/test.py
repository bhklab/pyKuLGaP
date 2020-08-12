#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 09:07:30 2020

@author: ortmann_j
"""

from pykulgap.io import read_pdx_data
from pykulgap.plotting import plot_everything, create_and_plot_agreements, get_classification_df, \
    create_and_plot_FDR, create_and_save_KT, plot_histograms_2c

results_folder = "results"
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




anon_filename = data_folder + "alldata_new.csv"

failed_plot = []
failed_gp=[]
failed_p_value = []
failed_mrecist = []



allowed_list = []

P_VAL = 0.05
fit_gp = True




all_patients = read_pdx_data(anon_filename)

all_patients = all_patients


tc = all_patients['P1']['C1']
cc = all_patients['P1']['Control']



full_stats_df = all_patients.summary_stats_df
        


# =============================================================================
# COMPILATION OF STATISTICS
# =============================================================================

# old_stats_df = pd.read_csv('https://raw.githubusercontent.com/bhklab/pyKuLGaP/8413a329ad64da4f8e1a2a8efd87d856619d3937/results/statistics_all.csv')
# old_stats_df = old_stats_df[full_stats_df.columns]
#
# i = 0
# pd.DataFrame([old_stats_df.iloc[i, :], full_stats_df.iloc[i, :]]).T


## TODO: re-write so it only needs full_stats_df (and no longer treatment_response_expt)
classifiers_df = get_classification_df(full_stats_df)


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

plot_everything(allplot_figname, all_patients, full_stats_df, classifiers_df, kl_null_filename)