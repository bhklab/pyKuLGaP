import gc
from collections import defaultdict

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .allplot import plot_everything, create_and_plot_agreements, get_classification_df, \
    plot_category, plot_gp, plot_histogram, \
    create_and_plot_FDR, create_and_save_KT, plot_histograms_2c
from .aux_functions import get_all_cats, calculate_AUC, calculate_null_kl, dict_to_string, \
    relativize, centre, compute_response_angle

from .read_data_from_anonymous import read_anonymised












ignore_list = []







if __name__ == '__main__':

    gc.collect()

    results_folder = "../results/"
    data_folder = "../data/"

    # =============================================================================
    #     Definition of file links
    # =============================================================================

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

    fig1a_figname = results_folder + "fig1a.pdf"
    fig1b_figname = results_folder + "fig1b.pdf"
    fig1c_figname = results_folder + "fig1c.pdf"
    fig1d_figname = results_folder + "fig1d.pdf"

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

    fit_gp = True  # True # whether to fit GPs or not
    draw_plots = True  # whether to make PDF with plots+stats about each model

    rerun_kl_null = False  # whether to re-compute the KL null distribution

    if fit_gp is False:
        rerun_kl_null = False

    all_patients = read_anonymised(anon_filename)

    failed_plot = []
    failed_p_value = []
    failed_mrecist = []
    failed_gp = []
    failed_response_angle = []

    allowed_list = []

    P_VAL = 0.05

    # =============================================================================
    # GP fitting and calculation of other parameters.
    # =============================================================================
    #TODO: replace by fit_all_gps(all_patients, ... )
    for i in range(0, len(all_patients)):
        print("Now dealing with patient %d of %d" % (i + 1, len(all_patients)))

        if (allowed_list == []) or (all_patients[i].name in allowed_list):
            # if all_patients[i].name not in ignore_list:
            print("Num failed mRECISTS: " + str(len(failed_mrecist)))
            print("Num failed plots: " + str(len(failed_plot)))
            print("Num failed p values: " + str(len(failed_p_value)))

            patient = all_patients[i]

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
                if patient.name not in ignore_list:
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
    # all_kl = [x["case"].kl_divergence for x in get_all_cats(all_patients).values() if
    #           str(x["case"].kl_divergence) != "nan"]

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

    crown_df = pd.read_csv(filename_crown, index_col="Seq.")

    crown_df.kl_p_cvsc = crown_df.kl.apply(lambda x: 1 - kl_control_vs_control["smoothed"].cdf([x]))

    full_stats_df = stats_df.append(crown_df.rename({"TGI": "tgi"}, axis=1), sort=True)
    print(full_stats_df.columns)

    classifiers_df_ours = get_classification_df(all_patients, stats_df, .05, kl_control_vs_control["list"], 0.05, .6)
    classifiers_df_ours.rename(columns={"mRECIST_ours": "mRECIST-ours", "mRECIST_Novartis": "mRECIST-Novartis"},
                               inplace=True)
    classifiers_df_crown = get_classification_df_from_df(crown_df, .05, kl_control_vs_control["list"], 0.05, .6)

    classifiers_df = classifiers_df_ours.append(classifiers_df_crown)

    classifiers_df.drop(["mRECIST-ours", "kulgap-prev"], axis=1, inplace=True)

    # =============================================================================
    #     Finally we save all our files to the disk and create the figures:
    # =============================================================================

    stats_df.to_csv(stats_outname)
    classifiers_df.to_csv(classifiers_outname)

    # Figure 1: pipeline

    case_fig1 = all_patients[53].categories["C1"]
    control_fig1 = all_patients[53].categories["Control"]

    ## Fig 1a: 
    plot_category(case_fig1, control_fig1, means=None, savename=fig1a_figname, normalised=False)
    ## Fig 1b:
    plot_category(case_fig1, control_fig1, means=None, savename=fig1b_figname)
    ## Fig 1c:
    plot_gp(case_fig1, control_fig1, savename=fig1c_figname)
    ## Fig 1d:
    plot_histogram(kl_control_vs_control["list"], "KL values", marked=7.97, savename=fig1d_figname, x_min=0, x_max=30,
                   smoothed=kl_control_vs_control["smoothed"].pdf)

    # Figure 2: heatmaps and scatterplot
    create_and_plot_agreements(classifiers_df, agreements_outfigname, agreements_outname)
    create_and_plot_FDR(classifiers_df, conservative_outfigname, conservative_outname)
    plot_histograms_2c(full_stats_df, classifiers_df, scatterplot_outfigname)

    # Figures 3 - 5: individual plots    
    #    plot_category(case,control,means=None)

    ## Figure 3:
    case_fig3 = all_patients[11].categories["C1"]
    control_fig3 = all_patients[11].categories["Control"]
    plot_category(case_fig3, control_fig3, means=None, savename=fig3a_figname)
    plot_category(case_fig3, control_fig3, means="only", savename=fig3b_figname)

    ## Figure 4:
    case_fig4ab = all_patients[40].categories["C1"]
    control_fig4ab = all_patients[40].categories["Control"]
    plot_category(case_fig4ab, control_fig4ab, means=None, savename=fig4a_figname)
    plot_category(case_fig4ab, control_fig4ab, means="only", savename=fig4b_figname)

    case_fig4cd = all_patients[34].categories["C2"]
    control_fig4cd = all_patients[34].categories["Control"]
    plot_category(case_fig4cd, control_fig4cd, means=None, savename=fig4c_figname)
    plot_category(case_fig4cd, control_fig4cd, means="only", savename=fig4d_figname)

    ## Figure 5:
    case_fig5ab = all_patients[48].categories["C1"]
    control_fig5ab = all_patients[48].categories["Control"]
    plot_category(case_fig5ab, control_fig5ab, means=None, savename=fig5a_figname)
    plot_category(case_fig5ab, control_fig5ab, means="only", savename=fig5b_figname)

    case_fig5cd = all_patients[5].categories["C1"]
    control_fig5cd = all_patients[5].categories["Control"]
    plot_category(case_fig5cd, control_fig5cd, means=None, savename=fig5c_figname)
    plot_category(case_fig5cd, control_fig5cd, means="only", savename=fig5d_figname)

    ## Supplementary Figure 1:

    case_fig1s = all_patients[28].categories["C1"]

    plot_category(case_fig1s, None, means=None, savename=supfig1_figname)

    ## Supplementary Figure 2:

    case_fig2s = all_patients[3].categories["C1"]
    control_fig2s = all_patients[3].categories["Control"]
    plot_category(case_fig2s, control_fig2s, means=None, savename=supfig2a_figname)
    plot_category(case_fig2s, control_fig2s, means="only", savename=supfig2b_figname)

    ## Supplementary Figure 3:    

    case_fig3s = all_patients[11].categories["C3"]
    control_fig3s = all_patients[11].categories["Control"]
    plot_category(case_fig3s, control_fig3s, means=None, savename=supfig3a_figname)
    plot_category(case_fig3s, control_fig3s, means="only", savename=supfig3b_figname)

    ## Supplementary Figure 4:    

    plot_histogram(kl_control_vs_control["list"], "KL values", solid=[7.97], dashed=[5.61, 13.9],
                   savename=supfig4_figname, x_min=0, x_max=30, smoothed=kl_control_vs_control["smoothed"].pdf)

    ## Supplementary Figure 5:    

    case_fig5s = all_patients[60].categories["C3"]
    control_fig5s = all_patients[60].categories["Control"]
    plot_category(case_fig5s, control_fig5s, means=None, savename=supfig5a_figname)
    plot_category(case_fig5s, control_fig5s, means="only", savename=supfig5b_figname)

    ## Supplementary Figure 6:    

    case_fig6s = all_patients[2].categories["C1"]
    control_fig6s = all_patients[2].categories["Control"]
    plot_category(case_fig6s, control_fig6s, means=None, savename=supfig6a_figname)
    plot_category(case_fig6s, control_fig6s, means="only", savename=supfig6b_figname)

    create_and_save_KT(classifiers_df, KT_outname)

    plot_everything(allplot_figname, all_patients, stats_df, classifiers_df, True, 0.05, 0.05,
                    kl_control_vs_control["list"], .6)

    l = ["P11*C1", "P40*C1", "P34*C2", "P48*C1", "P5*C1", "P28*C1", "P3*C1", "P11*C3", "P60*C3", "P2*C1"]
    figure_classifiers = classifiers_df.loc[l, :]
    c = ["Figure 3", "Figure 4AB", "Figure 4CD", "Figure 5AB", "Figure 5CD"]
    c += ["Supplementary Figure {}".format(i) for i in [1, 2, 3, 5, 6]]
    figure_classifiers.index = c
    figure_classifiers.to_csv(results_folder + "figure_classifiers.csv")
    del c
