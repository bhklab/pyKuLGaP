    # -*- coding: utf-8 -*-
"""
Created on Wed May 24 15:03:17 2017

@author: ortmann_j
"""

# TODO: Check which of the imports can be removed
import pandas as pd
import numpy as np
import pdxat
import statsmodels.api  as sm
from collections import defaultdict








from allplot import plot_everything, create_and_plot_agreements, create_scatterplot, create_and_plot_conservative,\
    get_classification_df,get_classification_df_from_df,  plot_category, plot_gp, plot_histogram,\
    create_and_plot_FDR, create_and_save_KT

import gc
from read_data_from_anonymous import read_anonymised

from aux_functions import get_all_cats, calculate_AUC, kl_divergence, calculate_null_kl

#from read_in import read_in_patients

    



    
    




    
def remove_extremal_nas(y,replacement_value):
    """
    Replaces leading and trailing n/a values in the rows of y by replacement_value
    Returns the modified y, the start (first measurement) and the end (last measurement) dates
    """
    firsts=[]
    lasts=[]
    for j,y_slice in enumerate(y):
        ind = np.where(~np.isnan(y_slice))[0]
        firsts.append(ind[0])
        lasts.append(ind[-1])
    
        y[j,:firsts[-1]] = replacement_value
        y[j,lasts[-1] + 1:] = replacement_value
    first = max(firsts)
    last=min(lasts)
    return y, first, last


def forward_fill_nas(arr):
    """
    forward-fills n/a values in numpy array arr: replaces it by previous valid choice
    """
    mask = np.isnan(arr)
    idx = np.where(~mask,np.arange(mask.shape[1]),0)
    np.maximum.accumulate(idx,axis=1, out=idx)
    out = arr[np.arange(idx.shape[0])[:,None], idx]
    return out
    
    
def relativize(y, start):
    return y/y[start]-1


def centre(y,start):
    return y-y[start]
    


def compute_response_angle(x,y, start):
    l=min(len(x),len(y))
    model = sm.OLS(y[start:l],x[start:l])
    results = model.fit()
    return np.arctan(results.params[0])    

    
    #return np.arctan((y[l-1] - y[start]) / (x[l-1] - x[start]) )
    
    
def dict_to_string(d):
    return "_".join([str(key)+":"+str(value) for key,value in d.items()])









# the list of data that we can't use because of missing data in cells
ignore_list=[]
#ignore_list = ['NSCLC #191', 'NSCLC#191', 'Model Breast Cancer Notch 101',
#              'Breast Cancer Notch 101', 'Model NOTCH xeno line', 'OVR line #2508-P8',
#               'NOTCH 010-002(F)- 001', 'Unknown',
#               '164-401F-502-601-701-803-903F-1001-1101& 1102',
#               '164 KOTA-P3-401S(F)-502S-601S-701S-803S-902S-1001S',
#               '164Kota-P3-401S(F)-503S(F)-611S-707']


#to_replace= { 'Doc+ Sel Concomitant': 'Doce +Sel Concomitant','Doc+ Sel Concomitant':'Doce +Sel Concomitant','Pacli + PF384' : 'Paclitaxel + PF384', 'PF384 + PF299': 'PF299+ PF384', 
#             'NVP-BJG398':'NVP-BGJ398', 'BJG398 + Crizotinib' : 'BGJ-398 + Crizotinib', 'Doc + Sel Concomitant' : 'Doce +Sel Concomitant', 'Criz' : 'Crizotinib', 'BKM120': 'BKM-120',
#             'Erlotinib 25mg' : 'Erlotinib', 'BGJ398':'BGJ-398',  'BGJ698':'BGJ-398', 'BGJ698+Crizotinib':'BGJ-398 + Crizotinib','AZAD4547':'AZD4547',
#             'BGJ398+Crizotinib': 'BGJ-398 + Crizotinib', 'Vin+Cis' : 'Cis + Vin', 'Doc + Sel Sequential' : 'Doce + Sel Sequential', 'BJG398':'BGJ398', 'Vendatinib' : 'Vandetanib', 
#             'PF384+Cisplatin' : 'Cis + PF384' }
#
#drugs={}




# TODO: check if following function can be deleted
#
#def create_category_to_growth(sheet_data, sheet):
#    category_to_growth = {}
#    for replicate in sheet_data[sheet]['Data']:
#        number = replicate[0]
#        category = replicate[1].strip()
#        if category in to_replace:
#            category=to_replace[category]
#        growth = sheet_data[sheet]['Data'][replicate]
#        if category not in category_to_growth:
#            category_to_growth[category] = {}
#            category_to_growth[category][number] = growth
#        else:
#            category_to_growth[category][number] = growth
#
#    return category_to_growth
    
    
    
    
    
    
    
                      

if __name__ == '__main__':
    
    gc.collect()
    
    results_folder = "results/"
    data_folder = "data/"
    
# =============================================================================
#     Definition of file links
# =============================================================================
    
    ############WHERE THE INPUT DATA IS SAVED##########################################################
    anon_filename = data_folder+"alldata_new.csv"
    filename_crown =  data_folder+"20180402_sheng_results.csv"
    ############WHERE THE REPORT (THINGS THAT FAILED) IS SAVED#########################################
    out_report=results_folder+'report_all.txt'#'report_LPDX.txt'
    
    
    ############WHERE THE OUTPUT STATISTICS ARE SAVED##################################################
    stats_outname = results_folder+"statistics_all.csv"#"statistics_safeside.csv"
    classifiers_outname = results_folder+"classifiers.csv"
    agreements_outname = results_folder+"Fig2_agreements.csv"
    agreements_outfigname=results_folder+"Fig2_agreements.pdf"
    conservative_outname = results_folder+"Fig2_conservative.csv"
    conservative_outfigname = results_folder+"Fig2_conservative.pdf"
    scatterplot_outfigname = results_folder+"Fig2_scatterplot.pdf"    
    
    fig1a_figname = results_folder+"fig1a.pdf"
    fig1b_figname = results_folder+"fig1b.pdf"
    fig1c_figname = results_folder+"fig1c.pdf"
    fig1d_figname = results_folder+"fig1d.pdf"    

    fig3a_figname = results_folder+"fig3a.pdf"
    fig3b_figname = results_folder+"fig3b.pdf"

    fig4a_figname = results_folder+"fig4a.pdf"
    fig4b_figname = results_folder+"fig4b.pdf"
    fig4c_figname = results_folder+"fig4c.pdf"
    fig4d_figname = results_folder+"fig4d.pdf"    
    
    
    fig5a_figname = results_folder+"fig5a.pdf"
    fig5b_figname = results_folder+"fig5b.pdf"
    fig5c_figname = results_folder+"fig5c.pdf"
    fig5d_figname = results_folder+"fig5d.pdf"      



    supfig1_figname = results_folder+"sup-fig1.pdf"


    supfig2a_figname = results_folder+"sup-fig2a.pdf"
    supfig2b_figname = results_folder+"sup-fig2b.pdf"


    supfig3a_figname =  results_folder+"sup-fig3a.pdf"
    supfig3b_figname = results_folder+"sup-fig3b.pdf"

    histograms_out = results_folder+"KLDivergenceHistograms/"

    histograms_outfile=results_folder+"kl_histograms.csv"
    
    KT_outname = results_folder+"Kendalls_tau.csv"


    ###################################################################################################

    fit_gp = True#True # whether to fit GPs or not
    draw_plots = True #whether to make PDF with plots+stats about each model
    
    
    
    

 
    all_patients = read_anonymised(anon_filename)
    
    failed_plot = []
    failed_p_value = []
    failed_mrecist = []
    failed_gp=[]
    failed_response_angle = []

   

    allowed_list = []


    P_VAL=0.05

# =============================================================================
# GP fitting and calculation of other parameters.
# =============================================================================


    for i in range(0,len(all_patients)):
        print("Now dealing with patient %d of %d"%(i+1,len(all_patients)))
        
        if (allowed_list ==[]) or (all_patients[i].name in allowed_list):
        #if all_patients[i].name not in ignore_list:
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
                    cur_case.start = max(cur_case.find_start_date_index(),control.measurement_start)
                    cur_case.end = min(control.measurement_end,cur_case.measurement_end)

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
                            #NEED TO FIGURE OUT HOW TO REFER TO GENERIC ERROR
                            failed_gp.append((cur_case.phlc_id,e))


                    # MRECIST
                    try:
                        cur_case.calculate_mrecist()
                        assert (cur_case.mrecist is not None)
                    except ValueError as e:
                        failed_mrecist.append((cur_case.phlc_id, e))
                        print(e)
                        continue
                    
                    #angle
                    
                    try:
                        cur_case.calculate_response_angles(control)
                        assert (cur_case.response_angle is not None)
                        cur_case.response_angle_control={}
                        for i in range(len(control.replicates)):
                            #cur_case.response_angle_control[control.replicates[i]] = compute_response_angle(control.x.ravel(),control.y[i],control.find_start_date_index())
                            start = control.find_start_date_index() - control.measurement_start
                            if start == None:
                                raise
                            else:
                                cur_case.response_angle_control[control.replicates[i]] = compute_response_angle(control.x_cut.ravel(),centre(control.y[i,control.measurement_start:control.measurement_end+1],start),start)
                                cur_case.response_angle_rel_control[control.replicates[i]] = compute_response_angle(control.x_cut.ravel(),relativize(control.y[i,control.measurement_start:control.measurement_end+1],start),start) 
            
                    except ValueError as e:
                        failed_response_angle.append((cur_case.phlc_id, e))
                        print(e)
                        continue
                    #compute AUC
                    try:
                        cur_case.calculate_auc(control)
                        cur_case.calculate_auc_norm(control)
                        if fit_gp:
                            cur_case.calculate_gp_auc()
                            cur_case.auc_gp_control = calculate_AUC(control.x_cut,control.gp.predict(control.x_cut) [0])
                        cur_case.auc_control={}
                        start = max(cur_case.find_start_date_index(),control.measurement_start)
                        end = min(cur_case.measurement_end,control.measurement_end)                        
                        for i in range(len(control.replicates)):
                            cur_case.auc_control[control.replicates[i]] = calculate_AUC(control.x[start:end],control.y[i,start:end])
                            cur_case.auc_control_norm[control.replicates[i]] = calculate_AUC(control.x[start:end],control.y_norm[i,start:end])
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
                        
                        #compute GP derivatives:                        
                        cur_case.compute_all_gp_derivatives(control)
                    
                    
   #COMPUTATION OF P-VALUES IN SEPARATE ITERATION: WE FIRST NEED TO HAVE FIT ALL THE GPs
            


    #NOW CYCLE AGAIN THROUGH all_patients TO COMPUTE kl p-values:        
     

    categories_by_drug = defaultdict(list)
    failed_by_drug=defaultdict(list)
    for patient in all_patients:
        for key in patient.categories.keys():
            if patient.categories[key].gp:
                categories_by_drug[key].append(patient.categories[key])
            else:
                failed_by_drug[key].append(patient.categories[key].name)

    fig_count=0 
    cur_case.kl_p_cvsc=None
    
    if fit_gp:    
        print("Now computing KL divergences between controls for kl_control_vs_control - this may take a moment")
        controls = [patient.categories["Control"] for patient in all_patients]
        kl_control_vs_control = calculate_null_kl(controls)
        
       
        
        
 
        print("Done computing KL divergences between controls for kl_control_vs_control")
    
        #The following  plots the KL histgrams
        kl_histograms=defaultdict(list)          
        print ("Now computing KL p-values")
        for i in range(0,len(all_patients)):
            if (allowed_list ==[]) or (all_patients[i].name in allowed_list):
                
                patient = all_patients[i]
                print("Patient: ", patient.name, "(",i+1,"of",len(all_patients), ")")
                if patient.name not in ignore_list:
                    for category in patient.categories.keys():
                        if category != 'Control':
                            print("Category: ", category)
                            cur_case = patient.categories[category]
                            
                            # IF FIRST OCCURRENCE OF DRUG: COMPUTE HISTOGRAM OF KL DIVERGENCES
                            #if cur_case.name not in kl_histograms:
                                
                                ###SOMETHING BAD GOING ON HERE:
                                #kl_histograms[cur_case.name] = [kl_divergence(x,y) for x in categories_by_drug[cur_case.name] for y in categories_by_drug['Control']]
    
                            try:
                                if cur_case.kl_divergence is not None:
                                ####COMPUTE KL DIVERGENCE PVALUES HERE!!
                                
                                    ##The old way of computing kl_p_value (by comparing against 
                                    ## [kl(x,y) for x in same_drug for y in controls]) doesn't really
                                    ## make sense in the aanonymised setting (the `drug' will be simply C1, C2, etc.)
                                    ## therefore replace by same calculation as kl_p_cvsc
                                    ## cur_case.kl_p_value= (len([x for x in kl_histograms[cur_case.name] if x >= cur_case.kl_divergence]) + 1) / (len(kl_histograms[cur_case.name]) + 1)                                    
                                    cur_case.kl_p_value = (len([x for x in kl_control_vs_control["list"] if x >= cur_case.kl_divergence]) + 1) / (len(kl_control_vs_control["list"]) + 1)

                                    cur_case.kl_p_cvsc= 1-kl_control_vs_control["smoothed"].cdf([cur_case.kl_divergence])
#                                    print(cur_case.kl_p_value,cur_case.kl_p_cvsc, (cur_case.kl_p_cvsc-cur_case.kl_p_value)/cur_case.kl_p_cvsc)

        
                                    assert (cur_case.kl_p_value is not None)
                            except Exception as e:
                                failed_p_value.append((cur_case.phlc_id, e))
                                print(e)
                                raise 
    if fit_gp:                    
        with open(histograms_outfile,'w') as outfile:
            for key,value in kl_histograms.items():
                outfile.write(str(key)+"\n")
                outfile.write(",".join(map(str,value)))
                outfile.write("\n")
    print("Done computing KL p-values, saved to {}".format(histograms_outfile))
    all_kl = [x["case"].kl_divergence for x in get_all_cats(all_patients).values() if str(x["case"].kl_divergence) != "nan"]    
    
    with open(out_report, 'w') as f:
        print("Errors during plotting:",file=f)
        print(failed_plot, file=f)
        print("\n\n\n", file=f)
        print("failed p-values:",file=f)
        print(failed_p_value, file=f)
        print("\n\n\n", file=f)
        print(failed_mrecist, file=f)
        print("\n\n\n", file=f)
        print("Errors during GP fitting:",file=f)
        print(failed_gp,file=f)
        

        
        
    
# =============================================================================
# COMPILATION OF STATISTICS
# =============================================================================

    

    stats_dict = {}
    for i in range(0, len(all_patients)):
        if (allowed_list ==[]) or (all_patients[i].name in allowed_list):
            patient = all_patients[i]
            
            control = patient.categories['Control']
        #     control.normalize_data()
        #     control.fit_gaussian_processes()
            
            for category in patient.categories.keys():
                if 'Control' not in category:
                    cur_case = patient.categories[category]
                    key = str(cur_case.phlc_id) + "*" + str(category)
                    stats_dict[key] = {'tumour_type':patient.tumour_type,'mRECIST':None,'num_mCR': None, 'num_mPR': None,
                                       'num_mSD': None, 'num_mPD': None,
                                       'perc_mCR': None, 'perc_mPR': None,
                                       'perc_mSD': None, 'perc_mPD': None,
                                       'drug': None,
                                       'response_angle': None, 'response_angle_control':None, 'perc_true_credible_intervals': None,
                                       'delta_log_likelihood': None, 
                                       'kl': None, 'kl_p_value': None,'kl_p_cvsc': None, 'gp_deriv': None,'gp_deriv_control':None,'auc':None, 
                                       'auc_control_norm' : None,'auc_norm' : None,'auc_control':None,'auc_gp':None,'auc_gp_control':None,
                                       'number_replicates':len(cur_case.replicates),'number_replicates_control':len(control.replicates),
                                        "tgi" : cur_case.tgi}
                    stats_dict[key]['drug'] = category
                    

                    try:
                        cur_case.calculate_mrecist()
                        cur_case.enumerate_mrecist()
                    except Exception as e:
                        print(e)
                        continue
                        
                    
                    num_replicates = len(cur_case.replicates)
                    stats_dict[key]['mRECIST']= dict_to_string(cur_case.mrecist)
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
    
                    stats_dict[key]['auc']=dict_to_string(cur_case.auc)
                    stats_dict[key]['auc_norm']=dict_to_string(cur_case.auc_norm)
                    stats_dict[key]['auc_control']=dict_to_string(cur_case.auc_control)
                    stats_dict[key]['auc_control_norm'] = dict_to_string(cur_case.auc_control_norm)
                    try:
                        stats_dict[key]['auc_gp']=cur_case.auc_gp[0]
                        stats_dict[key]['auc_gp_control']=cur_case.auc_gp_control[0]
                    except TypeError:
                        stats_dict[key]['auc_gp']=""
                        stats_dict[key]['auc_gp_control']=""
                    
                    stats_dict[key]['response_angle']=dict_to_string(cur_case.response_angle)
                    stats_dict[key]['response_angle_rel']=dict_to_string(cur_case.response_angle_rel)
                    stats_dict[key]['response_angle_control']=dict_to_string(cur_case.response_angle_control)
                    stats_dict[key]['response_angle_rel_control']=dict_to_string(cur_case.response_angle_rel_control)
                    
                    stats_dict[key]['average_angle']= cur_case.average_angle
                    stats_dict[key]['average_angle_rel']=cur_case.average_angle_rel
                    stats_dict[key]['average_angle_control']=cur_case.average_angle_control
                    stats_dict[key]['average_angle_rel_control']=cur_case.average_angle_rel_control
    

    
    stats_df = pd.DataFrame.from_dict(stats_dict).transpose()
    
    crown_df = pd.read_csv(filename_crown,index_col="Seq.")
    
    
    
    
    crown_df.kl_p_cvsc = crown_df.kl.apply(lambda x: 1-kl_control_vs_control["smoothed"].cdf([x]))
    

    full_stats_df = stats_df.append(crown_df.rename({"TGI":"tgi"},axis=1),sort=True)
    print(full_stats_df.columns)
    
    
    
 
    classifiers_df_ours = get_classification_df(all_patients,stats_df,.05,kl_control_vs_control["list"],0.05,.6)    
    classifiers_df_ours.rename(columns = {"mRECIST_ours":"mRECIST-ours","mRECIST_Novartis":"mRECIST-Novartis"},inplace=True)
    classifiers_df_crown=get_classification_df_from_df(crown_df, .05,kl_control_vs_control["list"],0.05,.6)
    

    classifiers_df = classifiers_df_ours.append(classifiers_df_crown)
    
    classifiers_df.drop(["mRECIST-ours","KuLGaP-prev"],axis=1,inplace=True)
    
    
    
    

# =============================================================================
#     Finally we save all our files to the disk and create the figures:
# =============================================================================
    
    stats_df.to_csv(stats_outname)
    classifiers_df.to_csv(classifiers_outname)
    

    
    # Figure 1: pipeline
    
    case_fig1 = all_patients[53].categories["C1"]
    control_fig1 = all_patients[53].categories["Control"]
    
    ## Fig 1a: 
    plot_category(case_fig1,control_fig1,means=None,savename=fig1a_figname,normalised=False)
    ## Fig 1b:
    plot_category(case_fig1,control_fig1,means=None,savename=fig1b_figname)
    ## Fig 1c:
    plot_gp(case_fig1,control_fig1,savename=fig1c_figname)
    ## Fig 1d:
    
    plot_histogram(kl_control_vs_control["list"],"KL values",7.97,fig1d_figname,0,30,smoothed=kl_control_vs_control["smoothed"].pdf)
    
    


    
    # Figure 2: heatmaps and scatterplot
    create_and_plot_agreements(classifiers_df,agreements_outfigname,agreements_outname)
    #create_and_plot_conservative(classifiers_df,conservative_outfigname,conservative_outname)
    create_and_plot_FDR(classifiers_df,conservative_outfigname,conservative_outname)
    create_scatterplot(full_stats_df,classifiers_df,scatterplot_outfigname)
#    create_scatterplot(stats_df,classifiers_df,scatterplot_outfigname)
    
    
    
    # Figures 3 - 5: individual plots    
#    plot_category(case,control,means=None)
    
    ## Figure 3:
    case_fig3= all_patients[11].categories["C1"]
    control_fig3= all_patients[11].categories["Control"]    
    plot_category(case_fig3,control_fig3,means=None,savename=fig3a_figname)
    plot_category(case_fig3,control_fig3,means="only",savename=fig3b_figname)
    
    
    ## Figure 4:
    case_fig4ab = all_patients[40].categories["C1"]
    control_fig4ab=all_patients[40].categories["Control"]
    plot_category(case_fig4ab,control_fig4ab,means=None,savename=fig4a_figname)
    plot_category(case_fig4ab,control_fig4ab,means="only",savename=fig4b_figname)
    
    case_fig4cd = all_patients[34].categories["C2"]
    control_fig4cd=all_patients[34].categories["Control"]
    plot_category(case_fig4cd,control_fig4cd,means=None,savename=fig4c_figname)
    plot_category(case_fig4cd,control_fig4cd,means="only",savename=fig4d_figname)
    


    ## Figure 5:
    case_fig5ab = all_patients[48].categories["C1"]
    control_fig5ab=all_patients[48].categories["Control"]
    plot_category(case_fig5ab,control_fig5ab,means=None,savename=fig5a_figname)
    plot_category(case_fig5ab,control_fig5ab,means="only",savename=fig5b_figname)
    
    case_fig5cd = all_patients[5].categories["C1"]
    control_fig5cd=all_patients[5].categories["Control"]
    plot_category(case_fig5cd,control_fig5cd,means=None,savename=fig5c_figname)
    plot_category(case_fig5cd,control_fig5cd,means="only",savename=fig5d_figname)
    

   ## Supplementary Figure 1:
    
    case_fig1s= all_patients[28].categories["C1"]

    plot_category(case_fig1s,None,means=None,savename=supfig1_figname)

    
    ## Supplementary Figure 2:
    
    case_fig2s= all_patients[3].categories["C1"]
    control_fig2s= all_patients[3].categories["Control"]    
    plot_category(case_fig2s,control_fig2s,means=None,savename=supfig2a_figname)
    plot_category(case_fig2s,control_fig2s,means="only",savename=supfig2b_figname)
    
    ## Supplementary Figure 3:    
    
    
    case_fig3s= all_patients[11].categories["C3"]
    control_fig3s= all_patients[11].categories["Control"]    
    plot_category(case_fig3s,control_fig3s,means=None,savename=supfig3a_figname)
    plot_category(case_fig3s,control_fig3s,means="only",savename=supfig3b_figname)
    
    create_and_save_KT(classifiers_df,KT_outname)
    
    
    
    

    
