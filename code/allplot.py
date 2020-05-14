# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 10:24:51 2018

@author: ortmann_j
"""

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from create_heatmaps import create_agreements, create_conservative, create_FDR, create_KT
import seaborn as sns
sns.set(style="ticks")


from matplotlib import pylab as pl, patches as mp



def plusnone(a,b):
    if (a is None) or (b is None):
        return None
    return a+b
    
def dictvals(d):
    try:
        return [x[0] for x in d.values()]
    except IndexError:
        return list( d.values() )
    except TypeError:
        return list( d.values() )

def bts(b,y="Y",n="N"):
    if b:
        return y
    return n


def tsmaller(v1,v2,y="Y",n="N",na="N/a"):
    if (v1 is not None) and (v2 is not None):
        return bts(v1<v2,y=y,n=n)
    return na
    
    
def mw_letter(d1,d2,pval=0.05,y="Y",n="N",na=None):
    l1=dictvals(d1)
    l2=dictvals(d2)
    try:
        return bts( mannwhitneyu(l1,l2).pvalue< pval, y=y,n=n)
    except ValueError as e:
        if na is None:
            return str(e)
        return na


def mw_letter_from_strings(s1,s2,pval=0.05,y="Y",n="N",na=None):
    if ("nan"==str(s1)) or ("nan"==str(s2)):
        if na is None:
            return "no value"
        return na
    return mw_letter(dict_from_string(s1),dict_from_string(s2),pval,y,n,na)

def dict_from_string(s):
    l = s.replace("[","").replace("]","").split("_")
    d = {x.split(":")[0]:float(x.split(":")[1]) for x in l}
    return d

def pointwise_kl(case,control,t):
    mean_control, var_control = control.gp.predict(np.asarray([[t]]))
    mean_case, var_case = case.gp.predict(np.asarray([[t]]))
    return  ((var_control + (mean_control - mean_case) ** 2) / (2 * var_case)) +  ((var_case + (mean_case - mean_control) ** 2) / (2 * var_control))    


def p_value(y,l2):
    #returns p-value for each x, based on l2
    return (len([x for x in l2 if x >= y]) + 1) / (len(l2) + 1)





def find_start_end(case,control):
    if control is None:
        start = case.find_start_date_index()
        end = case.measurement_end
    else:
        start = max(case.find_start_date_index(),control.measurement_start)
        end = min(case.measurement_end,control.measurement_end)        
        
    return start,end


def logna(x):
    if x is None:
        return 0
    return np.log(x)


def plot_gp(case,control,savename):
    
    start,end=find_start_end(case,control)
    plot_limits=[case.x[start][0],case.x[end-1][0]+1]
    fig, ax = plt.subplots() 
    

    
    plt.title("GP fits")
    plt.xlim(*plot_limits)
    plt.ylim([0,3.75])
    
    plt.xlabel("Time since start of experiment (days)")
    plt.ylabel("Log-normalized tumor size")
    
    control.gp.plot_data(ax=ax,color="blue")
    control.gp.plot_mean(ax=ax,color="blue",plot_limits=plot_limits,label="Control mean")
    control.gp.plot_confidence(ax=ax,color="blue",plot_limits=plot_limits,label="Control confidence")

    case.gp.plot_data(ax=ax,color="red")
    case.gp.plot_mean(ax=ax,color="red",plot_limits=plot_limits,label="Treatment mean")
    case.gp.plot_confidence(ax=ax,color="red",plot_limits=plot_limits,label="Treatment confidence")
    plt.savefig(savename)
    
   


def plot_category(case,control,means=None,savename="figure.pdf",normalised=True):
    """
    :param case: the category to be plotted. Not allowed to be None
    :param control : the corresponding control to be plotted. Can be None
    :paran meam:  whether the mean values across replicates are also plotted. Can be None
        (mean will not be plotted), "both" (mean is overlayed) or "only" 
        (only mean is plotted)
        
    :param normalised: If true, plots the normalised versions (case.y_norm). Otherwise case.y
    
    """
    case_y = case.y_norm if normalised else case.y
    
    if means not in [None,"only","both"]:
        raise ValueError("means must be None, 'only', or 'both'")
  
    start, end = find_start_end(case,control)
    if control is None:
#        start,end = case.find_start_date_index()
#        end = case.measurement_end
        high = case_y[:,start:end].max()
    else:
        control_y = control.y_norm if normalised else control.y        
        high = max(case_y[:,start:end].max(),control_y[:,start:end].max()) 
    low = min(case_y[:,start:end].min()*10,0)
    fig = plt.figure()  
    plt.ylim(low,high*1.05)
    plt.xlabel("Time since start of experiment (days)")
    if normalised:
        plt.ylabel("Log-normalized tumor size")
    else:
        plt.ylabel("Tumor size (mm3)")
    if means is None:
        plt.title("Replicates")
    elif means == "both":
        plt.title ("Replicates and mean")
    else:
        plt.title("Means")
    if means != "only":
        if case is not None:
            for (j,y_slice) in enumerate(case_y):
                if j==1:
                    s="treatment"
                else:
                    s="_treatment"
                plt.plot(case.x[start:end],y_slice[start:end],'.r-',label=s)
        if control is not None:
            for j,y_slice in enumerate(control_y):
                if j==1:
                    s="control"
                else:
                    s="_control"
                plt.plot(control.x[start:end],y_slice[start:end],'.b-',label=s)
    if means is not None:
        if means == "both":
            scase = ".k-"
            scontrol = ".k-"
        else:
            scase = ".r-"
            scontrol = ".b-"
        plt.plot(case.x[start:end],case_y.mean(axis=0)[start:end],scase,label="treatment")
        plt.plot(control.x[start:end],control_y.mean(axis=0)[start:end],scontrol,label="control")
    fig.legend(loc='upper left', bbox_to_anchor=(0.125,.875))#loc="upperleft"
#    fig.legend(loc=(0,0),ncol=2)#"upper left")
    fig.savefig(savename)
    return fig                           
                            
    



#######################PLOT EVERYTHING#########################
def plot_everything (outname,all_patients,stats_df,ag_df,fit_gp,p_val,p_val_kl,all_kl,tgi_thresh):
    #TO ADD: NICER LAYOUT - SET FIGURE SIZE BY SUBPLOT?
    #TO ADD: PLOT BEFORE CUT-OFF (to see whether it's just a single replicate)
    with PdfPages(outname) as pdf:
        for n,patient in enumerate(all_patients):
            control = patient.categories["Control"]
            for cat,cur_cat in patient.categories.items():
                if cat != "Control":
                    # TO ADD: SHOULD START ALSO CONTAIN control.measurement_start?!?
                    start = max(cur_cat.find_start_date_index(),cur_cat.measurement_start)
                    end = min(cur_cat.measurement_end,control.measurement_end)                    
                    name = str(patient.name)+"*"+str(cat)
#                    plt.figure(figsize = (24,18))
                                        
                    fig,axes=plt.subplots(4,2,figsize = (32,18))
                    fig.suptitle(name, fontsize="x-large")
                    axes[0,0].set_title("Replicates")

                    print("Now plotting patient", name)
                    for y_slice in cur_cat.y_norm:
                        axes[0,0].plot(cur_cat.x[start:end],y_slice[start:end],'.r-')
                    
                    if control.y_norm is None:
                        print("No control for patient %d, category %s"%(n, str(cat)))
                        print(patient)
                        print('----')
                    else:
                        for y_slice in control.y_norm:
                            axes[0,0].plot(control.x[start:end],y_slice[start:end],'.b-')
                            
                
                    axes[1,0].set_title("Means")
                    axes[1,0].plot(cur_cat.x[start:end],cur_cat.y_norm.mean(axis=0)[start:end],'.r-')    
                    if control.y_norm is not None:
                        axes[1,0].plot(control.x[start:end],control.y_norm.mean(axis=0)[start:end],'.b-')
#                    print("x")
#                    print(cur_cat.x[start:end].ravel())
#                    print("y")
#                    print([pointwise_kl(cur_cat,control,t).ravel()[0] for t in cur_cat.x[start:end].ravel()])
                    
                    axes[1,1].set_title("Pointwise KL divergence")
                    
                    if fit_gp:
                        axes[1,1].plot(cur_cat.x[start:end+1].ravel(),[pointwise_kl(cur_cat,control,t).ravel()[0] for t in cur_cat.x[start:end+1].ravel()],'ro')
                    else:
                        axes[1,1].axis("off")
                        axes[1,1].text(0.05,0.3,"no GP fitting, hence no KL values")
                    axes[2,0].set_title("GP plot: case")
                    axes[2,1].set_title("GP plot: control")
                    if fit_gp:
                        cur_cat.gp.plot(ax=axes[2,0])
                        pl.show(block=True)
                        control.gp.plot(ax=axes[2,1])
                        pl.show(block=True)
                    else:
                        for axis in [axes[2,0],axes[2,1]]:
                            axis.text(0.05,0.3,"not currently plotting GP fits")
                        
                    axes[3,0].axis("off")
                    txt =[]               
                    mrlist = [str(stats_df.loc[name,mr]) for mr in ["num_mCR","num_mPR","num_mSD","num_mPD"]]
                    txt.append("mRECIST: ("+",".join(mrlist))
                    for col in ["kl","response_angle_rel","response_angle_rel_control","auc_norm","auc_control_norm","tgi"]:
                        txt.append(col+": "+str(stats_df.loc[name,col]))
                    
                    #TO ADD: MAYBE BETTER AGGREGATE DATA?
                    txt.append( "red = treatment,       blue=control")
                    axes[3,0].text(0.05,0.3,'\n'.join(txt))
                    
                    axes[0,1].axis("off")
                    rtl = []
                    rtl.append("KuLGaP: " + tsmaller (p_value(cur_cat.kl_divergence,all_kl), p_val_kl))
                    
                    rtl.append( "KuLGaP2: "+ bts(cur_cat.kl_p_cvsc<p_val) )
                    rtl.append( "mRECIST (Novartis): "+ tsmaller(stats_df.loc[name,"perc_mPD"],0.5) )
                    
                    rtl.append( "mRECIST (ours): " + tsmaller(plusnone(stats_df.loc[name,"perc_mPD"] , stats_df.loc[name,"perc_mSD"]), 0.5) )
                    
                    rtl.append( "Angle: "+mw_letter(cur_cat.response_angle_rel, cur_cat.response_angle_rel_control,pval=p_val) )
                    
                    rtl.append( "AUC: " +mw_letter(cur_cat.auc_norm,cur_cat.auc_control_norm,pval=p_val ) )
                    
                    rtl.append( "TGI: "+tsmaller(tgi_thresh,cur_cat.tgi) )
#                    not yet implemented" )
                     # TO ADD: TGI                   
                    resp_text = "\n".join(rtl)
                    axes[0,1].text(0.05,0.3,resp_text,fontsize=20 )

                    
                    
                    
                    pdf.savefig(fig)
                    plt.close()


                        
###################################END PLOT_EVERYTHING############################  


def nfv(x):
    if ((x<1 and np.random.randint(0,5)<3) or x>25):
        return np.random.randint(0,3)
    return x





def get_classification_df_from_df(stats_df,p_val,all_kl,p_val_kl,tgi_thresh):
    responses =stats_df.copy()[["kl"]]
    ### TODO: FIX - kl_p_cvsc is still wrong in Sheng dataset (crown_df)

    responses["KuLGaP"] = stats_df.kl_p_cvsc.apply(lambda x: tsmaller(x,p_val,y=1,n=-1,na=0))
    responses["KuLGaP-prev"] = stats_df.kl.apply(lambda x: tsmaller(p_value(x,all_kl), p_val_kl,y=1,n=-1,na=0) )
    responses["mRECIST-Novartis"] = stats_df.perc_mPD.apply(lambda x: tsmaller(x,0.5,y=1,n=-1,na=0) )
    responses["mRECIST-ours"] = stats_df.apply(lambda row: tsmaller(plusnone(row["perc_mPD"] , row["perc_mSD"]), 0.5,y=1,n=-1,na=0) ,axis=1)

    responses["Angle"] = stats_df.apply(lambda row: mw_letter_from_strings(row["response_angle_rel"], row["response_angle_rel_control"],pval=p_val,y=1,n=-1,na=0) , axis=1)
    responses["AUC"] = stats_df.apply(lambda row: mw_letter_from_strings(row["auc_norm"],row["auc_control_norm"],pval=p_val,y=1,n=-1,na=0 ) , axis=1)
    responses["TGI"] = stats_df.TGI.apply(lambda x: tsmaller(tgi_thresh,x,y=1,n=-1,na=0)) 
    responses.drop("kl",axis=1,inplace=True)
    return responses
    
    
    

def get_classification_dict(all_patients,stats_df,p_val,all_kl,p_val_kl,tgi_thresh):
    predict = {"KuLGaP": [], "AUC":[],"Angle":[],"mRECIST_Novartis":[],"mRECIST_ours":[],
               "TGI":[]}
    for n,patient in enumerate(all_patients):
        for cat,cur_cat in patient.categories.items():
            if cat != "Control":
                name = str(patient.name)+"*"+str(cat)
                predict["KuLGaP"].append(tsmaller (p_value(cur_cat.kl_divergence,all_kl), p_val_kl,y=1,n=-1,na=0))                
                predict["mRECIST_Novartis"].append( tsmaller(stats_df.loc[name,"perc_mPD"],0.5,y=1,n=-1,na=0) )
                predict["mRECIST_ours"].append( tsmaller(plusnone(stats_df.loc[name,"perc_mPD"] , stats_df.loc[name,"perc_mSD"]), 0.5,y=1,n=-1,na=0) )
                predict["Angle"].append( mw_letter(cur_cat.response_angle_rel, cur_cat.response_angle_rel_control,pval=p_val,y=1,n=-1,na=0) )
                predict["AUC"].append(mw_letter(cur_cat.auc_norm,cur_cat.auc_control_norm,pval=p_val,y=1,n=-1,na=0 ) )
                predict["TGI"].append(tsmaller(tgi_thresh,cur_cat.tgi,y=1,n=0,na=2))
    return predict




def get_classification_df(all_patients,stats_df,p_val,all_kl,p_val_kl,tgi_thresh):
    predict = {}
#    predict = {"KuLGaP": [], "AUC":[],"Angle":[],"mRECIST_Novartis":[],"mRECIST_ours":[]}
    for n,patient in enumerate(all_patients):
        for cat,cur_cat in patient.categories.items():
            if cat != "Control":
                name = str(patient.name)+"*"+str(cat)
                
                predict[name] = [tsmaller(cur_cat.kl_p_cvsc,p_val,y=1,n=-1,na=0)]
                #predict[name] = [tsmaller (p_value(stats_df.loc[name,"kl"],all_kl), p_val_kl,y=1,n=-1,na=0), ]
                #KuLGaP-prev
                print(patient.name,cat)
                print(str(cur_cat.kl_divergence))
                predict[name].append (tsmaller (p_value(cur_cat.kl_divergence,all_kl), p_val_kl,y=1,n=-1,na=0), )
                #MRECIST_Novartis
    
                predict[name].append( tsmaller(stats_df.loc[name,"perc_mPD"],0.5,y=1,n=-1,na=0) )
                #MRECIST_ours
                predict[name].append( tsmaller(plusnone(stats_df.loc[name,"perc_mPD"] , stats_df.loc[name,"perc_mSD"]), 0.5,y=1,n=-1,na=0) )
                #angle
                predict[name].append( mw_letter(cur_cat.response_angle_rel, cur_cat.response_angle_rel_control,pval=p_val,y=1,n=-1,na=0) )
                #AUC
                predict[name].append(mw_letter(cur_cat.auc_norm,cur_cat.auc_control_norm,pval=p_val,y=1,n=-1,na=0 ) )
                predict[name].append(tsmaller(tgi_thresh,cur_cat.tgi,y=1,n=-1,na=0))
    df = pd.DataFrame.from_dict(predict,orient="index", columns = ["KuLGaP","KuLGaP-prev","mRECIST_Novartis","mRECIST_ours","Angle","AUC","TGI"])
    return df






def create_and_plot_agreements(classifiers_df,agreements_outfigname,agreements_outname):
    agreements = create_agreements(classifiers_df)
    agreements.to_csv(agreements_outname)
    paper_list=["KuLGaP","TGI", "mRECIST","AUC","Angle"]
    ag2=agreements[paper_list].reindex(paper_list)
    print(ag2)
    plt.figure()
    sns.heatmap(ag2, vmin=0, vmax=1,center=.5,square=True,annot=ag2,cbar=False, linewidths=.3,linecolor="k",cmap="Greens")
#    sns.heatmap(agreements, vmin=0, vmax=1, center=0,square=True,annot=agreements,cbar=False)
    plt.savefig(agreements_outfigname)
    
    
def create_and_plot_conservative(classifiers_df,conservative_outfigname,conservative_outname):
    conservative = create_conservative(classifiers_df)
    conservative.to_csv(conservative_outname)
    paper_list=["KuLGaP","TGI", "mRECIST","AUC","Angle"]
    con2=conservative[paper_list].reindex(paper_list)
    plt.figure()
    sns.heatmap(con2,cmap="coolwarm",square=True,annot=con2,
            cbar=False,linewidths=.3,linecolor="k",vmin=-.8,vmax=.8,center=-0.1)
    #sns.heatmap(conservative, square=True,annot=conservative.round(2),cmap="coolwarm",cbar=False)
    plt.savefig(conservative_outfigname)
    
    
    
    
def create_and_plot_FDR(classifiers_df,FDR_outfigname,FDR_outname):
    FDR = create_FDR(classifiers_df)
    FDR.to_csv(FDR_outname)
    paper_list=["KuLGaP","TGI", "mRECIST","AUC","Angle"]
    FDR = FDR[paper_list].reindex(paper_list)
    plt.figure()
    sns.heatmap(FDR,cmap="coolwarm",square=True,annot=FDR,
            cbar=False,linewidths=.3,linecolor="k",vmin=-.8,vmax=.8,center=-0.1)
    plt.savefig(FDR_outfigname)
    

def create_and_save_KT(classifiers_df,KT_outname):
    kts=create_KT(classifiers_df)
    print(kts)
    kts.to_csv(KT_outname)    
    


def plot_histogram(l, varname,marked=None,savename=None,smoothed=None,x_min=None,x_max=None,dashed=None,solid=None):
    """
    Plots the histogram of var, with an asterix and an arrow at marked
    Labels the x axis according to varname
    :param var: pandas Series object
    """
    fig=plt.figure()
    var=pd.Series(l)
    var.dropna().hist(bins=30,grid=False,density=True)
    if smoothed is not None:
        x=np.linspace(x_min,x_max,1000)
        plt.plot(x,smoothed(x),"-r")
    plt.xlabel(varname)
    plt.ylabel("frequency")
    if marked is not None:
        plt.plot(marked,.02,marker="*",c="r")
        style="Simple,tail_width=0.5,head_width=4,head_length=8"
        kw = dict(arrowstyle=style, color="k")
        plt.text(11,.2,"critical value")
        arrow=mp.FancyArrowPatch(posA=[11,.2],posB=[marked+.25,0.035] ,connectionstyle="arc3,rad=-.25",**kw  )
        plt.gca().add_patch(arrow)
    if dashed is not None:
        for val in dashed:
            ax=plt.gca()
            ax.axvline(x=val, color='black', linestyle="--")   
    if solid is not None:
        for val in solid:
            ax = plt.gca()
            ax.axvline(x=val, color='black', linestyle="-")  # critical value for p-val=0.1
        
    plt.savefig(savename)
    return fig
    
    
    
def create_scatterplot(stats_df,classifiers_df,savename) :
    
    #deprecated, previous way to plot figure 2C.
    df = stats_df[["kl"]]
    df.loc[:,"kl_p"] = stats_df.kl_p_cvsc
    df.loc[:,"Ys"] = classifiers_df.drop("KuLGaP",axis=1).apply(lambda row: row[row==1].count(), axis=1)
    
    plt.figure()
    plt.ylim(0,5)
    plt.plot(df.kl.apply(logna),df.Ys,'r',marker=".",markersize=2,linestyle="")
    c=np.log(7.97)
    plt.plot([c,c], [0,5], 'k-', lw=1)
    c=np.log(5.61)
    plt.plot([c,c], [0,5], 'k--', lw=1)
    c=np.log(13.9)
    plt.plot([c,c], [0,5], 'k--', lw=1)
    plt.xlabel("Log(KL)")
    plt.ylabel('Number of measures that agree on a "responder" label')
    plt.ylim(-0.2,4.2)
    plt.yticks(ticks=[0,1,2,3,4])
    plt.savefig(savename)
    
    
def plot_histograms_2c(stats_df,classifiers_df,savename):
    data = stats_df[["kl"]]
    data.loc[:,"klval"] = stats_df.kl.apply(logna)
    data.loc[:,"count"] = classifiers_df.drop("KuLGaP",axis=1).apply(lambda row: row[row==1].count(), axis=1)
    
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
        ax.axvline(x=np.log(7.97), color='black', linestyle="-")   # critical value for p-val=0.05 
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
    
    
    
