import numpy as np
import pandas as pd
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