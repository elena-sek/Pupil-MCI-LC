"""
ANALYSE ONE RECORDING SESSION OF ONE SUBJECT
============================================================================

Specific for our visuospatial n-back task

"""
import mne
from config import CONFIG

from data_io.data_loading import load_recording, list_subjects
from data_io.saving import save_results, save_epoch_average

from preprocessing.pupil_preprocess import preprocess_pupil
from preprocessing.evoked_processing import extract_evoked, epoch_pupil
from preprocessing.saccade_processing import extract_saccades

from plotting.group_plots import plot_group_results
from plotting.individual_plots import plot_raw_recording, plot_pupil_df, plot_evoked, plot_saccade_summary, plot_signal_comparison
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FILENAME = r"D:\PhD\testing\Test1404.asc"
BEH_FILE = r"C:\Users\582487\OneDrive - MUNI\PhD\CRstim\data\raw\combined_nback_data_full.csv"

