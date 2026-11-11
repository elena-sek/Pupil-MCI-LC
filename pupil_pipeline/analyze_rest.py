"""
ANALYZE RESTING STATE DATA
============

The preprocessing parameters are defined in the config.py file.
They should be consistent across the whole experiment.
→ Part # DEFAULT PREPROCESSING

Typical usage:
--------------
> python analyze_rest.py
"""
import mne
from config import CONFIG

from data_io.data_loading import load_rest_recording, list_subjects
from data_io.saving import save_rest_results

from preprocessing.pupil_preprocess import preprocess_pupil
from preprocessing.rest_processing import extract_rest_metrics

from plotting.individual_plots import plot_raw_recording, plot_pupil_df, plot_signal_comparison
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



def run():

    for subject in list_subjects(CONFIG):

        raw = load_rest_recording(subject, CONFIG)
        # plot_raw_recording(raw, subject)

        if raw is None:
            continue

        # -------------------------
        # 1. GLOBAL PUPIL CLEANING
        # -------------------------
        pupil, _, _ = preprocess_pupil(raw, CONFIG)
        # plot_pupil_df(pupil, "pupil_int", subject)
        # plot_pupil_df(pupil, "pupil_int_despiked", subject, session, task)
        # plot_pupil_df(pupil, "pupil_int_bp", subject, session, task)
        # plot_pupil_df(pupil, "pupil_int_lp_medianpsc", subject, session, task)

        # print(pupil.columns)

        # plot_signal_comparison(
        #     pupil_df=pupil,
        #     columns=["pupil_int", "pupil_int_despiked"],
        #     labels=["After blink interpolation", "After despiking"],
        #     title="sub-02 | ses-frontal | 2back",
        #     show_diff=True
        # )
        
        # -------------------------
        # 2. EXTRACT REST METRICS & SAVE RESULTS
        # -------------------------
        rest_mean, rest_SD, rest_CV = extract_rest_metrics(pupil, "pupil_int_despiked_lp", CONFIG)
        save_rest_results(subject, rest_mean, rest_SD, rest_CV, CONFIG)


if __name__ == "__main__":
    run()