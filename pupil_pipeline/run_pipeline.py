"""
RUN PIPELINE
============

Main entry point of the pupillometry preprocessing pipeline.

This script orchestrates the complete analysis workflow but does NOT
implement preprocessing itself.

Pipeline logic:
---------------
For each subject, session, and task:

    1. Load raw eye-tracking recording
    2. Apply global pupil preprocessing
    3. Extract stimulus-evoked pupil responses
    4. Extract saccade and fixation metrics
    5. Save processed outputs

After all recordings are processed:
    → group-level plotting is performed.

Important design principles:
----------------------------
- No analysis parameters are defined here.
- All parameters must be set in `config.py`.
- This file should remain stable across experiments.

Typical usage:
--------------
> python run_pipeline.py
"""
import mne
from config import CONFIG

from data_io.data_loading import load_recording, list_subjects
from data_io.saving import save_results, save_epoch_average, save_preprocessing_overview

from preprocessing.pupil_preprocess import preprocess_pupil
from preprocessing.evoked_processing import extract_evoked, epoch_pupil, compute_blink_pct_raw, flag_blink_in_baseline
from preprocessing.saccade_processing import extract_saccades

from plotting.group_plots import plot_group_results
from plotting.individual_plots import plot_raw_recording, plot_pupil_df, plot_evoked, plot_saccade_summary, plot_signal_comparison
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def run():

    results  = []
    preproc_records = []  # ← collect preprocessing stats here

    beh_data = pd.read_csv(CONFIG["BEH_FILE"])

    # Load group map once for labelling
    group_df  = pd.read_csv(CONFIG["GROUP_FILE"])
    group_map = (
        group_df[["sub_id", "group"]]
        .drop_duplicates()
        .set_index("sub_id")["group"]
        .to_dict()
    )

    def get_group(subject):
        sub_num = int(subject.split("-")[1])
        for key in [f"sub-{sub_num:02d}", sub_num, f"sub-{sub_num}", subject]:
            if key in group_map:
                return group_map[key]
        return None

    for subject in list_subjects(CONFIG):
        for session in CONFIG["SESSIONS"]:
            for task in CONFIG["TASKS"]:

                raw = load_recording(subject, session, task, CONFIG)
                if raw is None:
                    continue

                # Compute blink % from raw annotations before anything touches the signal
                pct_blink_raw = compute_blink_pct_raw(raw)

                # -------------------------
                # 1. GLOBAL PUPIL CLEANING
                # -------------------------
                pupil, pct_blink_interp, pct_despiked = preprocess_pupil(raw, CONFIG)

                # -------------------------
                # 2. STIMULUS-EVOKED
                # -------------------------
                evoked = extract_evoked(pupil, raw, "pupil_int_despiked_lp", CONFIG)

                # blink_in_baseline column is now in evoked;
                # derive the session-level percentage from it
                n_trials    = len(evoked)
                n_blink_bl  = evoked["blink_in_baseline"].sum()
                pct_blink_baseline = (n_blink_bl / n_trials * 100) if n_trials > 0 else np.nan

                epochs, times, block_indices = epoch_pupil(
                    pupil, raw, "pupil_int_despiked_lp", CONFIG
                )
                save_epoch_average(subject, session, task, epochs, times, CONFIG)

                # -------------------------
                # 3. SACCADES / FIXATIONS
                # -------------------------
                saccades = extract_saccades(
                    pupil, raw, beh_data, subject, session, task, CONFIG
                )

                # -------------------------
                # 4. SAVE RESULTS
                # -------------------------
                save_results(subject, session, task, evoked, saccades, CONFIG)
                results.append((evoked, saccades))

                preproc_records.append({
                    "subject":                    subject,
                    "session":                    session,
                    "task":                       task,
                    "group":                      get_group(subject),
                    "pct_blink_interp":           pct_blink_interp,
                    "pct_despiked":               pct_despiked,
                    "pct_blink_raw":              pct_blink_raw,
                    "pct_trials_blink_in_baseline": pct_blink_baseline,
                })

    # -------------------------
    # 5. GROUP PLOTTING
    # -------------------------
    plot_group_results(CONFIG)

    # -------------------------
    # 6. PREPROCESSING OVERVIEW
    # -------------------------
    save_preprocessing_overview(preproc_records, CONFIG)



# def run():

#     results = []

#     beh_data = pd.read_csv(CONFIG["BEH_FILE"])

#     for subject in list_subjects(CONFIG):
#         for session in CONFIG["SESSIONS"]:
#             for task in CONFIG["TASKS"]:

#                 raw = load_recording(subject, session, task, CONFIG)
#                 # plot_raw_recording(raw, subject, session, task)

#                 if raw is None:
#                     continue

#                 # -------------------------
#                 # 1. GLOBAL PUPIL CLEANING
#                 # -------------------------
#                 pupil = preprocess_pupil(raw, CONFIG)
#                 # plot_pupil_df(pupil, "pupil_int", subject, session, task)
#                 # plot_pupil_df(pupil, "pupil_int_despiked", subject, session, task)
#                 # plot_pupil_df(pupil, "pupil_int_bp", subject, session, task)
#                 # plot_pupil_df(pupil, "pupil_int_lp_medianpsc", subject, session, task)

#                 # plot_signal_comparison(
#                 #     pupil_df=pupil,
#                 #     columns=["pupil_int", "pupil_int_despiked"],
#                 #     labels=["After blink interpolation", "After despiking"],
#                 #     title="sub-02 | ses-frontal | 2back",
#                 #     show_diff=True
#                 # )


#                 # -------------------------
#                 # 2. STIMULUS-EVOKED
#                 # -------------------------
#                 evoked = extract_evoked(pupil, raw, "pupil_int_despiked_lp", CONFIG)

#                 epochs, times, block_indices = epoch_pupil(pupil, raw, "pupil_int_despiked_lp", CONFIG)
#                 # plot_evoked(epochs, times, block_indices, "pupil_int_despiked_lp", subject, session, task)
#                 save_epoch_average(subject, session, task, epochs, times, CONFIG)

#                 # -------------------------
#                 # 3. SACCADES / FIXATIONS
#                 # -------------------------
#                 saccades = extract_saccades(pupil, raw, beh_data, subject, session, task, CONFIG)
#                 # plot_saccade_summary(pupil, saccades, beh_data, subject, session, task, CONFIG)
                
#                 # -------------------------
#                 # 4. SAVE RESULTS
#                 # -------------------------

#                 save_results(subject, session, task, evoked, saccades, CONFIG)
#                 results.append((evoked, saccades))

#     # -------------------------
#     # 5. GROUP PLOTTING
#     # -------------------------
#     plot_group_results(CONFIG)


if __name__ == "__main__":
    run()