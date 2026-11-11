"""
PIPELINE CONFIGURATION
======================

Central configuration file controlling the entire preprocessing pipeline.

All preprocessing and analysis parameters must be defined here.

Purpose:
--------
This file enables reproducible preprocessing experiments by allowing
one parameter to be modified while keeping the rest of the pipeline fixed.

DO NOT hardcode parameters inside preprocessing functions.

Categories of parameters:
-------------------------
- Data locations
- Subjects / sessions / tasks
- Global pupil preprocessing settings
- Epoch extraction settings
- Baseline correction definitions

Recommended workflow:
---------------------
1. Define a DEFAULT preprocessing pipeline.
2. Modify ONE parameter at a time.
3. Re-run `run_pipeline.py`.
4. Compare outputs across configurations.

This file represents the methodological specification of the study.
"""
from pathlib import Path

CONFIG = dict(
    # ------------------------
    # EXPERIMENTAL SETTINGS
    # ------------------------
    SCREEN_RESOLUTION=(1920, 1200),
    VIEWING_DISTANCE_CM=78,
    SCREEN_WIDTH_CM=52,
    SCREEN_HEIGHT_CM=32.5,
    STIMULUS_SIZE_PCT=(0.08, 0.13),  # width, height as proportion of screen

    # ------------------------
    # DATA - When analyzing task data
    # ------------------------
    RAW_ROOT=Path(r"C:\Users\582487\OneDrive - MUNI\PhD\CRstim\data\raw"),
    OUTPUT_ROOT=Path(r"C:\Users\582487\OneDrive - MUNI\PhD\CRstim\data\preproc_v2_results"),
    BEH_FILE=Path(r"C:\Users\582487\OneDrive - MUNI\PhD\CRstim\data\raw\combined_nback_data_full.csv"),
    GROUP_FILE=Path(r"C:\Users\582487\OneDrive - MUNI\PhD\CRstim\data\group_dataframes\task_df.csv"),

    SUBJECTS= None, # ["sub-03", "sub-15"], # None, # ["sub-03"],# None,  # None = all
    SESSIONS= ["ses-placebo"], # ["ses-placebo","ses-frontal","ses-frontoparietal"], #["ses-placebo"],   # ["ses-placebo","ses-frontal","ses-frontoparietal"],
    TASKS=["2back","3back"], # ["2back","3back"],

    # ------------------------
    # DEFAULT PREPROCESSING
    # ------------------------
    BLINK_INTERP_TYPE="linear",
    BLINK_INTERP_S=0.1,   # seconds #TO-DO → create a tuple out of this so that the two intervals can differ
    BLINK_REGRESS=False,
    SACCADE_REGRESS=False, # works only if BLINK_REGRESS is True
    LOWPASS_HZ=8, # 10
    HIGHPASS_HZ= False, # 0.01, # False, # 0.01
    SMOOTHING_MS=None,
    DOWNSAMPLE_HZ=100,
    SPIKE_REMOVAL=True, # True,
    SPIKE_MAD_THRESHOLD=2.5, # 3 # number of median absolute deviations (MAD) above which a point is considered a spike, larger → fewer rejections
    SPIKE_USE_TRENDLINE_FILTER=False, # enables two-pass trendline deviation rejection → False for now because it got rid of my signal too
    SPIKE_TREND_WINDOW_MS=2000, # ms 

    # ------------------------
    # EVOKED SETTINGS
    # ------------------------
    EPOCH_WINDOW=(-0.5, 3.0),
    BASELINE_WINDOW=(-0.05, 0), # (-0.5, 0) # If baseline starts before stimulus onset, the first number will be negative
    PEAK_WINDOW=(0.2, 2), # window where to search for the peak pupil amplitude

    # ------------------------
    # SACCADE SETTINGS
    # ------------------------    
    SACCADE_TEMPORAL_WINDOW=(0.08, 1.5),   # seconds after stimulus onset
    SACCADE_SPATIAL_CRITERION_DEG=4.0     # degrees of visual angle
)

# # 1. Your fixed baseline — the "canonical" preprocessing
# DEFAULTS = {
#     "interpolation_limit": 100,   # ms
#     "interpolation_type":"linear",   # ms
#     "blink_regression": False,
#     "baseline_window":[-500, 0],   # ms
#     "filter_high_pass": 0,
#     "filter_low_pass": 10,
#     "downsample_rate": 100,        # ms
#     "smoothing": None                     # ms
# }

# # 2. Variations — each entry changes ONE key from DEFAULTS
# EXPERIMENTS = {
#     # label              param to vary       values to try
#     "blink_reg":     ("blink_regression",    [True]),
#     "high_pass":     ("filter_high_pass",    [0, 0.01, 0.1]),
#     "low_pass":      ("filter_low_pass",     [6, 8, None]),
#     "smooth_window": ("smoothing",           [50, 100, 200]),
#     "baseline":      ("baseline_window",     [[-250, 0], [-50, 0], [0, 50]]),
# }

