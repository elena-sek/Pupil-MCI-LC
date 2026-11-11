"""
GLOBAL PUPIL PREPROCESSING
==========================

Implements preprocessing applied to the continuous pupil signal.

This stage operates on the FULL recording before any event-related
analysis is performed.

Typical preprocessing steps:
----------------------------
- Eye selection or averaging
- Removal of invalid samples
- Blink detection and interpolation
- Temporal filtering
- Optional normalization
- Downsampling

Critical concept:
-----------------
This stage produces ONE canonical cleaned pupil signal.

After preprocessing:
    → downstream analyses MUST NOT access raw pupil data.

Preprocessing parameters are defined exclusively in `config.py`.
"""
"""
PUPIL PREPROCESSING PIPELINE
=============================

Orchestrates all global signal preprocessing steps.

Responsibilities:
-----------------
- Build the pupil DataFrame from the raw MNE object
- Apply preprocessing steps in order
- Track the current pupil column across optional steps
- Return a fully preprocessed DataFrame with one column per step

Design principles:
------------------
- No signal processing logic lives here — all steps are in pupil_utils.py.
- No event-related processing (epoching, evoked responses) occurs here.
- All parameters come from CONFIG.
- Each preprocessing step adds a new column; no columns are overwritten.
"""
import numpy as np
import pandas as pd

from preprocessing.pupil_utils import (
    interpolate_blinks,
    temporal_filter,
    regress_blinks,
    median_psc,
    despike_pupil
)


def preprocess_pupil(raw, CONFIG):
    """
    Preprocess the pupil signal from a raw MNE object.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw object with stimulus onsets already injected into annotations.
    CONFIG : dict
        Pipeline configuration.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with columns for each preprocessing step and a
        'final_pupil_column' entry in CONFIG updated to reflect the
        last processed column.
    """

    fs = raw.info['sfreq']

    print("CHECKING ANNOTATIONS:")
    print(raw.annotations.to_data_frame()["description"].unique())
    print(raw.annotations.to_data_frame().columns.tolist())

    # -------------------------
    # BUILD BASE DATAFRAME
    # -------------------------
    df_raw  = raw.to_data_frame()
    pupil_col = [c for c in df_raw.columns if 'pupil' in c.lower()][0]
    gaze_x_col = [c for c in df_raw.columns if 'xpos' in c.lower()][0]
    gaze_y_col = [c for c in df_raw.columns if 'ypos' in c.lower()][0]

    df = pd.DataFrame()
    df['time']      = df_raw['time']
    df['pupil_raw'] = df_raw[pupil_col].copy()
    df['gaze_x']    = df_raw[gaze_x_col].copy()
    df['gaze_y']    = df_raw[gaze_y_col].copy()

    current_pupil = 'pupil_raw'

    # -------------------------
    # 1. BLINK INTERPOLATION
    # -------------------------
    interpolate_blinks(raw, buffer_s=CONFIG["BLINK_INTERP_S"])
    df_int = raw.to_data_frame()
    df['pupil_int'] = df_int[pupil_col].copy()
    current_pupil = 'pupil_int'

    # -------------------------
    # 2. SPIKE REMOVAL
    # -------------------------
    if CONFIG["SPIKE_REMOVAL"]:
        despike_pupil(
            df=df,
            measure="pupil_int",
            fs=fs,
            n=CONFIG["SPIKE_MAD_THRESHOLD"],
            trend_window_ms=CONFIG["SPIKE_TREND_WINDOW_MS"],
            use_trendline_filter=CONFIG["SPIKE_USE_TRENDLINE_FILTER"],
        )
        current_pupil = "pupil_int_despiked"

    # -------------------------
    # 3. TEMPORAL FILTERING
    # -------------------------
    temporal_filter(
        df=df,
        measure=current_pupil,
        fs=fs,
        lp=CONFIG["LOWPASS_HZ"],
        hp=CONFIG["HIGHPASS_HZ"],
        order=3,
    )
    # temporal_filter produces both _lp and _bp in one call
    current_pupil = f'{current_pupil}_lp' 

    # -------------------------
    # 3. BLINK REGRESSION
    # -------------------------
    # TODO: SMOOTHING_MS not yet implemented — placeholder for future step

    if CONFIG["BLINK_REGRESS"]:
        events = raw.annotations.to_data_frame()
        events['onset'] = raw.annotations.onset
        regress_blinks(
            df=df,
            events=events,
            current_pupil=current_pupil,
            fs=fs,
            regress_sacs=CONFIG["SACCADE_REGRESS"]
        )
        current_pupil = f'{current_pupil}_clean'

    # -------------------------
    # 4. PERCENT SIGNAL CHANGE
    # -------------------------
    median_psc(df, measure=current_pupil)
    # current_pupil = f'{current_pupil}_medianpsc'

    # -------------------------
    # 5. DOWNSAMPLING
    # -------------------------
    step = int(fs / CONFIG["DOWNSAMPLE_HZ"])
    df = df.iloc[::step].reset_index(drop=True)

    # Store the final column name in CONFIG so downstream functions
    # always know which column to epoch from
    CONFIG["FINAL_PUPIL_COLUMN"] = current_pupil

    # NEW ADDITION
    # -------------------------
    # TRACK PREPROCESSING STATS
    # -------------------------
    # % blink-interpolated: samples that changed between raw and interpolated
    n_total     = len(df)
    n_blink_interp = int(np.sum(np.abs(df["pupil_int"] - df["pupil_raw"]) > 1e-6))
    pct_blink_interp = n_blink_interp / n_total * 100

    # % despiked: already printed inside despike_pupil, but recompute here
    if CONFIG["SPIKE_REMOVAL"]:
        n_despiked  = int(np.sum(
            ~np.isnan(df["pupil_int"]) &
            (np.abs(df["pupil_int_despiked"] - df["pupil_int"]) > 1e-6)
        ))
        pct_despiked = n_despiked / n_total * 100
    else:
        pct_despiked = 0.0

    print(f"  ✓ Preprocessing complete. Final column: {current_pupil}")
    print(f"  ✓ DataFrame shape after downsampling: {df.shape}")

    return df, pct_blink_interp, pct_despiked  # ← was: return df

    # return df