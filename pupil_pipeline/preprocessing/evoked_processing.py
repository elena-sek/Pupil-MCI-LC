"""
EVOKED PUPIL RESPONSE EXTRACTION
=================================

Extracts stimulus-evoked pupil response metrics from preprocessed data.

Responsibilities:
-----------------
- Extract block-level fixation baseline metrics
- Extract trial-level evoked response metrics
- Compute peak amplitude, latency, TEPR, and velocity (three methods)

Design principles:
------------------
- No preprocessing occurs here.
- No subject/session/task identifiers are added here — that is save_results' responsibility.
- All window parameters come from CONFIG.
"""

import numpy as np
import pandas as pd
import scipy as sp


def extract_evoked(pupil_df, raw, pupil_col, CONFIG):

    # -------------------------
    # EXTRACT EVENTS
    # -------------------------
    event_df = raw.annotations.to_data_frame()
    event_df["onset"] = raw.annotations.onset

    stim_events = event_df[
        event_df["description"].str.contains("Stimulus", case=False, na=False)
    ].sort_values("onset")

    block_events = event_df[
        event_df["description"].str.contains("Trial Ready", case=False, na=False)
    ].sort_values("onset")

    if stim_events.empty:
        print("    ⚠️ No Stimulus events found")
    if block_events.empty:
        print("    ⚠️ No Trial Ready events found")

    # -------------------------
    # VELOCITY
    # -------------------------
    fs = CONFIG["DOWNSAMPLE_HZ"]

    velocity_diff     = np.concatenate([[0], np.diff(pupil_df[pupil_col].values)]) * fs
    velocity_gradient = np.gradient(pupil_df[pupil_col].values) * fs
    velocity_savgol   = sp.signal.savgol_filter(
                            pupil_df[pupil_col].values,
                            window_length=11, polyorder=3, deriv=1, delta=1/fs
                        )

    # -------------------------
    # BLOCK-LEVEL METRICS
    # Fixation period = 10s after each Trial Ready marker
    # This is specific to the CRstim n-back task timing
    # -------------------------
    block_rows = []
    for block_idx, block_onset in enumerate(block_events["onset"].values):
        block_start = block_onset
        block_end   = block_onset + 10.0
        block_mask  = (
            (pupil_df["time"] >= block_start) &
            (pupil_df["time"] <  block_end)
        )
        block_values = pupil_df.loc[block_mask, pupil_col].dropna()

        if block_values.empty:
            fix_mean = fix_sd = fix_cv = np.nan
        else:
            fix_mean = block_values.mean()
            fix_sd   = block_values.std(ddof=1)
            fix_cv   = (fix_sd / fix_mean) * 100 if fix_mean != 0 else np.nan

        block_rows.append({
            "block_index":          block_idx + 1,
            "block_onset":          block_onset,
            "fix_mean_pupil_size":  fix_mean,
            "fix_pupil_sd":         fix_sd,
            "fix_pupil_cv":         fix_cv,
        })

    block_df = pd.DataFrame(block_rows)

    # -------------------------
    # TRIAL-LEVEL METRICS
    # -------------------------
    # -------------------------
    # BLINK IN BASELINE FLAGS
    # Computed here so the column is available in the returned trial_df
    # -------------------------
    baseline_flags, _ = flag_blink_in_baseline(pupil_df, raw, CONFIG)
    # Pad or trim to match number of stim events in case of length mismatch
    while len(baseline_flags) < len(stim_events):
        baseline_flags.append(None)

    trial_rows = []
    for trial_idx, stim_onset in enumerate(stim_events["onset"].values):

        baseline_start = stim_onset + CONFIG["BASELINE_WINDOW"][0]
        baseline_end   = stim_onset + CONFIG["BASELINE_WINDOW"][1]
        peak_start     = stim_onset + CONFIG["PEAK_WINDOW"][0]
        peak_end       = stim_onset + CONFIG["PEAK_WINDOW"][1]

        baseline_mask = (
            (pupil_df["time"] >= baseline_start) &
            (pupil_df["time"] <  baseline_end)
        )
        peak_mask = (
            (pupil_df["time"] >= peak_start) &
            (pupil_df["time"] <  peak_end)
        )

        baseline_values = pupil_df.loc[baseline_mask, pupil_col].dropna()
        peak_values     = pupil_df.loc[peak_mask,     pupil_col].dropna()

        if baseline_values.empty:
            mean_baseline = baseline_sd = baseline_cv = np.nan
        else:
            mean_baseline = baseline_values.mean()
            baseline_sd   = baseline_values.std(ddof=1)
            baseline_cv   = (baseline_sd / mean_baseline) * 100 if mean_baseline != 0 else np.nan

        peak_amp = peak_values.max() if not peak_values.empty else np.nan
        tepr     = peak_amp - mean_baseline if not np.isnan(peak_amp) else np.nan

        # Peak time and latency
        if peak_values.empty:
            peak_time    = np.nan
            peak_latency = np.nan
        else:
            peak_idx     = peak_values.idxmax()
            peak_time    = pupil_df.loc[peak_idx, "time"]
            peak_latency = peak_time - stim_onset

        # Peak velocity
        if np.isnan(peak_time):
            peak_velocity_diff = peak_velocity_gradient = peak_velocity_savgol = np.nan
        else:
            velocity_mask = (
                (pupil_df["time"] >= stim_onset) &
                (pupil_df["time"] <= peak_time)
            )
            idx = pupil_df.index[velocity_mask]
            peak_velocity_diff     = velocity_diff[idx].max()     if len(idx) > 0 else np.nan
            peak_velocity_gradient = velocity_gradient[idx].max() if len(idx) > 0 else np.nan
            peak_velocity_savgol   = velocity_savgol[idx].max()   if len(idx) > 0 else np.nan

        # Assign block index
        block_idx = None
        if not block_events.empty:
            block_before = block_events[block_events["onset"] <= stim_onset]
            if not block_before.empty:
                block_idx = block_before.shape[0]

        trial_rows.append({
            "trial_index":             trial_idx + 1,
            "stim_onset":              stim_onset,
            "blink_in_baseline":       baseline_flags[trial_idx],
            "mean_baseline_pupil_size": mean_baseline,
            "baseline_sd":             baseline_sd,
            "baseline_cv":             baseline_cv,
            "peak_pupil_amplitude":    peak_amp,
            "peak_latency_s":          peak_latency,
            "tepr":                    tepr,
            "peak_velocity_diff":      peak_velocity_diff,
            "peak_velocity_gradient":  peak_velocity_gradient,
            "peak_velocity_savgol":    peak_velocity_savgol,
            "block_index":             block_idx,
        })

    trial_df = pd.DataFrame(trial_rows)

    # Merge block metrics onto trials
    if not block_df.empty:
        trial_df = trial_df.merge(block_df, on="block_index", how="left")

    # -------------------------
    # QUALITY CHECKS
    # -------------------------
    if not trial_df.empty:
        missing_baseline = trial_df["mean_baseline_pupil_size"].isna().sum()
        missing_peak     = trial_df["peak_pupil_amplitude"].isna().sum()
        if missing_baseline > 0:
            print(f"    ⚠️ {missing_baseline} trials missing baseline window samples")
        if missing_peak > 0:
            print(f"    ⚠️ {missing_peak} trials missing peak window samples")

    if not block_df.empty and "block_index" in trial_df.columns:
        block_variability = (
            trial_df.groupby("block_index")[
                ["fix_mean_pupil_size", "fix_pupil_sd", "fix_pupil_cv"]
            ].nunique(dropna=False)
        )
        if (block_variability > 1).any().any():
            print("    ⚠️ Block metrics vary within block; check block alignment")

    print(f"  ✓ Extracted {len(trial_df)} trials across {len(block_df)} blocks")

    return trial_df

def epoch_pupil(pupil_df, raw, pupil_col, CONFIG):
    """
    Cut stimulus-locked epochs from the pupil signal.

    Returns
    -------
    epochs : np.ndarray, shape (n_trials, n_timepoints)
    times  : np.ndarray, shape (n_timepoints,)
        Time axis in seconds relative to stimulus onset.
    block_indices : list of int
        Block index for each trial, for color-coding.
    """

    event_df          = raw.annotations.to_data_frame()
    event_df["onset"] = raw.annotations.onset

    stim_events = event_df[
        event_df["description"].str.contains("Stimulus", case=False, na=False)
    ].sort_values("onset")

    block_events = event_df[
        event_df["description"].str.contains("Trial Ready", case=False, na=False)
    ].sort_values("onset")

    fs         = CONFIG["DOWNSAMPLE_HZ"]
    t_start    = CONFIG["EPOCH_WINDOW"][0]
    t_end      = CONFIG["EPOCH_WINDOW"][1]
    n_samples  = int((t_end - t_start) * fs)
    times      = np.linspace(t_start, t_end, n_samples, endpoint=False)

    epochs        = []
    block_indices = []

    for stim_onset in stim_events["onset"].values:
        epoch_start = stim_onset + t_start
        epoch_end   = stim_onset + t_end

        mask   = (pupil_df["time"] >= epoch_start) & (pupil_df["time"] < epoch_end)
        values = pupil_df.loc[mask, pupil_col].values

        if len(values) != n_samples:
            print(f"    ⚠️ Incomplete epoch around {stim_onset:.2f}s: expected {n_samples} samples, got {len(values)}")
            continue  # skip incomplete epochs at recording edges

        # --- Baseline correction ---
        baseline_mask = (times >= CONFIG["BASELINE_WINDOW"][0]) & (times < CONFIG["BASELINE_WINDOW"][1])
        baseline_mean = np.nanmean(values[baseline_mask])
        values        = values - baseline_mean

        # Assign block index
        block_idx = None
        if not block_events.empty:
            block_before = block_events[block_events["onset"] <= stim_onset]
            if not block_before.empty:
                block_idx = block_before.shape[0]

        epochs.append(values)
        block_indices.append(block_idx)

    epochs = np.array(epochs)

    print(f"  ✓ Epoched {len(epochs)} trials, {n_samples} samples each")

    return epochs, times, block_indices

def compute_blink_pct_raw(raw):
    """
    Percentage of the recording duration marked as blink events
    by the eye tracker, before any padding or interpolation.
    This is the number to report in a paper.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw object with annotations (called before preprocessing).

    Returns
    -------
    pct : float
    """
    event_df          = raw.annotations.to_data_frame()
    event_df["onset"] = raw.annotations.onset

    blinks            = event_df[event_df["description"] == "BAD_blink"]
    total_blink_s     = blinks["duration"].sum()
    total_recording_s = raw.times[-1] - raw.times[0]

    return (total_blink_s / total_recording_s) * 100


def flag_blink_in_baseline(pupil_df, raw, CONFIG):
    """
    For each trial, check whether any sample in the baseline window
    was interpolated over (i.e. differs between pupil_raw and pupil_int).

    Adds a boolean column 'blink_in_baseline' to pupil_df in place,
    and returns the percentage of trials where this is True.

    Parameters
    ----------
    pupil_df : pd.DataFrame
        Preprocessed pupil DataFrame — must contain pupil_raw and pupil_int.
    raw : mne.io.Raw
        Raw object with annotations (stimulus onsets already injected).
    CONFIG : dict

    Returns
    -------
    pct_trials_blink_in_baseline : float
    """
    event_df          = raw.annotations.to_data_frame()
    event_df["onset"] = raw.annotations.onset

    stim_events = event_df[
        event_df["description"].str.contains("Stimulus", case=False, na=False)
    ].sort_values("onset").reset_index(drop=True)

    baseline_start_offset = CONFIG["BASELINE_WINDOW"][0]
    baseline_end_offset   = CONFIG["BASELINE_WINDOW"][1]

    blink_flags = []

    for stim_onset in stim_events["onset"].values:
        mask = (
            (pupil_df["time"] >= stim_onset + baseline_start_offset) &
            (pupil_df["time"] <  stim_onset + baseline_end_offset)
        )
        window = pupil_df.loc[mask]

        has_blink = bool(
            np.any(np.abs(window["pupil_int"] - window["pupil_raw"]) > 1e-6)
        )
        blink_flags.append(has_blink)

    n_trials = len(blink_flags)
    n_blink  = sum(blink_flags)
    pct      = (n_blink / n_trials * 100) if n_trials > 0 else np.nan

    print(f"  ✓ Baseline blink check: {n_blink}/{n_trials} trials affected ({pct:.1f}%)")

    return blink_flags, pct