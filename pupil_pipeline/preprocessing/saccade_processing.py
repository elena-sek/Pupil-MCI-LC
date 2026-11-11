"""
SACCADE AND FIXATION PROCESSING
================================

Extracts trial-level saccade and fixation metrics from preprocessed gaze data.

Responsibilities:
-----------------
- Identify target-directed saccades per trial
- Extract saccade latency and peak velocity
- Compute saccade landing position variability (across trials)
- Extract fixation duration
- Compute fixation stability (BCEA) relative to target and to fixation centroid

Design principles:
------------------
- No preprocessing occurs here.
- No subject/session/task identifiers are added here — that is save_results' responsibility.
- All window and criterion parameters come from CONFIG.
"""

import numpy as np
import pandas as pd


# =============================================================================
# PUBLIC FUNCTION
# =============================================================================

def extract_saccades(pupil_df, raw, beh_data, subject, session, task, CONFIG):
    """
    Extract trial-level saccade and fixation metrics.

    Parameters
    ----------
    pupil_df : pd.DataFrame
        Preprocessed pupil DataFrame containing time, gaze_x, gaze_y.
    raw : mne.io.Raw
        Raw object with annotations.
    beh_data : pd.DataFrame
        Full behavioral CSV (used to get stimulus positions).
    subject, session, task : str
        Used to filter behavioral data for this recording.
    CONFIG : dict

    Returns
    -------
    trial_df : pd.DataFrame
        One row per trial with saccade and fixation metrics.
    """

    # -------------------------
    # SETUP
    # -------------------------
    px_per_deg = (
        np.tan(np.radians(1))
        * CONFIG["VIEWING_DISTANCE_CM"]
        / CONFIG["SCREEN_WIDTH_CM"]
        * CONFIG["SCREEN_RESOLUTION"][0]
    )
    spatial_criterion_px = CONFIG["SACCADE_SPATIAL_CRITERION_DEG"] * px_per_deg

    t_min, t_max = CONFIG["SACCADE_TEMPORAL_WINDOW"]

    # -------------------------
    # EXTRACT EVENTS
    # -------------------------
    event_df          = raw.annotations.to_data_frame()
    event_df["onset"] = raw.annotations.onset

    stim_events = event_df[
        event_df["description"].str.contains("Stimulus", case=False, na=False)
    ].sort_values("onset").reset_index(drop=True)

    sac_events = event_df[
        event_df["description"].str.contains("saccade", case=False, na=False)
    ].sort_values("onset")

    fix_events = event_df[
        event_df["description"].str.contains("fixation", case=False, na=False)
    ].sort_values("onset")

    # -------------------------
    # STIMULUS POSITIONS
    # -------------------------
    task_num = int(''.join(filter(str.isdigit, task)))
    sub_id   = int(subject.split("-")[1])
    ses_name = session.replace("ses-", "")

    beh_sub = beh_data[
        (beh_data["Subject"]     == sub_id)  &
        (beh_data["stimulation"] == ses_name) &
        (beh_data["task"]        == task_num)
    ].reset_index(drop=True)

    # Convert stimulus positions from percentage to pixels (top-left corner → centre)
    stim_x_px = (
        beh_sub["square_X"].str.replace("%", "").astype(float) / 100
        + CONFIG["STIMULUS_SIZE_PCT"][0] / 2
    ) * CONFIG["SCREEN_RESOLUTION"][0]

    stim_y_px = (
        beh_sub["square_Y"].str.replace("%", "").astype(float) / 100
        + CONFIG["STIMULUS_SIZE_PCT"][1] / 2
    ) * CONFIG["SCREEN_RESOLUTION"][1]

    # -------------------------
    # TRIAL-LEVEL METRICS
    # -------------------------
    trial_rows = []

    for trial_idx, stim_onset in enumerate(stim_events["onset"].values):

        target_x = stim_x_px.iloc[trial_idx] if trial_idx < len(stim_x_px) else np.nan
        target_y = stim_y_px.iloc[trial_idx] if trial_idx < len(stim_y_px) else np.nan

        # --- Find candidate saccades within temporal window ---
        sac_window = sac_events[
            (sac_events["onset"] >= stim_onset + t_min) &
            (sac_events["onset"] <= stim_onset + t_max)
        ]

        sac_latency      = np.nan
        sac_peak_vel     = np.nan
        sac_landing_x    = np.nan
        sac_landing_y    = np.nan
        sac_dist_to_tgt  = np.nan
        target_sac_found = False

        for _, sac in sac_window.iterrows():
            sac_offset_time = sac["onset"] + sac["duration"]

            # Get gaze position at saccade offset as landing position
            offset_mask = (
                (pupil_df["time"] >= sac_offset_time - 0.01) &
                (pupil_df["time"] <= sac_offset_time + 0.01)
            )
            offset_samples = pupil_df.loc[offset_mask]

            if offset_samples.empty:
                continue

            land_x = offset_samples["gaze_x"].median()
            land_y = offset_samples["gaze_y"].median()

            dist = np.sqrt((land_x - target_x) ** 2 + (land_y - target_y) ** 2)

            if dist <= spatial_criterion_px:
                sac_latency      = sac["onset"] - stim_onset
                sac_landing_x    = land_x
                sac_landing_y    = land_y
                sac_dist_to_tgt  = dist
                target_sac_found = True
                break  # take first qualifying saccade

        # --- Fixation immediately after saccade offset ---
        fix_duration         = np.nan
        bcea_vs_target       = np.nan
        bcea_vs_centroid     = np.nan

        if target_sac_found:
            sac_end_time = sac["onset"] + sac["duration"]

            # First fixation starting after saccade offset
            fix_after = fix_events[fix_events["onset"] >= sac_end_time]

            if not fix_after.empty:
                fix       = fix_after.iloc[0]
                fix_start = fix["onset"]
                fix_end   = fix["onset"] + fix["duration"]
                fix_duration = fix["duration"]

                # Gaze samples during fixation
                fix_mask = (
                    (pupil_df["time"] >= fix_start) &
                    (pupil_df["time"] <= fix_end)
                )
                fix_samples = pupil_df.loc[fix_mask].dropna(subset=["gaze_x", "gaze_y"])

                if len(fix_samples) >= 3:
                    gx = fix_samples["gaze_x"].values
                    gy = fix_samples["gaze_y"].values

                    # BCEA vs fixation centroid
                    bcea_vs_centroid = _compute_bcea(gx, gy,
                                                     ref_x=gx.mean(),
                                                     ref_y=gy.mean())

                    # BCEA vs target location
                    bcea_vs_target = _compute_bcea(gx, gy,
                                                   ref_x=target_x,
                                                   ref_y=target_y)

        trial_rows.append({
            "trial_index":        trial_idx + 1,
            "stim_onset":         stim_onset,
            "target_x_px":        target_x,
            "target_y_px":        target_y,
            "target_sac_found":   target_sac_found,
            "sac_latency_s":      sac_latency,
            "sac_landing_x":      sac_landing_x,
            "sac_landing_y":      sac_landing_y,
            "sac_dist_to_tgt_px": sac_dist_to_tgt,
            "fix_duration_s":     fix_duration,
            "bcea_vs_centroid":   bcea_vs_centroid,
            "bcea_vs_target":     bcea_vs_target,
        })

    trial_df = pd.DataFrame(trial_rows)

    # -------------------------
    # LANDING POSITION VARIABILITY
    # Collapsed across all trials regardless of target location
    # -------------------------
    valid_landings = trial_df.dropna(subset=["sac_landing_x", "sac_landing_y"])

    if len(valid_landings) >= 3:
        landing_var_x  = valid_landings["sac_landing_x"].std(ddof=1)
        landing_var_y  = valid_landings["sac_landing_y"].std(ddof=1)
        landing_bcea   = _compute_bcea(
            valid_landings["sac_landing_x"].values,
            valid_landings["sac_landing_y"].values,
            ref_x=valid_landings["sac_landing_x"].mean(),
            ref_y=valid_landings["sac_landing_y"].mean()
        )
    else:
        landing_var_x = landing_var_y = landing_bcea = np.nan

    # Add as recording-level columns onto every trial row
    trial_df["landing_var_x_px"]  = landing_var_x
    trial_df["landing_var_y_px"]  = landing_var_y
    trial_df["landing_bcea"]      = landing_bcea

    # -------------------------
    # QUALITY CHECKS
    # -------------------------
    n_found   = trial_df["target_sac_found"].sum()
    n_missing = (~trial_df["target_sac_found"]).sum()
    print(f"  ✓ Target saccade found on {n_found}/{len(trial_df)} trials")
    if n_missing > 0:
        print(f"  ⚠️  No qualifying saccade found on {n_missing} trials")

    # DELETE
    # How many saccades are detected overall
    # event_df = raw.annotations.to_data_frame()
    # event_df["onset"] = raw.annotations.onset
    # sac_events = event_df[event_df["description"].str.contains("saccade", case=False)]
    # print(f"Total saccades in recording: {len(sac_events)}")
    # print(f"Total trials: {len(trial_df)}")
    # print(f"Trials with saccade found: {trial_df['target_sac_found'].sum()}")

    # # For a few missed trials, show what saccades were nearby
    # missed = trial_df[~trial_df["target_sac_found"]].head(5)
    # for _, trial in missed.iterrows():
    #     stim_onset = trial["stim_onset"]
    #     nearby = sac_events[
    #         (sac_events["onset"] >= stim_onset - 0.5) &
    #         (sac_events["onset"] <= stim_onset + 2.0)
    #     ]
    #     print(f"\nTrial {int(trial['trial_index'])} (stim onset {stim_onset:.2f}s):")
    #     print(nearby[["onset", "duration"]].to_string())
    #     print(f"  Target: ({trial['target_x_px']:.0f}, {trial['target_y_px']:.0f})")
    

    # # Check Trial 2 - saccade at onset 13.781, duration 0.055
    # sac_offset = 13.781 + 0.055  # = 13.836

    # window = pupil_df[
    #     (pupil_df["time"] >= sac_offset - 0.05) &
    #     (pupil_df["time"] <= sac_offset + 0.05)
    # ]
    # print(window[["time", "gaze_x", "gaze_y"]])
    # print(f"\nTarget: (1536, 942)")
    # print(f"Landing estimate: ({window['gaze_x'].median():.0f}, {window['gaze_y'].median():.0f})")
    # print(f"Distance to target: {((window['gaze_x'].median()-1536)**2 + (window['gaze_y'].median()-942)**2)**0.5:.0f} px")
    return trial_df


# =============================================================================
# PRIVATE HELPERS
# =============================================================================

def _compute_bcea(gx, gy, ref_x, ref_y, p=0.68):
    """
    Bivariate Contour Ellipse Area (BCEA).

    Computes the area of the ellipse containing proportion p of gaze samples
    relative to a reference point (either centroid or target location).

    Parameters
    ----------
    gx, gy : np.ndarray
        Gaze x and y coordinates in pixels.
    ref_x, ref_y : float
        Reference point to compute deviations from.
    p : float
        Proportion of samples to include (default 0.68, i.e. 1 SD ellipse).

    Returns
    -------
    bcea : float
        Area in pixels squared.
    """
    k       = np.log(1 / (1 - p))
    std_x   = np.std(gx - ref_x, ddof=1)
    std_y   = np.std(gy - ref_y, ddof=1)
    rho     = np.corrcoef(gx - ref_x, gy - ref_y)[0, 1]
    bcea    = 2 * np.pi * k * std_x * std_y * np.sqrt(1 - rho ** 2)
    return bcea