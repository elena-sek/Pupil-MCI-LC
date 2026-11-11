"""
PUPIL PREPROCESSING UTILITIES
=============================

Low-level helper functions used during global pupil preprocessing.

This module contains algorithmic implementations required by
`pupil_preprocess.py`.

Typical functionality:
----------------------
- Detection of invalid samples
- Blink detection
- Blink interpolation
- Temporal filtering
- Resampling / downsampling
- Signal normalization

Design philosophy:
------------------
These functions implement HOW preprocessing is performed,
while `pupil_preprocess.py` defines WHAT preprocessing steps
are applied and in which order.

Design principles:
------------------
- All functions operate on DataFrames in place.
- No pipeline logic or CONFIG handling here.
- Functions are reusable and independent of each other.
"""
import numpy as np
import scipy as sp
import pandas as pd


# =============================================================================
# PRIVATE FILTER HELPERS
# =============================================================================

def _butter_lowpass(highcut, fs, order=3):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = sp.signal.butter(order, [high], btype='lowpass')
    return b, a


def _butter_lowpass_filter(data, highcut, fs, order=3):
    b, a = _butter_lowpass(highcut, fs, order=order)
    return sp.signal.filtfilt(b, a, data)


def _butter_highpass(lowcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = sp.signal.butter(order, [low], btype='highpass')
    return b, a


def _butter_highpass_filter(data, lowcut, fs, order=3):
    b, a = _butter_highpass(lowcut, fs, order=order)
    return sp.signal.filtfilt(b, a, data)


def _double_gamma(params, x):
    a1  = params['a1']
    sh1 = params['sh1']
    sc1 = params['sc1']
    a2  = params['a2']
    sh2 = params['sh2']
    sc2 = params['sc2']
    return (a1 * sp.stats.gamma.pdf(x, sh1, loc=0.0, scale=sc1) +
            a2 * sp.stats.gamma.pdf(x, sh2, loc=0.0, scale=sc2))


# =============================================================================
# PUBLIC PREPROCESSING FUNCTIONS
# =============================================================================

def interpolate_blinks(raw, buffer_s):
    """
    Interpolate blinks in the MNE raw object in place.

    Parameters
    ----------
    raw : mne.io.Raw
    buffer_s : float
        Buffer around each blink in seconds (e.g. 0.1).
    """
    import mne
    try:
        mne.preprocessing.eyetracking.interpolate_blinks(
            raw, buffer=(buffer_s, buffer_s), interpolate_gaze=True
        )
    except ValueError as e:
        if "array of sample points is empty" in str(e):
            print(f"  ⚠ Warning: No valid samples to interpolate from (all data marked as blinks)")
            print(f"  → Skipping blink interpolation for this recording")
        else:
            raise


def temporal_filter(df, measure, fs, lp, hp, order=3):
    """
    Apply lowpass and highpass Butterworth filters.

    Produces two new columns:
        {measure}_lp  ... lowpass filtered
        {measure}_bp  ... bandpass (highpass of lowpass)

    Both are created in a single call since _bp is derived from _lp.
    """
    clean_signal = (
        df[measure]
        .interpolate(limit_direction='both')
        .ffill()
        .bfill()
    )
    df[f'{measure}_lp'] = _butter_lowpass_filter(clean_signal, highcut=lp, fs=fs, order=order)
    # df[f'{measure}_bp'] = _butter_highpass_filter(clean_signal, lowcut=hp, fs=fs, order=order) - (clean_signal - df[f'{measure}_lp'])
    if hp:
        df[f'{measure}_bp'] = _butter_highpass_filter(df[f'{measure}_lp'], lowcut=hp, fs=fs, order=order)


def regress_blinks(df, events, current_pupil, fs, interval=7, regress_blinks = True, regress_sacs=False):
    """
    Regress out blink and saccade artifacts using double-gamma kernels
    (Knapen et al., 2016).

    Operates on `current_pupil` column and produces:
        {current_pupil}_clean

    Parameters
    ----------
    df : pd.DataFrame
    events : pd.DataFrame
        Annotations DataFrame from raw.annotations.to_data_frame().
    current_pupil : str
        Column name to regress (should be the bandpass filtered signal).
    fs : float
        Sampling frequency.
    interval : int
        Kernel length in seconds.
    """
    early_cutoff = 25
    late_cutoff  = interval

    x = np.linspace(0, interval, int(interval * fs), endpoint=False)

    standard_blink_parameters = {
        'a1': -0.604, 'sh1': 8.337,  'sc1': 0.115,
        'a2':  0.419, 'sh2': 15.433, 'sc2': 0.178
    }
    standard_sac_parameters = {
        'a1': -0.175, 'sh1': 6.451, 'sc1': 0.178,
        'a2':  0.0,   'sh2': 1,     'sc2': 1
    }

    blink_kernel = _double_gamma(standard_blink_parameters, x)
    sac_kernel   = _double_gamma(standard_sac_parameters,   x)

    # --- Blink regressor ---
    blink_ends = (
        events.loc[events['description'] == 'BAD_blink', 'onset'].values +
        events.loc[events['description'] == 'BAD_blink', 'duration'].values
    )

    blink_ends = blink_ends[
        (blink_ends > early_cutoff) &
        (blink_ends < (df['time'].iloc[-1] - late_cutoff))
    ]
    blink_ends = np.array([0], dtype=int) if blink_ends.size == 0 else blink_ends.astype(int)
    blink_ends_ind = np.array(df['time'].searchsorted(blink_ends).ravel())
    blink_reg      = np.zeros(df.shape[0])
    blink_reg[blink_ends_ind] = 1
    blink_reg_conv = sp.signal.fftconvolve(blink_reg, blink_kernel, 'full')[:-(len(blink_kernel) - 1)]

    # --- Saccade regressor ---
    sac_ends = (
        events.loc[events['description'] == 'saccade', 'onset'].values +
        events.loc[events['description'] == 'saccade', 'duration'].values
    )
    sac_ends = sac_ends[
        (sac_ends > early_cutoff) &
        (sac_ends < (df['time'].iloc[-1] - late_cutoff))
    ]
    sac_ends = np.array([0], dtype=int) if sac_ends.size == 0 else sac_ends.astype(int)
    sac_ends_ind = np.array(df['time'].searchsorted(sac_ends).ravel())
    sac_reg      = np.zeros(df.shape[0])
    sac_reg[sac_ends_ind] = 1
    sac_reg_conv = sp.signal.fftconvolve(sac_reg, sac_kernel, 'full')[:-(len(sac_kernel) - 1)]

    # combine regressors:
    regs = []
    regs_titles = []
    if regress_blinks:
        regs.append(blink_reg_conv)
        regs_titles.append('blink')
    if regress_sacs:
        regs.append(sac_reg_conv)
        regs_titles.append('saccade')
    print([r.shape for r in regs])

    # --- GLM ---
    regs = [blink_reg_conv, sac_reg_conv]
    design_matrix = np.matrix(np.vstack(regs)).T
    betas = np.array(
        ((design_matrix.T * design_matrix).I * design_matrix.T) *
        np.matrix(df[current_pupil].values).T
    ).ravel()
    explained = np.sum(
        np.vstack([betas[i] * regs[i] for i in range(len(betas))]), axis=0
    )
    rsq = sp.stats.pearsonr(df[current_pupil].values, explained)[0] ** 2
    print(f'  Blink regression explained variance: {round(rsq * 100, 2)}%')

    df[f'{current_pupil}_clean'] = df[current_pupil] - explained


def median_psc(df, measure):
    """
    Percent signal change relative to the median.

    Produces:
        {measure}_medianpsc
    """
    df[f'{measure}_medianpsc'] = (
        (df[measure] - df[measure].median()) / df[measure].median() * 100
    )

def despike_pupil(df, measure, fs, n=3, gap_threshold_ms=75, edge_buffer_ms=50,
                  sparsity_gap_ms=40, sparsity_min_ms=50, trend_window_ms=2000,
                  use_trendline_filter=True):
    """
    Four-step spike detection and removal based on Kret & Sjak-Shie (2019).

    Produces:
        {measure}_despiked

    Steps:
        1. Dilation speed filter — rejects physiologically impossible changes
        2. Edge artifact rejection — rejects samples near blink/gap edges
        3. Trend-line deviation filter — two-pass rejection of cluster artifacts
        4. Sparsity filter — rejects temporally isolated sample groups

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'time' and {measure} columns.
    measure : str
        Column to despike (should be pupil_int).
    fs : float
        Sampling frequency in Hz.
    n : float
        MAD multiplier for threshold (CONFIG["SPIKE_MAD_THRESHOLD"]).
    gap_threshold_ms : float
        Minimum gap size in ms to be considered a gap for edge rejection.
    edge_buffer_ms : float
        Samples within this many ms of a gap are rejected.
    sparsity_gap_ms : float
        Gaps larger than this split the signal for sparsity filtering.
    sparsity_min_ms : float
        Sections shorter than this after splitting are rejected.
    """

    signal = df[measure].copy().values
    time   = df["time"].values
    valid  = ~np.isnan(signal)  # boolean mask of currently valid samples

    # -------------------------
    # NORMALISE TO UNIT SCALE
    # The algorithm was designed for mm-scale signals (~2-8mm).
    # Normalising ensures MAD thresholds are scale-independent.
    # -------------------------
    valid_signal = signal[valid]
    sig_median   = np.median(valid_signal)
    sig_mad      = np.median(np.abs(valid_signal - sig_median))

    # Avoid division by zero
    if sig_mad < 1e-10:
        sig_mad = valid_signal.std()

    signal_norm = (signal - sig_median) / sig_mad

    # -------------------------
    # STEP 1: DILATION SPEED FILTER
    # -------------------------
    dilation_speed = np.full(len(signal), np.nan)

    for i in range(1, len(signal) - 1):
        if not valid[i]:
            continue
        speeds = []
        if valid[i - 1] and (time[i] - time[i - 1]) > 0:
            speeds.append(abs(signal_norm[i] - signal_norm[i - 1]) / (time[i] - time[i - 1]))
        if valid[i + 1] and (time[i + 1] - time[i]) > 0:
            speeds.append(abs(signal_norm[i + 1] - signal_norm[i]) / (time[i + 1] - time[i]))
        if speeds:
            dilation_speed[i] = max(speeds)

    valid_speeds = dilation_speed[~np.isnan(dilation_speed)]
    if len(valid_speeds) > 0:
        med_speed = np.median(valid_speeds)
        mad_speed = np.median(np.abs(valid_speeds - med_speed))
        threshold = med_speed + n * mad_speed
        speed_outliers = dilation_speed > threshold
        valid[speed_outliers] = False

    # -------------------------
    # STEP 2: EDGE ARTIFACT REJECTION
    # -------------------------
    gap_threshold_s  = gap_threshold_ms  / 1000
    edge_buffer_s    = edge_buffer_ms    / 1000

    # Find gaps in valid data
    signal_masked = signal.copy()
    signal_masked[~valid] = np.nan

    is_nan = np.isnan(signal_masked)
    gap_starts = np.where(np.diff(is_nan.astype(int)) == 1)[0] + 1
    gap_ends   = np.where(np.diff(is_nan.astype(int)) == -1)[0] + 1

    for gs, ge in zip(gap_starts, gap_ends):
        gap_duration = time[ge] - time[gs]
        if gap_duration >= gap_threshold_s:
            # Reject samples within edge_buffer_s before gap start
            pre_mask = (time >= time[gs] - edge_buffer_s) & (time < time[gs])
            valid[pre_mask] = False
            # Reject samples within edge_buffer_s after gap end
            post_mask = (time > time[ge]) & (time <= time[ge] + edge_buffer_s)
            valid[post_mask] = False

    # -------------------------
    # STEP 3: TREND-LINE DEVIATION FILTER (two passes)
    # NOTE: Disabled by default for task recordings where genuine
    # stimulus-evoked responses may be misidentified as artifacts.
    # Only enable for resting-state or baseline-only recordings.
    # -------------------------
    def _trendline_pass(signal_norm, valid, time, n, fs, trend_window_ms):
            trend = pd.Series(signal_norm.copy())
            trend[~valid] = np.nan
            trend = trend.interpolate(method="linear", limit_direction="both")
            trend = trend.rolling(
                window=max(3, int(trend_window_ms / 1000 * fs)),
                center=True,
                min_periods=1
            ).mean().values

            deviations = np.abs(signal_norm - trend)
            valid_devs = deviations[valid & ~np.isnan(deviations)]

            if len(valid_devs) == 0:
                return valid

            med_dev   = np.median(valid_devs)
            mad_dev   = np.median(np.abs(valid_devs - med_dev))
            threshold = med_dev + n * mad_dev

            new_valid = valid.copy()
            new_valid[deviations > threshold] = False
            return new_valid
    if use_trendline_filter:
        # Pass 1 — build trend from post-step-2 valid samples
        valid = _trendline_pass(signal_norm, valid, time, n, fs, trend_window_ms)

        # Pass 2 — rebuild trend and re-evaluate ALL original samples
        # (reintroduces samples that were wrongly rejected in pass 1
        # because invalid samples pulled the trend line away)
        valid_after_speed_edge = ~np.isnan(signal) & ~speed_outliers
        # reset edge rejections temporarily for pass 2 re-evaluation
        valid_pass2 = valid_after_speed_edge.copy()
        # keep step 1 and 2 rejections but re-evaluate trend deviations
        valid = _trendline_pass(signal_norm, valid_pass2, time, n, fs, trend_window_ms)

    # -------------------------
    # STEP 4: SPARSITY FILTER
    # -------------------------
    sparsity_gap_s = sparsity_gap_ms / 1000
    sparsity_min_s = sparsity_min_ms / 1000

    # Find split points — gaps larger than sparsity_gap_s
    signal_masked      = signal.copy()
    signal_masked[~valid] = np.nan
    is_nan             = np.isnan(signal_masked)
    gap_starts_sp      = np.where(np.diff(is_nan.astype(int)) == 1)[0] + 1
    gap_ends_sp        = np.where(np.diff(is_nan.astype(int)) == -1)[0] + 1

    # Build list of valid sections between large gaps
    split_points = [0]
    for gs, ge in zip(gap_starts_sp, gap_ends_sp):
        if (time[ge] - time[gs]) >= sparsity_gap_s:
            split_points.append(gs)
            split_points.append(ge)
    split_points.append(len(signal))

    # Reject sections shorter than sparsity_min_s
    for k in range(0, len(split_points) - 1, 2):
        sec_start = split_points[k]
        sec_end   = split_points[k + 1]
        sec_time  = time[sec_start:sec_end]
        sec_valid = valid[sec_start:sec_end]
        valid_times = sec_time[sec_valid]
        if len(valid_times) > 0:
            sec_duration = valid_times[-1] - valid_times[0]
            if sec_duration < sparsity_min_s:
                valid[sec_start:sec_end] = False

    # -------------------------
    # WRITE OUTPUT COLUMN
    # -------------------------
    result = signal.copy()
    result[~valid] = np.nan

    # Linear interpolation over rejected samples
    result_series = pd.Series(result)
    result_series = result_series.interpolate(
        method="linear", limit_direction="both"
    )

    # Count only samples that were valid before but rejected by despiker
    n_rejected = int(np.sum(~np.isnan(signal) & ~valid))
    pct        = n_rejected / max(np.sum(~np.isnan(signal)), 1) * 100
    print(f"  ✓ Despiking removed {n_rejected} samples ({pct:.1f}%)")

    df[f"{measure}_despiked"] = result_series.values