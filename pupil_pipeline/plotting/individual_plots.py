"""
SINGLE-RECORDING-LEVEL PLOTTING
====================

Generates visualization and quality-control plots from different stages 
of the pipeline for individual recordings.

This serves as a quality control step to ensure that preprocessing steps 
are performing as expected before proceeding to group-level analyses.

"""
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde
import pandas as pd

def plot_raw_recording(raw, subject, session = "baseline", task = "rest"):
    df_raw = raw.to_data_frame()
    df_raw.columns = [c.split('_')[0] for c in df_raw.columns]
    title = f"{subject} {session} {task} - Pupil Size Over Time"

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(df_raw['time'], df_raw['pupil'], linewidth=0.8, color='blue', alpha=0.7)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Pupil Size', fontsize=12)
    plt.grid(True, alpha=0.3)

    # Format the plot
    plt.tight_layout()
    plt.show()

def plot_pupil_df(df_raw, pupil_column, subject, session="baseline", task="rest"):
    title = f"{subject} {session} {task} - Preprocessed Pupil Size Over Time"
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(df_raw['time'], df_raw[pupil_column], linewidth=0.8, color='blue', alpha=0.7)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel(f'Pupil Size - {pupil_column}', fontsize=12)
    plt.grid(True, alpha=0.3)

    # Format the plot
    plt.tight_layout()
    plt.show()

def plot_evoked(epochs, times, block_indices, pupil_col, subject, session, task):
    """
    Plot all stimulus-evoked epochs color-coded by block with the
    average response overlaid.

    Parameters
    ----------
    epochs : np.ndarray, shape (n_trials, n_timepoints)
    times : np.ndarray
        Time axis in seconds relative to stimulus onset.
    block_indices : list of int
        Block index per trial for color-coding.
    pupil_col : str
        Name of the pupil column, used for the y-axis label.
    subject, session, task : str
        Used for the plot title.
    """

    unique_blocks = sorted(set(b for b in block_indices if b is not None))
    n_blocks      = max(unique_blocks) if unique_blocks else 1
    cmap          = cm.get_cmap("tab10", n_blocks)
    colors        = {b: cmap(b - 1) for b in unique_blocks}

    fig, ax = plt.subplots(figsize=(10, 5))

    # --- Individual epochs ---
    for epoch, block_idx in zip(epochs, block_indices):
        color = colors.get(block_idx, "grey")
        ax.plot(times, epoch, color=color, alpha=0.15, linewidth=0.8)

    # --- Block-level legend patches ---
    handles = [
        plt.Line2D([0], [0], color=colors[b], linewidth=2, label=f"Block {b}")
        for b in unique_blocks
    ]

    # --- Grand average ---
    average = np.nanmean(epochs, axis=0)
    ax.plot(times, average, color="black", linewidth=2.5, label="Average", zorder=5)
    handles.append(
        plt.Line2D([0], [0], color="black", linewidth=2.5, label="Average")
    )

    # --- Stimulus onset line ---
    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.axhline(0, color="black", linestyle="-",  linewidth=0.5, alpha=0.3)

    ax.set_xlabel("Time relative to stimulus onset (s)")
    ax.set_ylabel(pupil_col)
    ax.set_title(f"Stimulus-evoked pupil response\n{subject} | {session} | {task}")
    ax.legend(handles=handles, frameon=False, loc="upper left")

    plt.tight_layout()
    print("PLOTTING NOW")
    plt.show()

    return fig

def plot_saccade_summary(pupil_df, trial_df, beh_data, subject, session, task, CONFIG):
    """
    Plot a single-recording summary figure with 5 subplots:
        Top row:    heatmap | landing scatter | BCEA over trials
        Bottom row: saccade latency histogram | fixation duration histogram

    Parameters
    ----------
    pupil_df : pd.DataFrame
        Preprocessed pupil DataFrame containing gaze_x and gaze_y.
    trial_df : pd.DataFrame
        Output of extract_saccades — one row per trial.
    beh_data : pd.DataFrame
        Full behavioral CSV for extracting stimulus positions.
    subject, session, task : str
    CONFIG : dict
    """

    screen_w = CONFIG["SCREEN_RESOLUTION"][0]
    screen_h = CONFIG["SCREEN_RESOLUTION"][1]

    # Stimulus positions for this recording
    task_num = int(''.join(filter(str.isdigit, task)))
    sub_id   = int(subject.split("-")[1])
    ses_name = session.replace("ses-", "")

    beh_sub = beh_data[
        (beh_data["Subject"]     == sub_id)  &
        (beh_data["stimulation"] == ses_name) &
        (beh_data["task"]        == task_num)
    ].reset_index(drop=True)

    stim_x_px = (
        beh_sub["square_X"].str.replace("%", "").astype(float) / 100
        + CONFIG["STIMULUS_SIZE_PCT"][0] / 2
    ) * screen_w

    stim_y_px = (
        beh_sub["square_Y"].str.replace("%", "").astype(float) / 100
        + CONFIG["STIMULUS_SIZE_PCT"][1] / 2
    ) * screen_h

    # -------------------------
    # FIGURE LAYOUT
    # Top row: heatmap (wider) | landing scatter | BCEA over trials
    # Bottom row: saccade latency histogram | fixation duration histogram
    # -------------------------
    aspect = CONFIG["SCREEN_RESOLUTION"][0] / CONFIG["SCREEN_RESOLUTION"][1]  # 1920/1200 = 1.6

    # Base height for spatial plots, width derived from aspect ratio
    spatial_h = 4.5
    spatial_w = spatial_h * aspect

    fig = plt.figure(figsize=(spatial_w * 2.8, spatial_h * 2.2))
    fig.suptitle(
        f"Saccade & Fixation Summary\n{subject} | {session} | {task}",
        fontsize=13, y=1.01
    )

    # Two separate gridspecs — top row (2 cols) and bottom row (3 cols)
    gs_top = gridspec.GridSpec(
        1, 2,
        figure=fig,
        left=0.05, right=0.95,
        top=0.88, bottom=0.55,
        wspace=0.12
    )
    gs_bot = gridspec.GridSpec(
        1, 3,
        figure=fig,
        left=0.05, right=0.95,
        top=0.44, bottom=0.08,
        wspace=0.35
    )

    ax_heatmap = fig.add_subplot(gs_top[0, 0])
    ax_scatter = fig.add_subplot(gs_top[0, 1])
    ax_bcea    = fig.add_subplot(gs_bot[0, 0])
    ax_lat     = fig.add_subplot(gs_bot[0, 1])
    ax_fix     = fig.add_subplot(gs_bot[0, 2])

    # -------------------------
    # 1. GAZE POSITION HEATMAP
    # -------------------------
    gaze_valid = pupil_df.dropna(subset=["gaze_x", "gaze_y"])
    gx = gaze_valid["gaze_x"].values
    gy = gaze_valid["gaze_y"].values

    if len(gx) > 100:
        heatmap, xedges, yedges = np.histogram2d(
            gx, gy,
            bins=100,
            range=[[0, screen_w], [0, screen_h]]
        )
        # Smooth with gaussian for visual clarity
        from scipy.ndimage import gaussian_filter
        heatmap = gaussian_filter(heatmap.T, sigma=2)

        ax_heatmap.imshow(
            heatmap,
            origin="upper",
            extent=[0, screen_w, screen_h, 0],
            cmap="hot",
            aspect="auto",
            interpolation="bilinear"
        )

        ax_heatmap.set_aspect(1 / aspect)

    # Overlay stimulus locations
    # Count appearances per unique stimulus location
    stim_pos_df = pd.DataFrame({
        "x": stim_x_px.values,
        "y": stim_y_px.values
    })
    stim_counts = (
        stim_pos_df
        .round(1)  # round to avoid floating point mismatches
        .groupby(["x", "y"])
        .size()
        .reset_index(name="count")
    )

    ax_heatmap.scatter(
        stim_counts["x"], stim_counts["y"],
        s=80, c="cyan", marker="+", linewidths=1.5,
        label="Stimulus locations & counts", zorder=5
    )
    for _, row in stim_counts.iterrows():
        ax_heatmap.annotate(
            str(int(row["count"])),
            xy=(row["x"], row["y"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            color="cyan",
            zorder=6
        )

    ax_heatmap.set_xlim(0, screen_w)
    ax_heatmap.set_ylim(screen_h, 0)  # y flipped to match screen coords
    ax_heatmap.set_xlabel("Screen X (px)")
    ax_heatmap.set_ylabel("Screen Y (px)")
    ax_heatmap.set_title("Gaze Position Heatmap")
    ax_heatmap.legend(fontsize=6, loc="lower left")

    # -------------------------
    # 2. SACCADE LANDING SCATTER
    # -------------------------
    valid_landings = trial_df.dropna(subset=["sac_landing_x", "sac_landing_y"])

    ax_scatter.scatter(
        valid_landings["sac_landing_x"],
        valid_landings["sac_landing_y"],
        s=20, alpha=0.5, color="steelblue", label="Landing positions"
    )

    # Overlay stimulus locations
    ax_scatter.scatter(
        stim_x_px, stim_y_px,
        s=120, c="red", marker="+", linewidths=2,
        label="Stimulus locations", zorder=5
    )

    # Draw spatial criterion circle around each stimulus location
    spatial_criterion_px = (
        CONFIG["SACCADE_SPATIAL_CRITERION_DEG"] *
        np.tan(np.radians(1)) *
        CONFIG["VIEWING_DISTANCE_CM"] /
        CONFIG["SCREEN_WIDTH_CM"] *
        screen_w
    )
    unique_stim = pd.DataFrame({
        "x": stim_x_px, "y": stim_y_px
    }).drop_duplicates()

    for _, row in unique_stim.iterrows():
        circle = plt.Circle(
            (row["x"], row["y"]),
            spatial_criterion_px,
            color="red", fill=False, linewidth=0.8, alpha=0.4
        )
        ax_scatter.add_patch(circle)

    ax_scatter.set_xlim(0, screen_w)
    ax_scatter.set_ylim(screen_h, 0)
    ax_scatter.set_xlabel("Screen X (px)")
    ax_scatter.set_ylabel("Screen Y (px)")
    ax_scatter.set_title("Saccade Landing Positions")
    ax_scatter.legend(fontsize=7, loc="lower left")

    # -------------------------
    # 3. BCEA OVER TRIALS
    # -------------------------
    bcea_valid = trial_df.dropna(subset=["bcea_vs_centroid", "bcea_vs_target"])

    if not bcea_valid.empty:
        ax_bcea.plot(
            bcea_valid["trial_index"],
            bcea_valid["bcea_vs_centroid"],
            color="steelblue", linewidth=1, alpha=0.8,
            label="vs centroid"
        )
        ax_bcea.plot(
            bcea_valid["trial_index"],
            bcea_valid["bcea_vs_target"],
            color="tomato", linewidth=1, alpha=0.8,
            label="vs target"
        )
        # Rolling average to show trend
        window = min(13, len(bcea_valid))
        ax_bcea.plot(
            bcea_valid["trial_index"],
            bcea_valid["bcea_vs_centroid"].rolling(window, min_periods=1).mean(),
            color="steelblue", linewidth=2.5
        )
        ax_bcea.plot(
            bcea_valid["trial_index"],
            bcea_valid["bcea_vs_target"].rolling(window, min_periods=1).mean(),
            color="tomato", linewidth=2.5
        )

    ax_bcea.set_xlabel("Trial")
    ax_bcea.set_ylabel("BCEA (px²)")
    ax_bcea.set_title("Fixation Stability over Trials")
    ax_bcea.legend(fontsize=7)

    # -------------------------
    # 4. SACCADE LATENCY HISTOGRAM
    # -------------------------
    lat_valid = trial_df["sac_latency_s"].dropna() * 1000  # convert to ms

    if not lat_valid.empty:
        ax_lat.hist(
            lat_valid,
            bins=30,
            color="steelblue",
            edgecolor="white",
            linewidth=0.5
        )
        ax_lat.axvline(
            lat_valid.median(),
            color="tomato", linewidth=1.5,
            linestyle="--", label=f"Median: {lat_valid.median():.0f} ms"
        )

    ax_lat.set_xlabel("Saccade Latency (ms)")
    ax_lat.set_ylabel("Trial count")
    ax_lat.set_title("Saccade Latency Distribution")
    ax_lat.legend(fontsize=7)

    # -------------------------
    # 5. FIXATION DURATION HISTOGRAM
    # -------------------------
    fix_valid = trial_df["fix_duration_s"].dropna() * 1000  # convert to ms

    if not fix_valid.empty:
        ax_fix.hist(
            fix_valid,
            bins=30,
            color="mediumseagreen",
            edgecolor="white",
            linewidth=0.5
        )
        ax_fix.axvline(
            fix_valid.median(),
            color="tomato", linewidth=1.5,
            linestyle="--", label=f"Median: {fix_valid.median():.0f} ms"
        )

    ax_fix.set_xlabel("Fixation Duration (ms)")
    ax_fix.set_ylabel("Trial count")
    ax_fix.set_title("Fixation Duration Distribution")
    ax_fix.legend(fontsize=7)

    plt.tight_layout()
    print("PLOTTING SACCADES NOW")
    plt.show()

    return fig

def plot_signal_comparison(pupil_df, columns, labels=None, title=None,
                            time_range=None, show_diff=False):
    """
    Plot the pupil signal at multiple preprocessing stages for direct comparison.

    Parameters
    ----------
    pupil_df : pd.DataFrame
        Preprocessed pupil DataFrame containing 'time' column.
    columns : list of str
        Column names to plot, e.g. ['pupil_int', 'pupil_int_despiked'].
    labels : list of str, optional
        Display labels for each column. Defaults to column names.
    title : str, optional
        Plot title.
    time_range : tuple of (float, float), optional
        (start_s, end_s) to zoom into a specific time window.
        If None, plots the full recording.
    show_diff : bool
        If True and exactly two columns are provided, adds a third panel
        showing the difference (what was removed/changed).
    """

    if labels is None:
        labels = columns

    # Filter time range
    df = pupil_df.copy()
    if time_range is not None:
        df = df[(df["time"] >= time_range[0]) & (df["time"] <= time_range[1])]

    time = df["time"].values

    # Determine number of panels
    n_panels = len(columns) + (1 if show_diff and len(columns) == 2 else 0)

    colors = [
        "#2196F3",  # blue
        "#FF5722",  # orange-red
        "#4CAF50",  # green
        "#9C27B0",  # purple
        "#FF9800",  # amber
    ]

    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(16, 3.5 * n_panels),
        sharex=True
    )
    if n_panels == 1:
        axes = [axes]

    # -------------------------
    # SIGNAL PANELS
    # -------------------------
    for idx, (col, label) in enumerate(zip(columns, labels)):
        ax = axes[idx]

        if col not in df.columns:
            ax.text(0.5, 0.5, f"Column '{col}' not found",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_ylabel(label)
            continue

        values = df[col].values
        color  = colors[idx % len(colors)]

        # Plot signal
        ax.plot(time, values, color=color, linewidth=0.8, alpha=0.9, label=label)

        # Shade NaN regions in light grey so gaps are visible
        nan_mask = np.isnan(values)
        if nan_mask.any():
            _shade_nan_regions(ax, time, nan_mask)

        # Annotate basic stats
        valid_vals = values[~nan_mask]
        if len(valid_vals) > 0:
            stats_text = (
                f"valid: {100*(1-nan_mask.mean()):.1f}%  |  "
                f"mean: {valid_vals.mean():.2f}  |  "
                f"sd: {valid_vals.std():.2f}"
            )
            ax.text(
                0.01, 0.97, stats_text,
                transform=ax.transAxes,
                fontsize=8, va="top", ha="left",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2)
            )

        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # -------------------------
    # DIFFERENCE PANEL
    # -------------------------
    if show_diff and len(columns) == 2:
        ax   = axes[-1]
        col1 = df[columns[0]].values
        col2 = df[columns[1]].values
        diff = col1 - col2

        # Samples that were set to NaN (removed by despiking)
        newly_nan = ~np.isnan(col1) & np.isnan(col2)
        # Samples that were changed but not set to NaN (interpolated)
        changed   = ~np.isnan(col1) & ~np.isnan(col2) & (np.abs(diff) > 1e-6)

        ax.plot(time, col1, color=colors[0], linewidth=0.6,
                alpha=0.4, label=labels[0])
        ax.plot(time, col2, color=colors[1], linewidth=0.6,
                alpha=0.4, label=labels[1])

        # Highlight removed samples in red
        if newly_nan.any():
            ax.scatter(
                time[newly_nan], col1[newly_nan],
                color="red", s=4, alpha=0.6,
                label=f"Removed ({newly_nan.sum()} samples)"
            )

        # Highlight changed samples in orange
        if changed.any():
            ax.scatter(
                time[changed], col1[changed],
                color="orange", s=4, alpha=0.4,
                label=f"Changed ({changed.sum()} samples)"
            )

        ax.set_ylabel("Difference", fontsize=9)
        ax.legend(fontsize=8, loc="upper right", frameon=False)
        ax.grid(True, alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Time (s)", fontsize=10)

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)

    plt.tight_layout()
    plt.show()
    return fig


def _shade_nan_regions(ax, time, nan_mask):
    """Shade NaN regions in light grey."""
    in_nan    = False
    nan_start = None

    for i, is_nan in enumerate(nan_mask):
        if is_nan and not in_nan:
            nan_start = time[i]
            in_nan    = True
        elif not is_nan and in_nan:
            ax.axvspan(nan_start, time[i], color="lightgrey", alpha=0.5, linewidth=0)
            in_nan = False

    if in_nan:
        ax.axvspan(nan_start, time[-1], color="lightgrey", alpha=0.5, linewidth=0)
# ADD FUNCTIONS FOR INDIVIDUAL INTERACTIVE PLOTS WHERE ALL 
# SELECTED PREPROCESSING STEPS ARE VISUALIZED
