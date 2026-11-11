"""
GROUP-LEVEL PLOTS
=================

Generates group-level visualisations across all subjects.

Responsibilities:
-----------------
- Group-average stimulus-evoked pupil response per session
- Metric summary violin plots with individual subject points

Design principles:
------------------
- Reads from saved output files, not from pipeline objects.
- No processing occurs here.
- All plots saved to CONFIG["OUTPUT_ROOT"].
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


# =============================================================================
# PUBLIC FUNCTION
# =============================================================================

def plot_group_results(CONFIG):
    """
    Orchestrate all group-level plots.
    Reads group_trial_metrics.csv and group_epoch_averages.csv
    from CONFIG["OUTPUT_ROOT"].
    """

    output_root = Path(CONFIG["OUTPUT_ROOT"])
    metrics_path = output_root / "group_trial_metrics.csv"
    epochs_path  = output_root / "group_epoch_averages.csv"

    # Load group mapping
    group_file = Path(CONFIG["GROUP_FILE"])
    if not group_file.exists():
        print(f"  ⚠️ Group file not found: {group_file}")
        return

    task_df = pd.read_csv(group_file)
    if not {"sub_id", "group"}.issubset(task_df.columns):
        print("  ⚠️ GROUP_FILE must contain 'sub_id' and 'group' columns")
        return

    subject_group_map = (
        task_df[["sub_id", "group"]]
        .drop_duplicates()
        .set_index("sub_id")["group"]
        .to_dict()
    )

    # -------------------------
    # EVOKED RESPONSE PLOTS
    # -------------------------
    if epochs_path.exists():
        epochs_df = pd.read_csv(epochs_path)
        epochs_df = _add_group(epochs_df, subject_group_map)

        for session in ["ses-placebo", "ses-frontal", "ses-frontoparietal"]:
            ses_df = epochs_df[epochs_df["session"] == session]
            if ses_df.empty:
                print(f"  ⚠️ No epoch data for {session}, skipping")
                continue
            fig = _plot_evoked_response(ses_df, session, CONFIG)
            ses_name = session.replace("ses-", "")
            fig.savefig(
                output_root / f"group_evoked_{ses_name}.png",
                dpi=300, bbox_inches="tight"
            )
            plt.close(fig)
            print(f"  ✓ Saved group evoked plot → group_evoked_{ses_name}.png")
    else:
        print(f"  ⚠️ No epoch averages file found at {epochs_path}")

    # -------------------------
    # METRICS VIOLIN PLOTS
    # -------------------------
    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path)
        metrics_df = _add_group(metrics_df, subject_group_map)

        fig = _plot_metrics(metrics_df, CONFIG)
        fig.savefig(
            output_root / "group_metrics.png",
            dpi=300, bbox_inches="tight"
        )
        plt.close(fig)
        print(f"  ✓ Saved group metrics plot → group_metrics.png")
    else:
        print(f"  ⚠️ No metrics file found at {metrics_path}")


# =============================================================================
# PRIVATE HELPERS
# =============================================================================

def _add_group(df, subject_group_map):
    """Add group column from subject_group_map."""

    def _lookup(subject):
        sub_num = int(subject.split("-")[1])
        for key in [f"sub-{sub_num:02d}", sub_num, f"sub-{sub_num}", subject]:
            if key in subject_group_map:
                return subject_group_map[key]
        return None

    df = df.copy()
    df["group"] = df["subject"].apply(_lookup)
    missing = df["group"].isna().sum()
    if missing > 0:
        print(f"  ⚠️ Could not find group for {missing} rows")
    return df


def _plot_evoked_response(ses_df, session, CONFIG, show_se=True):
    """
    Plot group-average stimulus-evoked pupil response for one session.
    Replicates the style of the reference figure with optional SE shading.

    Parameters
    ----------
    ses_df : pd.DataFrame
        Epoch averages filtered to one session, with group column added.
    session : str
    show_se : bool
        Whether to show shaded standard error band.
    """

    ses_name = session.replace("ses-", "").capitalize()

    # plot_labels = {
    #     ("healthy", "2back"): ("Healthy – 2-back", "#009acd", "dotted"),
    #     ("healthy", "3back"): ("Healthy – 3-back", "#009acd", "-"),
    #     ("mci",     "2back"): ("MCI – 2-back",     "#cd5555", "dotted"),
    #     ("mci",     "3back"): ("MCI – 3-back",     "#cd5555", "-"),
    # }

    # POSTER COLORS
    # plot_labels = {
    #     ("healthy", "2back"): ("HC – 2-back", "#79C243", "dotted"),
    #     ("healthy", "3back"): ("HC – 3-back", "#79C243", "-"),
    #     ("mci",     "2back"): ("MCI – 2-back",     "#444243", "dotted"),
    #     ("mci",     "3back"): ("MCI – 3-back",     "#444243", "-"),
    # }

    plot_labels = {
        ("healthy", "2back"): ("HC – 2-back", "#009acd", "dotted"),
        ("healthy", "3back"): ("HC – 3-back", "#009acd", "-"),
        ("mci",     "2back"): ("MCI – 2-back",     "#cd5555", "dotted"),
        ("mci",     "3back"): ("MCI – 3-back",     "#cd5555", "-"),
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for (group, task), (label, color, linestyle) in plot_labels.items():
        mask = (
            (ses_df["group"].str.lower() == group) &
            (ses_df["task"] == task)
        )
        subset = ses_df[mask]

        if subset.empty:
            continue

        # Pivot to subjects × timepoints
        pivot = subset.pivot_table(
            index="subject", columns="time", values="pupil"
        )
        times    = pivot.columns.values
        data_arr = pivot.values  # shape: (n_subjects, n_timepoints)

        avg = np.nanmean(data_arr, axis=0) / 1000.0 # Divided by 1000 since it's arbitrary units anyway
        n   = data_arr.shape[0]

        ax.plot(
            times, avg,
            label=f"{label} (n={n})",
            color=color, linewidth=2.5, linestyle=linestyle
        )

        if show_se and n > 1:
            se = np.nanstd(data_arr, axis=0, ddof=1) / np.sqrt(n) / 1000.0
            ax.fill_between(
                times, avg - se, avg + se,
                color=color, alpha=0.1
            )

    # Stimulus onset line
    ax.axvline(0, color="black", linestyle="--", alpha=0.7, linewidth=1.5)
    ymin, ymax = ax.get_ylim()
    ax.text(
        0, ymax - 1,
        "stimulus onset",
        rotation=90, va="top", ha="left",
        fontsize=16, color="black",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=2)
    )

    ax.set_xlabel("Time (s)", fontsize=22, labelpad=15)
    ax.set_ylabel("Pupil size (a.u.)", fontsize=22)
    ax.tick_params(axis="both", which="major", labelsize=16)
    # ax.set_title("Group-average pupil response", fontsize=16, fontweight="bold")
    ax.legend(fontsize=16, loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def _plot_metrics(metrics_df, CONFIG):
    """
    Violin plots with individual subject points colorcoded by group
    for all pupil and saccade metrics.

    One panel per metric, x-axis = session, colored by group.
    """

    # Compute per-subject per-session per-task means
    pupil_metrics = [
        "mean_baseline_pupil_size",
        "peak_pupil_amplitude",
        "peak_latency_s",
        "tepr",
        "peak_velocity_diff",
        "peak_velocity_gradient",
        "peak_velocity_savgol",
    ]
    saccade_metrics = [
        "sac_latency_s",
        "sac_dist_to_tgt_px",
        "fix_duration_s",
        "bcea_vs_centroid",
        "bcea_vs_target",
        "landing_var_x_px",
        "landing_var_y_px",
        "landing_bcea",
    ]

    all_metrics = [m for m in pupil_metrics + saccade_metrics if m in metrics_df.columns]

    # Subject-level means across trials
    subj_df = (
        metrics_df
        .groupby(["subject", "session", "task", "group"])[all_metrics]
        .mean()
        .reset_index()
    )

    sessions = ["ses-placebo", "ses-frontal", "ses-frontoparietal"]
    ses_labels = ["Placebo", "Frontal", "Frontoparietal"]
    tasks = ["2back", "3back"]

    group_colors = {"healthy": "#009acd", "mci": "#cd5555"}
    groups       = ["healthy", "mci"]

    n_metrics = len(all_metrics)
    n_cols    = 4
    n_rows    = int(np.ceil(n_metrics / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 4.5, n_rows * 4),
    )
    axes = axes.flatten()

    for ax_idx, metric in enumerate(all_metrics):
        ax = axes[ax_idx]

        # x positions: sessions × tasks side by side
        x_positions = []
        x_labels    = []
        x           = 0

        for ses_idx, session in enumerate(sessions):
            for task in tasks:
                x_positions.append((x, session, task))
                x_labels.append(f"{ses_labels[ses_idx]}\n{task}")
                x += 1
            x += 0.5  # gap between sessions

        group_offset = {"healthy": -0.18, "mci": 0.18}

        for x_pos, session, task in x_positions:
            for group in groups:
                mask = (
                    (subj_df["session"] == session) &
                    (subj_df["task"]    == task)    &
                    (subj_df["group"].str.lower() == group)
                )
                values = subj_df.loc[mask, metric].dropna().values
                color  = group_colors[group]
                offset = group_offset[group]

                if len(values) < 3:
                    continue

                # Violin
                parts = ax.violinplot(
                    values,
                    positions=[x_pos + offset],
                    widths=0.3,
                    showmedians=True,
                    showextrema=False
                )
                for pc in parts["bodies"]:
                    pc.set_facecolor(color)
                    pc.set_alpha(0.4)
                parts["cmedians"].set_color(color)
                parts["cmedians"].set_linewidth(2)

                # Individual points with jitter
                jitter = np.random.uniform(-0.06, 0.06, size=len(values))
                ax.scatter(
                    np.full(len(values), x_pos + offset) + jitter,
                    values,
                    color=color, s=15, alpha=0.7, zorder=3
                )

                # Group mean marker
                mean_val = np.nanmean(values)
                ax.scatter(
                    x_pos + offset,
                    mean_val,
                    marker="D",
                    s=48,
                    facecolor="white",
                    edgecolor=color,
                    linewidth=1.5,
                    zorder=4,
                )

        ax.set_xticks([x for x, _, _ in x_positions])
        ax.set_xticklabels(x_labels, fontsize=7)
        ax.set_title(metric.replace("_", " "), fontsize=9)
        ax.grid(True, alpha=0.2, axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide unused axes
    for ax_idx in range(len(all_metrics), len(axes)):
        axes[ax_idx].set_visible(False)

    # Group legend
    legend_handles = [
        mpatches.Patch(color=group_colors["healthy"], alpha=0.7, label="Healthy"),
        mpatches.Patch(color=group_colors["mci"],     alpha=0.7, label="MCI"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        fontsize=11,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01)
    )

    fig.suptitle(
        "Group metrics — all sessions and tasks",
        fontsize=14, fontweight="bold", y=1.01
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    return fig