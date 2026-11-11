"""
SAVING UTILITIES
================

Handles all output writing for the pupillometry pipeline.

Responsibilities:
-----------------
- Save trial-level metrics (evoked + saccades) to a single group CSV
- Save per-subject average waveforms for group plotting

Design principles:
------------------
- No processing occurs here.
- All outputs go to CONFIG["OUTPUT_ROOT"].
- Files are appended to across subjects/sessions/tasks so the
  pipeline can be re-run for individual subjects without overwriting
  the whole group file.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def save_results(subject, session, task, evoked, saccades, CONFIG):
    """
    Merge evoked and saccade trial-level metrics and append to the
    group CSV file.

    Parameters
    ----------
    subject, session, task : str
    evoked : pd.DataFrame
        Output of extract_evoked — one row per trial.
    saccades : pd.DataFrame
        Output of extract_saccades — one row per trial.
    CONFIG : dict
    """

    output_path = Path(CONFIG["OUTPUT_ROOT"]) / "group_trial_metrics.csv"

    # Merge evoked and saccade DataFrames on trial_index
    merged = evoked.merge(saccades, on="trial_index", how="outer", suffixes=("_evoked", "_saccade"))

    # Add identifiers
    merged["subject"] = subject
    merged["session"] = session
    merged["task"]    = task

    # Reorder identifier columns to front
    id_cols    = ["subject", "session", "task", "trial_index"]
    other_cols = [c for c in merged.columns if c not in id_cols]
    merged     = merged[id_cols + other_cols]

    # Append to group file, write header only if file doesn't exist yet
    write_header = not output_path.exists()
    merged.to_csv(output_path, mode="a", header=write_header, index=False)

    print(f"  ✓ Saved {len(merged)} trials → {output_path.name}")

def save_rest_results(subject, rest_mean, rest_SD, rest_CV, CONFIG):
    output_path = Path(CONFIG["OUTPUT_ROOT"]) / "resting_state_metrics.csv"

    df = pd.DataFrame([{
        "subject": subject,
        "rest_mean": rest_mean,
        "rest_SD": rest_SD,
        "rest_CV": rest_CV
    }])

    # Append to group file, write header only if file doesn't exist yet
    write_header = not output_path.exists()
    df.to_csv(output_path, mode="a", header=write_header, index=False)



def save_epoch_average(subject, session, task, epochs, times, CONFIG):
    """
    Save the per-subject average waveform for group evoked plotting.

    Parameters
    ----------
    subject, session, task : str
    epochs : np.ndarray, shape (n_trials, n_timepoints)
        Output of epoch_pupil.
    times : np.ndarray
        Time axis in seconds relative to stimulus onset.
    CONFIG : dict
    """

    output_path = Path(CONFIG["OUTPUT_ROOT"]) / "group_epoch_averages.csv"

    avg_waveform = np.nanmean(epochs, axis=0)

    df = pd.DataFrame({
        "subject": subject,
        "session": session,
        "task":    task,
        "time":    times,
        "pupil":   avg_waveform
    })

    write_header = not output_path.exists()
    df.to_csv(output_path, mode="a", header=write_header, index=False)

    print(f"  ✓ Saved epoch average ({len(avg_waveform)} timepoints) → {output_path.name}")

def save_preprocessing_overview(records, CONFIG):
    """
    Write a human-readable preprocessing overview to preprocessing_overview.txt.

    Parameters
    ----------
    records : list of dict
        Each dict contains subject, session, task, group,
        pct_blink_interp, pct_despiked, pct_blink_raw, pct_trials_blink_in_baseline.
    CONFIG : dict
    """
    from pathlib import Path
    import numpy as np

    df = pd.DataFrame(records)
    out = Path(CONFIG["OUTPUT_ROOT"]) / "preprocessing_overview.txt"

    def fmt(mean, sd, n):
        return f"{mean:.2f}% (SD = {sd:.2f}, n = {n})"

    def group_stats(subset, col):
        vals = subset[col].dropna().values
        if len(vals) == 0:
            return "no data"
        return fmt(np.mean(vals), np.std(vals, ddof=1), len(vals))

    lines = []
    lines.append("=" * 80)
    lines.append("PREPROCESSING OVERVIEW")
    lines.append("=" * 80)

    # -------------------------
    # SUBJECT-LEVEL
    # -------------------------
    lines.append("\nSUBJECT-LEVEL SUMMARY")
    lines.append("-" * 80)
    lines.append(f"{'Subject':<12} {'Session':<22} {'Task':<8} "
                 f"{'Blink interp %':>14} {'Despiked %':>12} "
                 f"{'Blink events %':>15} {'Baseline blink %':>17}")
    lines.append("-" * 80)

    for _, row in df.sort_values(["subject", "session", "task"]).iterrows():
        bi  = f"{row['pct_blink_interp']:.2f}%"          if pd.notna(row['pct_blink_interp'])             else "N/A"
        ds  = f"{row['pct_despiked']:.2f}%"               if pd.notna(row['pct_despiked'])                 else "N/A"
        br  = f"{row['pct_blink_raw']:.2f}%"              if pd.notna(row['pct_blink_raw'])                else "N/A"
        bb  = f"{row['pct_trials_blink_in_baseline']:.2f}%" if pd.notna(row['pct_trials_blink_in_baseline']) else "N/A"
        lines.append(f"{row['subject']:<12} {row['session']:<22} {row['task']:<8} "
                     f"{bi:>14} {ds:>12} {br:>15} {bb:>17}")

    # -------------------------
    # GROUP-LEVEL
    # -------------------------
    lines.append("\n" + "=" * 80)
    lines.append("GROUP-LEVEL SUMMARY")
    lines.append("=" * 80)

    lines.append("\nOverall (all subjects, sessions, tasks):")
    lines.append(f"  Blink interpolation      : {group_stats(df, 'pct_blink_interp')}")
    lines.append(f"  Despiking                : {group_stats(df, 'pct_despiked')}")
    lines.append(f"  Blink events (raw)       : {group_stats(df, 'pct_blink_raw')}")
    lines.append(f"  Trials with baseline blink: {group_stats(df, 'pct_trials_blink_in_baseline')}")

    lines.append("\nBy group:")
    for group in ["healthy", "mci"]:
        sub = df[df["group"].str.lower() == group]
        lines.append(f"\n  {group.upper()}")
        lines.append(f"    Blink interpolation      : {group_stats(sub, 'pct_blink_interp')}")
        lines.append(f"    Despiking                : {group_stats(sub, 'pct_despiked')}")
        lines.append(f"    Blink events (raw)       : {group_stats(sub, 'pct_blink_raw')}")
        lines.append(f"    Trials with baseline blink: {group_stats(sub, 'pct_trials_blink_in_baseline')}")

    lines.append("\nBy group and task:")
    for group in ["healthy", "mci"]:
        for task in ["2back", "3back"]:
            sub = df[(df["group"].str.lower() == group) & (df["task"] == task)]
            lines.append(f"\n  {group.upper()} — {task}")
            lines.append(f"    Blink interpolation      : {group_stats(sub, 'pct_blink_interp')}")
            lines.append(f"    Despiking                : {group_stats(sub, 'pct_despiked')}")
            lines.append(f"    Blink events (raw)       : {group_stats(sub, 'pct_blink_raw')}")
            lines.append(f"    Trials with baseline blink: {group_stats(sub, 'pct_trials_blink_in_baseline')}")

    lines.append("\n" + "=" * 80)

    with open(out, "w") as f:
        f.write("\n".join(lines))

    print(f"  ✓ Saved preprocessing overview → {out.name}")
    
# def save_preprocessing_overview(records, CONFIG):
#     """
#     Write a human-readable preprocessing overview to preprocessing_overview.txt.

#     Parameters
#     ----------
#     records : list of dict
#         Each dict contains subject, session, task, group,
#         pct_blink_interp, pct_despiked.
#     CONFIG : dict
#     """
#     from pathlib import Path
#     import numpy as np

#     df = pd.DataFrame(records)
#     out = Path(CONFIG["OUTPUT_ROOT"]) / "preprocessing_overview.txt"

#     def fmt(mean, sd, n):
#         return f"{mean:.2f}% (SD = {sd:.2f}, n = {n})"

#     def group_stats(subset, col):
#         vals = subset[col].dropna().values
#         if len(vals) == 0:
#             return "no data"
#         return fmt(np.mean(vals), np.std(vals, ddof=1), len(vals))

#     lines = []
#     lines.append("=" * 60)
#     lines.append("PREPROCESSING OVERVIEW")
#     lines.append("=" * 60)

#     # -------------------------
#     # SUBJECT-LEVEL
#     # -------------------------
#     lines.append("\nSUBJECT-LEVEL SUMMARY")
#     lines.append("-" * 60)
#     lines.append(f"{'Subject':<12} {'Session':<22} {'Task':<8} "
#                  f"{'Blink interp %':>14} {'Despiked %':>12}")
#     lines.append("-" * 60)

#     for _, row in df.sort_values(["subject", "session", "task"]).iterrows():
#         blink_str = f"{row['pct_blink_interp']:.2f}%" if pd.notna(row['pct_blink_interp']) else "N/A"
#         spike_str = f"{row['pct_despiked']:.2f}%"     if pd.notna(row['pct_despiked'])     else "N/A"
#         lines.append(f"{row['subject']:<12} {row['session']:<22} {row['task']:<8} "
#                      f"{blink_str:>14} {spike_str:>12}")

#     # -------------------------
#     # GROUP-LEVEL
#     # -------------------------
#     lines.append("\n" + "=" * 60)
#     lines.append("GROUP-LEVEL SUMMARY")
#     lines.append("=" * 60)

#     lines.append("\nOverall (all subjects, sessions, tasks):")
#     lines.append(f"  Blink interpolation : {group_stats(df, 'pct_blink_interp')}")
#     lines.append(f"  Despiking           : {group_stats(df, 'pct_despiked')}")

#     lines.append("\nBy group:")
#     for group in ["healthy", "mci"]:
#         sub = df[df["group"].str.lower() == group]
#         lines.append(f"\n  {group.upper()}")
#         lines.append(f"    Blink interpolation : {group_stats(sub, 'pct_blink_interp')}")
#         lines.append(f"    Despiking           : {group_stats(sub, 'pct_despiked')}")

#     lines.append("\nBy group and task:")
#     for group in ["healthy", "mci"]:
#         for task in ["2back", "3back"]:
#             sub = df[(df["group"].str.lower() == group) & (df["task"] == task)]
#             lines.append(f"\n  {group.upper()} — {task}")
#             lines.append(f"    Blink interpolation : {group_stats(sub, 'pct_blink_interp')}")
#             lines.append(f"    Despiking           : {group_stats(sub, 'pct_despiked')}")

#     lines.append("\n" + "=" * 60)

#     with open(out, "w") as f:
#         f.write("\n".join(lines))

#     print(f"  ✓ Saved preprocessing overview → {out.name}")