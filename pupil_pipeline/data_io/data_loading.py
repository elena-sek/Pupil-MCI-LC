"""
DATA LOADING UTILITIES
======================

Handles all interaction with raw eye-tracking data.

Responsibilities:
-----------------
- Discover available subjects
- Load pupil recordings
- Load behavioral event markers
- Load Eyelink saccade/fixation events
- Standardize data into a common Recording object

Design principles:
------------------
- NO preprocessing occurs here.
- NO filtering, interpolation, or baseline correction.
- Only reading and organizing data.

Why this separation exists:
---------------------------
Raw data formats frequently change during projects.
Keeping loading logic isolated prevents changes from
breaking preprocessing or analysis code.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import mne


# =============================================================================
# PUBLIC FUNCTIONS
# =============================================================================

def list_subjects(CONFIG):
    root = CONFIG["RAW_ROOT"]
    subs = sorted(p.name for p in root.glob("sub-*"))

    if CONFIG["SUBJECTS"] is None:
        return subs

    return CONFIG["SUBJECTS"]


def load_recording(subject, session, task, CONFIG):
    try:
        raw = _load_pupil(subject, session, task, CONFIG)
        beh_data = _load_behavioral(CONFIG)
        _inject_stimulus_onsets(raw, beh_data, subject, session, task)
        return raw

    except Exception as e:
        print(f"  ⚠️  Could not load {subject} {session} {task}: {e}")
        return None
    
def load_rest_recording(subject, CONFIG):
    root = CONFIG["RAW_ROOT"]
    filename = f"{subject}_ses-baseline_task-rest_eyetrack.asc"
    filepath = root / subject / "ses-baseline" / "eyetrack" / filename
    try:
       raw = mne.io.read_raw_eyelink(filepath)
       return raw
    except Exception as e:
        print(f"  ⚠️  Could not load resting state data for {subject}: {e}")
        return None


# =============================================================================
# PRIVATE HELPERS
# =============================================================================

def _load_pupil(subject, session, task, CONFIG):
    root = CONFIG["RAW_ROOT"]
    filename = f"{subject}_{session}_task-{task}_eyetrack.asc"
    filepath = root / subject / session / "eyetrack" / filename
    return mne.io.read_raw_eyelink(filepath)


def _load_behavioral(CONFIG):
    return pd.read_csv(CONFIG["BEH_FILE"])


def _inject_stimulus_onsets(raw, beh_data, subject, session, task):
    task_num = int(''.join(filter(str.isdigit, task)))
    sub_id = int(subject.split("-")[1])
    ses_name = session.replace("ses-", "")

    beh_sub = beh_data[
        (beh_data["Subject"] == sub_id) &
        (beh_data["stimulation"] == ses_name) &
        (beh_data["task"] == task_num)
    ]

    if beh_sub.empty:
        raise RuntimeError(
            f"No behavioral data found for {subject} {session} {task}"
        )

    events_df = raw.annotations.to_data_frame()
    events_df["onset_ms"] = raw.annotations.onset * 1000

    block_events = events_df[
        events_df["description"].str.contains("Trial Ready", case=False)
    ].sort_values("onset_ms")

    beh_blocks = (
        beh_sub.groupby("block_id")["last_cross_onset_time"]
        .first()
        .reset_index()
    )

    if len(block_events) != len(beh_blocks):
        raise RuntimeError(
            f"Block count mismatch for {subject} {session} {task}: "
            f"behavioral={len(beh_blocks)}, eyetracking={len(block_events)}"
        )

    beh_blocks["eyelink_block_onset_ms"] = block_events["onset_ms"].values
    beh_blocks["time_offset_ms"] = (
        beh_blocks["eyelink_block_onset_ms"]
        - beh_blocks["last_cross_onset_time"]
    )

    beh_sub = beh_sub.merge(
        beh_blocks[["block_id", "time_offset_ms"]],
        on="block_id",
        how="left"
    )

    stim_onsets = (beh_sub["square_onset"] + beh_sub["time_offset_ms"]) / 1000

    new_annotations = mne.Annotations(
        onset=stim_onsets.values,
        duration=np.zeros(len(stim_onsets)),
        description=["Stimulus"] * len(stim_onsets),
        orig_time=raw.annotations.orig_time
    )

    raw.set_annotations(raw.annotations + new_annotations)