"""
EXTRACTING PUPIL METRICS FROM RESTING STATE RECORDINGS
=================================

"""
import numpy as np
import pandas as pd

def extract_rest_metrics(pupil_df, pupil_col, CONFIG):
    # Allow callers to pass preprocess output tuple (df, pct_blink_interp, pct_despiked).
    if isinstance(pupil_df, tuple):
        pupil_df = pupil_df[0]

    vals = pupil_df[pupil_col].dropna().values
    fix_mean = vals.mean()
    fix_sd   = vals.std(ddof=1)
    fix_cv   = (fix_sd / fix_mean) * 100 if fix_mean != 0 else np.nan

    return fix_mean, fix_sd, fix_cv