"""
SUPPLEMENTARY FIGURE — STIMULUS LOCATIONS
==========================================

Generates a figure illustrating the nine possible stimulus locations
used in the CRstim n-back task, for inclusion in supplementary materials.

Stimulus positions are read directly from the behavioral CSV and converted
using the same logic as extract_saccades(), ensuring exact consistency
with the preprocessing pipeline.

Usage:
------
    python plotting/supplementary.py

Output:
-------
    supplementary_stimulus_locations.png  (saved to CONFIG["OUTPUT_ROOT"])

Design notes:
-------------
- Axes aspect ratio matches the physical screen dimensions from CONFIG.
- Stimulus squares are drawn at their true proportional size.
- Positions are derived from unique square_X / square_Y values in the
  behavioral CSV — no grid assumption is made.
- The fixation cross at screen centre is included for spatial reference.
- All nine locations are numbered in the order they appear in the CSV.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import sys

# Allow running directly from the plotting/ folder or from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CONFIG


def plot_stimulus_locations(CONFIG, save=True, upward_shift_pct=0.02):
    """
    Draw all unique stimulus locations on a to-scale screen outline.

    Positions are read from CONFIG["BEH_FILE"] and converted to pixel
    coordinates using the same formula as extract_saccades().

    Parameters
    ----------
    CONFIG : dict
        Pipeline configuration.
    save : bool
        If True, saves the figure to CONFIG["OUTPUT_ROOT"].

    Returns
    -------
    fig : matplotlib.figure.Figure
    """

    screen_w, screen_h     = CONFIG["SCREEN_RESOLUTION"]
    stim_w_pct, stim_h_pct = CONFIG["STIMULUS_SIZE_PCT"]

    # Stimulus size in pixels (for drawing rectangles)
    stim_w_px = (stim_w_pct * screen_w) / 2
    stim_h_px = (stim_h_pct * screen_h) / 2

    # -------------------------
    # LOAD UNIQUE STIMULUS POSITIONS
    # Replicates the conversion in extract_saccades():
    #   left-edge % → centre pixel
    # -------------------------
    beh = pd.read_csv(CONFIG["BEH_FILE"])

    unique_pos = (
        beh[["square_X", "square_Y"]]
        .drop_duplicates()
        .dropna()
        .reset_index(drop=True)
    )

    # Left-edge percentage → centre in pixels
    centre_x_px = (
        unique_pos["square_X"].str.replace("%", "").astype(float) / 100
        + stim_w_pct / 2
    ) * screen_w

    centre_y_px = (
        unique_pos["square_Y"].str.replace("%", "").astype(float) / 100
        + stim_h_pct / 2
    ) * screen_h

    upward_shift_px = upward_shift_pct * screen_h
    centre_y_px = np.clip(centre_y_px - upward_shift_px, 0, screen_h)

    # Left-edge in pixels (for drawing the rectangle)
    left_px = centre_x_px - stim_w_px / 2
    top_px  = centre_y_px - stim_h_px / 2

    n_locations = len(unique_pos)
    print(f"  Found {n_locations} unique stimulus locations")

    # -------------------------
    # FIGURE
    # Axes are in pixel coordinates so distances are true-to-screen.
    # Figure width/height ratio matches the physical screen aspect ratio.
    # -------------------------
    screen_aspect = CONFIG["SCREEN_WIDTH_CM"] / CONFIG["SCREEN_HEIGHT_CM"]
    fig_width  = 8.0
    fig_height = fig_width / screen_aspect

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # -------------------------
    # SCREEN OUTLINE
    # -------------------------
    screen_rect = patches.Rectangle(
        (0, 0), screen_w, screen_h,
        linewidth=1.5,
        edgecolor="black",
        facecolor="#f8f8f8",
        zorder=0,
    )
    ax.add_patch(screen_rect)

    # -------------------------
    # FIXATION CROSS (spatial reference at screen centre)
    # -------------------------
    cx, cy   = screen_w / 2, screen_h / 2
    cross_px = screen_w * 0.018
    ax.plot([cx - cross_px, cx + cross_px], [cy, cy],
            color="gray", linewidth=1.2, zorder=2)
    ax.plot([cx, cx], [cy - cross_px, cy + cross_px],
            color="gray", linewidth=1.2, zorder=2)
    ax.text(cx + cross_px * 1.4, cy, "fixation",
            va="center", fontsize=7, color="gray", zorder=2)

    # -------------------------
    # STIMULUS SQUARES
    # -------------------------
    for i, (lx, ty, cx_i, cy_i) in enumerate(
        zip(left_px, top_px, centre_x_px, centre_y_px), start=1
    ):
        rect = patches.Rectangle(
            (lx, ty),
            stim_w_px,
            stim_h_px,
            linewidth=0.8,
            edgecolor="black",
            facecolor="black",
            zorder=3,
        )
        ax.add_patch(rect)

        # Location number — placed above each square with a small offset
        ax.text(
            cx_i,
            ty - screen_h * 0.02,
            str(i),
            ha="center", va="bottom",
            fontsize=8,
            color="#333333",
            zorder=4,
        )

    # -------------------------
    # AXIS FORMATTING
    # -------------------------
    ax.set_xlim(0, screen_w)
    ax.set_ylim(screen_h, 0)   # inverted: (0,0) = top-left, matching screen coords

    # Enforce exact screen aspect ratio on the axes box
    ax.set_aspect("equal", adjustable="box")

    # Tick marks at every 10% of each axis
    x_ticks = np.linspace(0, screen_w, 11)
    y_ticks = np.linspace(0, screen_h, 11)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels([f"{int(t / screen_w * 100)}%" for t in x_ticks], fontsize=8)
    ax.set_yticklabels([f"{int(t / screen_h * 100)}%" for t in y_ticks], fontsize=8)

    ax.tick_params(axis="both", which="major", length=3, width=0.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    ax.set_xlabel("Horizontal position (% of screen width)",  fontsize=10)
    ax.set_ylabel("Vertical position (% of screen height)",   fontsize=10)

    ax.set_title(
        f"Possible stimulus locations (n = {n_locations})\n"
        f"Screen: {screen_w} × {screen_h} px  |  "
        f"Stimulus size: {stim_w_pct*100:.0f}% × {stim_h_pct*100:.0f}% of screen  |  "
        f"Viewing distance: {CONFIG['VIEWING_DISTANCE_CM']} cm",
        fontsize=9,
        pad=10,
    )

    plt.tight_layout()

    # -------------------------
    # SAVE
    # -------------------------
    if save:
        out_path = (
            Path(CONFIG["OUTPUT_ROOT"]) / "supplementary_stimulus_locations.png"
        )
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"  ✓ Saved → {out_path}")

    return fig


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    fig = plot_stimulus_locations(CONFIG, save=True)
    plt.show()