"""
RUN GROUP PLOTS ONLY
====================

Standalone entrypoint for regenerating group-level figures from saved outputs.

This script does not run preprocessing. It only reads the saved group output
files and writes the group-level plots into CONFIG["OUTPUT_ROOT"].

Expected inputs:
----------------
- group_trial_metrics.csv
- group_epoch_averages.csv
- GROUP_FILE with subject-to-group labels
"""

from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import CONFIG
from plotting.group_plots import plot_group_results


def main():
    output_root = Path(CONFIG["OUTPUT_ROOT"])
    metrics_path = output_root / "group_trial_metrics.csv"
    epochs_path = output_root / "group_epoch_averages.csv"
    group_file = Path(CONFIG["GROUP_FILE"])

    missing = [path for path in [metrics_path, epochs_path, group_file] if not path.exists()]
    if missing:
        print("Missing required input files:")
        for path in missing:
            print(f"  - {path}")
        print("Nothing was plotted.")
        return

    plot_group_results(CONFIG)


if __name__ == "__main__":
    main()