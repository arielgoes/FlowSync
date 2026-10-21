"""
Script: evaluate_classifier.py
Overview:
    Evaluates synthetic or real traffic traces against pre-trained
    Decision Tree (DT) and Random Forest (RF) classifiers from PixeltoPacket
    to verify preservation of cloud gaming (CG) behavior.

Usage example:
    python3 src/flowsync/scripts/evaluate_classifier.py \
        --trace_dir data/generated \
        --original_dir data/extracted_csv \
        --model_dir models/saved/classifiers
"""

import argparse
import glob
import os
import joblib
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate traffic traces against pre-trained CG classifiers."
    )
    parser.add_argument(
        "--trace_dir",
        type=str,
        default="data/generated",
        help="Directory containing synthetic IPI CSVs.",
    )
    parser.add_argument(
        "--original_dir",
        type=str,
        default="data/extracted_csv",
        help="Directory containing original extracted packet CSVs with Length (PS).",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models/saved/classifiers",
        help="Directory containing pre-trained classifier .joblib files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    dt_path = os.path.join(args.model_dir, "DT_CG_NONCG_Model_Arial.joblib")
    rf_path = os.path.join(args.model_dir, "RF_CG_NONCG_Model_Arial.joblib")

    if not os.path.exists(dt_path) or not os.path.exists(rf_path):
        print(f"Classifier models not found in {args.model_dir}")
        return

    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    dt_model = joblib.load(dt_path)
    rf_model = joblib.load(rf_path)

    print("=" * 70)
    print("Cloud Gaming Traffic Classifier Evaluation (PixeltoPacket)")
    print("=" * 70)
    print(f"{'Trace':<20} | {'Packets':<8} | {'DT CG %':<10} | {'RF CG %':<10} | {'Status'}")
    print("-" * 70)

    games = ["Forza", "Kombat", "Fortnite"]
    directions = ["DL", "UL"]

    for game in games:
        for direction in directions:
            dir_suffix = "downlink_only" if direction == "DL" else "uplink_only"
            synth_csv = os.path.join(
                args.trace_dir, f"{game}_{direction}_synthetic_{dir_suffix}.csv"
            )
            orig_csv = os.path.join(
                args.original_dir, f"DS_{game}_{direction}_only.csv"
            )

            if not os.path.exists(synth_csv) or not os.path.exists(orig_csv):
                continue

            synth_df = pd.read_csv(synth_csv)
            n_samples = len(synth_df)
            orig_df = pd.read_csv(orig_csv, nrows=n_samples)

            # Match lengths
            n_eval = min(len(synth_df), len(orig_df))
            ipi_vals = synth_df["IPI"].values[:n_eval]
            ps_vals = orig_df["Length"].values[:n_eval]

            X = pd.DataFrame({"IPI": ipi_vals, "PS": ps_vals})

            dt_preds = dt_model.predict(X)
            rf_preds = rf_model.predict(X)

            dt_pct = (dt_preds == "CG").mean() * 100
            rf_pct = (rf_preds == "CG").mean() * 100

            status = "Consistent (CG)" if rf_pct >= 80.0 else "Variable"

            trace_label = f"{game} ({direction})"
            print(f"{trace_label:<20} | {n_eval:<8} | {dt_pct:<10.1f} | {rf_pct:<10.1f} | {status}")

    print("=" * 70)


if __name__ == "__main__":
    main()

