"""
Script: modify_ipi_csv.py
Overview:
    This script modifies the IPI values in a CSV file using a trained IPI
    prediction model, without touching the original PCAP. It is inspired by
    the legacy notebooks/processing/replace_IPI_only_DL.ipynb (MODE == 4), but
    adapted to the new project structure and artifacts produced by
    train_ipi_model.py.

Usage example:
    python3 src/flowsync/scripts/modify_ipi_csv.py \
      --input_csv data/extracted_csv/DS_Forza_DL_UL.csv \
      --model models/Forza_DL/Forza_DL.keras \
      --scaler models/Forza_DL/Forza_DL_scaler.joblib \
      --remaining_csv models/Forza_DL/Forza_DL_packets_N_to_end_scaled.csv \
      --window_size 10 \
      --modify_percent 100 \
      --output_csv Forza_DL_modified_IPI.csv

Notes:
    - The model is assumed to have been trained on scaled IPI values using
      MinMaxScaler(feature_range=(-1, 1)) on the same units as the input CSV
      (typically seconds), as in train_ipi_model.py.
    - The "remaining" CSV (packets_N_to_end_scaled) is used as a source of
      already-scaled windows, similar to MODE == 4 in the legacy notebook.
"""

import argparse
import os
from collections import deque

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.models import load_model

from flowsync.models.architectures import mish


def parse_args():
    parser = argparse.ArgumentParser(description="Modify IPI values in a CSV using a trained model.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV (original packets with IPI column).")
    parser.add_argument("--remaining_csv", type=str, required=True, help="Path to CSV with remaining scaled packets (packets_N_to_end_scaled_*.csv).")
    parser.add_argument("--model", type=str, required=True, help="Path to trained .keras model.")
    parser.add_argument("--scaler", type=str, required=True, help="Path to remaining scaler .joblib.")
    parser.add_argument("--window_size", type=int, default=10, help="Sliding window size used during training.")
    parser.add_argument("--modify_percent", type=int, default=100, help="Percentage of packets within each window to modify.")
    parser.add_argument(
        "--output_mode",
        type=str,
        choices=["modified_full", "synthetic_only"],
        default="modified_full",
        help=(
            "Output mode: 'modified_full' (default) writes full CSV with IPIs overwritten in selected "
            "rows; 'synthetic_only' writes a CSV containing only the generated synthetic IPIs, one "
            "per row."
        ),
    )
    parser.add_argument(
        "--synthetic_count",
        type=int,
        default=None,
        help=(
            "When output_mode='synthetic_only', optionally limit the number of synthetic IPIs "
            "written to this value. If greater than the number available, the script will warn "
            "and use the maximum possible. Ignored for other output modes."
        ),
    )
    parser.add_argument(
        "--direction_column",
        type=str,
        default="Direction",
        help="Name of the column indicating traffic direction (e.g., 'Direction').",
    )
    parser.add_argument(
        "--direction_to_modify",
        type=str,
        default=None,
        help=(
            "Only rows whose value in direction_column matches this string will have their IPI "
            "overwritten. If not set, all rows selected by the window logic may be modified."
        ),
    )
    parser.add_argument("--output_csv", type=str, required=True, help="Path to output CSV with modified IPIs.")
    return parser.parse_args()


def generate_model_prediction(window_packets, scaler, model, num_features):
    """Generate predictions for a window of scaled IPIs.

    window_packets: deque/list of arrays, each shaped (num_features,).
    Returns predictions in original (unscaled) space, shape (1, num_features).
    """
    # Stack to (window_size, num_features) and add batch dim -> (1, W, F)
    window = np.expand_dims(np.stack(window_packets, axis=0), axis=0)

    # Model expects scaled input; we assume window_packets are already scaled
    preds = model(window, training=False)

    if preds.shape[1] != num_features:
        raise ValueError(f"Unexpected predictions shape: {preds.shape}. Expected (batch_size, {num_features}).")

    # Inverse scale
    num_predictions = preds.shape[0]
    if preds.shape[1] > num_features:
        raise ValueError(
            f"Predictions have more features ({preds.shape[1]}) than expected num_features ({num_features})."
        )

    if num_features != preds.shape[1]:
        dummy = np.random.rand(num_predictions, num_features - preds.shape[1])
        preds_with_dummy = np.hstack([preds, dummy])
        preds_original = scaler.inverse_transform(preds_with_dummy)[:, : preds.shape[1]]
    else:
        preds_original = scaler.inverse_transform(preds)
        preds_original = preds_original[:, : preds.shape[1]]

    return preds_original


def main():
    args = parse_args()

    if args.output_mode != "synthetic_only" and args.synthetic_count is not None:
        print("Warning: --synthetic_count is ignored when output_mode != 'synthetic_only'.")

    # Resolve input CSV path
    input_path = args.input_csv
    # If only a filename is given, look under data/processed/
    if not os.path.isabs(input_path) and os.path.dirname(input_path) == "":
        base_dir = os.path.join("data", "processed")
        input_path = os.path.join(base_dir, input_path)

    # Load original and remaining CSVs
    print(f"Loading input CSV: {input_path}")
    packets = pd.read_csv(input_path, header=0)

    print(f"Loading remaining scaled CSV: {args.remaining_csv}")
    remaining_packets = pd.read_csv(args.remaining_csv, header=0)

    if "IPI" not in packets.columns:
        raise ValueError("Input CSV must contain an 'IPI' column.")

    if "IPI" not in remaining_packets.columns:
        raise ValueError("Remaining CSV must contain an 'IPI' column.")

    # Direction handling
    direction_column = args.direction_column
    use_direction_filter = args.direction_to_modify is not None

    if use_direction_filter and direction_column not in packets.columns:
        raise ValueError(
            f"direction_to_modify was set, but direction_column '{direction_column}' was not found in input CSV."
        )

    # Load model and scaler
    print(f"Loading model: {args.model}")
    model = load_model(args.model, custom_objects={"mish": mish, "log_cosh_loss": "mae"})

    print(f"Loading scaler: {args.scaler}")
    scaler = joblib.load(args.scaler)

    window_size = args.window_size
    modify_count = round((args.modify_percent / 100.0) * window_size)

    print(f"Window size: {window_size}, modify_percent: {args.modify_percent}, modify_count: {modify_count}")

    # We assume a single feature (IPI) as in train_ipi_model.py
    num_features = 1

    window_packets = deque(maxlen=window_size)
    modified_packets = []
    synthetic_ipis = []
    remaining_offset = 0

    total_packets = len(packets)
    print(f"Total packets in input CSV: {total_packets}")

    for i, packet in tqdm(packets.iterrows(), total=total_packets, desc="Processing packets", unit="packet"):
        modified_packet = packet.copy()

        # --- Decide if this row should be modified ---
        # 1) Index-based selection via modify_percent
        index_selected = (i + 1) % window_size < modify_count

        # 2) Optional direction-based selection
        if use_direction_filter:
            dir_val = packet[direction_column]
            direction_selected = dir_val == args.direction_to_modify
        else:
            direction_selected = True

        should_modify = index_selected and direction_selected

        if should_modify:
            # Ensure we have a full window from remaining_packets, which are already scaled
            if remaining_offset + window_size <= len(remaining_packets):
                next_window = remaining_packets.iloc[remaining_offset : remaining_offset + window_size]
                remaining_offset += window_size

                window_packets.clear()
                for _, row in next_window.iterrows():
                    # remaining CSV is scaled IPI only; shape (1,)
                    window_packets.append(np.array([row["IPI"]], dtype=np.float32))

                preds_original = generate_model_prediction(window_packets, scaler, model, num_features)

                # We only have one target feature: IPI
                predicted_ipi = float(preds_original[-1][0])

                # Replace IPI or collect synthetic-only, depending on output_mode
                if args.output_mode == "modified_full":
                    # Replace IPI (units match those used during training and in the scaler)
                    modified_packet["IPI"] = predicted_ipi
                else:
                    synthetic_ipis.append(predicted_ipi)

                    # Periodic progress logging for synthetic-only mode
                    if args.synthetic_count is not None:
                        # Log every 1000 synthetic IPIs generated (and at exactly synthetic_count)
                        if len(synthetic_ipis) % 1000 == 0 or len(synthetic_ipis) == args.synthetic_count:
                            print(
                                f"Generated {len(synthetic_ipis)} synthetic IPIs "
                                f"(target: {args.synthetic_count})."
                            )

                    # Early stop if we have reached the requested number of synthetic IPIs
                    if args.synthetic_count is not None and len(synthetic_ipis) >= args.synthetic_count:
                        modified_packets.append(modified_packet)
                        break

        else:
            # Only append modified_packet if we did not break out of the loop early
            modified_packets.append(modified_packet)

    if args.output_mode == "synthetic_only":
        total_synth = len(synthetic_ipis)
        desired = args.synthetic_count

        if desired is None:
            effective = total_synth
        else:
            if desired > total_synth:
                print(
                    f"Requested synthetic_count={desired}, but only {total_synth} synthetic IPIs "
                    "could be generated; using the maximum available."
                )
                effective = total_synth
            else:
                effective = desired

        synthetic_df = pd.DataFrame({"IPI": synthetic_ipis[:effective]})
        print(f"Total synthetic IPIs generated: {total_synth}")
        # Determine final output path
        output_path = args.output_csv
        # If only a filename is given, save under data/generated/
        if not os.path.isabs(output_path) and os.path.dirname(output_path) == "":
            base_dir = os.path.join("data", "generated")
            os.makedirs(base_dir, exist_ok=True)
            output_path = os.path.join(base_dir, output_path)
        else:
            out_dir = os.path.dirname(output_path)
            if out_dir and not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)

        print(f"Writing {effective} synthetic IPIs to: {output_path}")
        synthetic_df.to_csv(output_path, index=False)
    else:
        modified_df = pd.DataFrame(modified_packets)

        output_path = args.output_csv
        if not os.path.isabs(output_path) and os.path.dirname(output_path) == "":
            base_dir = os.path.join("data", "generated")
            os.makedirs(base_dir, exist_ok=True)
            output_path = os.path.join(base_dir, output_path)
        else:
            out_dir = os.path.dirname(output_path)
            if out_dir and not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)

        print(f"Saving modified CSV to: {output_path}")
        modified_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
