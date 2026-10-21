#!/usr/bin/env python3

import argparse
import os
import sys

import pandas as pd


def infer_game_name(path: str) -> str:
    """Infer game name from the input path (mimics the notebook logic)."""
    if "Kombat" in path:
        return "Kombat"
    if "Forza" in path:
        return "Forza"
    if "Fortnite" in path:
        return "Fortnite"
    return "Other"


def detect_server_and_client(df: pd.DataFrame, first_packets: int = 50):
    """Detect cloud-gaming server and client IPs using the first packets.

    This mirrors the logic in notebooks/processing/create_direction_column.ipynb:
    - Look at the first N packets.
    - Find the most common Source/Destination pair.
    - Among UDP/RTP packets, decide which side is more often sending to the other.
    """
    required_cols = {"Source", "Destination", "Protocol"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

    df_first = df.head(first_packets).copy()

    if df_first.empty:
        raise ValueError("Input CSV is empty or has fewer than 1 row.")

    pair_counts = (
        df_first.groupby(["Source", "Destination"]).size().reset_index(name="count")
    )
    if pair_counts.empty:
        raise ValueError("Could not compute Source/Destination pair counts.")

    top_pair = pair_counts.sort_values("count", ascending=False).iloc[0]
    ip1, ip2 = top_pair["Source"], top_pair["Destination"]

    udp_rtp = df_first[df_first["Protocol"].isin(["UDP", "RTP"])]

    count_ip1_to_ip2 = (
        (udp_rtp["Source"] == ip1) & (udp_rtp["Destination"] == ip2)
    ).sum()
    count_ip2_to_ip1 = (
        (udp_rtp["Source"] == ip2) & (udp_rtp["Destination"] == ip1)
    ).sum()

    if count_ip1_to_ip2 >= count_ip2_to_ip1:
        cg_server_ip, cg_client_ip = ip1, ip2
    else:
        cg_server_ip, cg_client_ip = ip2, ip1

    return cg_server_ip, cg_client_ip


def assign_direction(row, server_ip: str, client_ip: str) -> str:
    """Assign Uplink/Downlink direction to a single row.

    - UDP/RTP from server->client: Downlink
    - UDP/RTP from client->server: Uplink
    - Everything else: Uplink (matches the notebook fallback)
    """
    protocol = row.get("Protocol")
    src = row.get("Source")
    dst = row.get("Destination")

    if protocol in ["UDP", "RTP"]:
        if src == server_ip and dst == client_ip:
            return "Downlink"
        if src == client_ip and dst == server_ip:
            return "Uplink"

    return "Uplink"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Infer Direction column (Uplink/Downlink) from a DS_*_DL_UL.csv file "
            "and export a CSV with IPI and Direction columns."
        )
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to DS_<Game>_DL_UL.csv (must contain IPI, Source, Destination, Protocol).",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help=(
            "Optional explicit output path. If not set, defaults to "
            "<input_dir>/DS_<Game>_all_packets.csv based on game name."
        ),
    )
    parser.add_argument(
        "--first_packets",
        type=int,
        default=50,
        help="Number of initial packets to use for server/client inference (default: 50).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = os.path.abspath(args.input_csv)
    if not os.path.exists(input_path):
        print(f"Input CSV not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    game = infer_game_name(input_path)
    print(f"Detected game from path: {game}")
    if game == "Other":
        print("Game not supported by direction inference (expected 'Kombat', 'Forza', or 'Fortnite').", file=sys.stderr)
        sys.exit(1)

    print(f"Loading CSV: {input_path}")
    df_all = pd.read_csv(input_path)

    required_cols = {"IPI", "Source", "Destination", "Protocol"}
    missing = required_cols - set(df_all.columns)
    if missing:
        print(f"Input CSV is missing required columns: {sorted(missing)}", file=sys.stderr)
        sys.exit(1)

    print(f"Using first {args.first_packets} packets to infer server/client.")
    server_ip, client_ip = detect_server_and_client(df_all, first_packets=args.first_packets)
    print(f"Detected server IP: {server_ip}, client IP: {client_ip}")

    print("Assigning Direction column (Uplink/Downlink)...")
    df_all["Direction"] = df_all.apply(
        lambda row: assign_direction(row, server_ip=server_ip, client_ip=client_ip), axis=1
    )

    # Keep only the columns that downstream code expects
    output_df = df_all[["IPI", "Direction"]].copy()

    if args.output_csv is not None:
        output_path = os.path.abspath(args.output_csv)
    else:
        # Default to data/processed/DS_<Game>_all_packets.csv to keep stages organized
        output_dir = os.path.join("data", "processed")
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"DS_{game}_all_packets.csv"
        output_path = os.path.join(output_dir, output_filename)

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print(f"Saving output CSV to: {output_path}")
    output_df.to_csv(output_path, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
