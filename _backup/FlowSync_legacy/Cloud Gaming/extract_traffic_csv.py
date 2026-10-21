"""
Script: extract_traffic_csv.py
Overview:
    This script extracts traffic from a PCAP file based on a Wireshark display filter
    and saves it to a CSV file. It automatically calculates the Inter-Packet Interval (IPI)
    and formats the CSV for training the IPI prediction model.

Usage:
    python3 extract_traffic_csv.py --input_pcap <path> --output_csv <path> --filter "<filter>"

Example:
    python3 extract_traffic_csv.py \
      --input_pcap out_Forza_W.pcap \
      --output_csv Forza_DL.csv \
      --filter "ip.src == 13.104.104.176 and ip.dst == 200.18.102.42 and rtp"
"""
import argparse
import pandas as pd
import subprocess
import io
import sys
import csv

def parse_args():
    parser = argparse.ArgumentParser(description="Extract traffic from PCAP to CSV with IPI.")
    parser.add_argument("--input_pcap", type=str, required=True, help="Path to input PCAP file.")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to output CSV file.")
    parser.add_argument("--filter", type=str, required=True, help="Wireshark display filter.")
    return parser.parse_args()

def run_tshark(input_pcap, display_filter):
    # Command to extract fields
    cmd = [
        "tshark",
        "-r", input_pcap,
        "-Y", display_filter,
        "-T", "fields",
        "-E", "header=y",
        "-E", "separator=,",
        "-E", "quote=d",
        "-e", "frame.number",
        "-e", "frame.time_relative",
        "-e", "ip.src",
        "-e", "ip.dst",
        "-e", "_ws.col.Protocol",
        "-e", "frame.len",
        "-e", "_ws.col.Info"
    ]
    
    print(f"Running tshark command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Error running tshark:")
        print(result.stderr)
        sys.exit(1)
        
    return result.stdout

def main():
    args = parse_args()
    
    csv_data = run_tshark(args.input_pcap, args.filter)
    
    if not csv_data:
        print("No packets found matching the filter.")
        sys.exit(0)

    # Load into pandas
    try:
        df = pd.read_csv(io.StringIO(csv_data))
    except pd.errors.EmptyDataError:
        print("No data returned from tshark.")
        sys.exit(0)
    
    rename_map = {
        "frame.number": "No.",
        "frame.time_relative": "Time",
        "ip.src": "Source",
        "ip.dst": "Destination",
        "_ws.col.Protocol": "Protocol",
        "frame.len": "Length",
        "_ws.col.Info": "Info"
    }
    
    df = df.rename(columns=rename_map)
    
    # Calculate IPI
    df['Time'] = pd.to_numeric(df['Time'])
    df['IPI'] = df['Time'].diff().fillna(0)
    
    # Reorder columns
    desired_order = ["No.", "Time", "IPI", "Source", "Destination", "Protocol", "Length", "Info"]
    
    cols = [c for c in desired_order if c in df.columns]
    df = df[cols]
    
    print(f"Saving to {args.output_csv}")
    # Save with quoting all fields to match user sample format
    df.to_csv(args.output_csv, index=False, quoting=csv.QUOTE_ALL)

if __name__ == "__main__":
    main()
