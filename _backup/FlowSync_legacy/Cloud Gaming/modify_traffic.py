"""
Script: modify_traffic.py
Overview:
    This script modifies the timestamps of packets in a PCAP file based on the
    Inter-Packet Intervals (IPI) predicted by a trained model. It ensures the
    synthetic traffic follows the patterns learned by the model.

Usage:
    python3 modify_traffic.py --input_pcap <path> --output_pcap <path> --model <path> --scaler <path> --start_index <N>

Example:
    python3 modify_traffic.py \
      --input_pcap out_Forza_W.pcap \
      --output_pcap Forza_Synthesized.pcap \
      --model Forza_DL.keras \
      --scaler Forza_DL_scaler.joblib \
      --start_index 10000
"""
import argparse
import sys
import numpy as np
import pandas as pd
import joblib
from scapy.all import rdpcap, wrpcap, IP, UDP, TCP
from tensorflow.keras.models import load_model
import tensorflow as tf

def parse_args():
    parser = argparse.ArgumentParser(description="Modify PCAP IPIs using trained model.")
    parser.add_argument("--input_pcap", type=str, required=True, help="Path to input PCAP.")
    parser.add_argument("--output_pcap", type=str, help="Path to output modified PCAP (optional).")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to output CSV with IPI, PS, Class.")
    parser.add_argument("--model", type=str, required=True, help="Path to trained .keras model.")
    parser.add_argument("--scaler", type=str, required=True, help="Path to scaler .joblib used during training.")
    parser.add_argument("--window_size", type=int, default=10, help="Window size used in model.")
    parser.add_argument("--start_index", type=int, default=0, help="Packet index to start modification from.")
    parser.add_argument("--direction", type=str, choices=["DL", "UL"], required=True, help="Traffic direction to modify (DL or UL).")
    parser.add_argument("--client_ip", type=str, help="Client IP address to distinguish UL (Src=Client) from DL.")
    parser.add_argument("--limit", type=int, default=0, help="Max number of packets to process (0 for all).")
    return parser.parse_args()

def get_packet_direction(pkt, client_ip):
    if not client_ip:
        return None
    
    # Try to find IP layer
    if pkt.haslayer('IP'):
        src = pkt['IP'].src
        if src == client_ip:
            return "UL"
        else:
            return "DL"
    return None

def compute_ipi(packets, idx):
    if idx == 0:
        return 0.0
    return float(packets[idx].time - packets[idx-1].time)

def main():
    args = parse_args()
    
    print(f"Loading model: {args.model}")
    # Custom object needed for mish activation if saved with it, though usually Keras handles it if it's a function.
    # We might need to pass custom_objects if 'mish' was saved as a custom function.
    try:
        model = load_model(args.model)
    except Exception as e:
        # Fallback for custom objects if needed
        print(f"Standard load failed, trying with custom objects: {e}")
        def mish(x):
            return x * tf.math.tanh(tf.math.softplus(x))
        model = load_model(args.model, custom_objects={'mish': mish, 'log_cosh_loss': 'mae'}) # loss might be mae as per training script

    print(f"Loading scaler: {args.scaler}")
    scaler = joblib.load(args.scaler)
    
    print(f"Reading PCAP: {args.input_pcap}")
    packets = rdpcap(args.input_pcap)
    n_packets = len(packets)
    print(f"Loaded {n_packets} packets.")
    
    if args.limit > 0 and args.limit < n_packets:
        print(f"Limiting processing to first {args.limit} packets.")
        n_packets = args.limit
    
    modified_packets = []
    csv_rows = []
    
    # We need to maintain a timeline.
    # We can't modify packets in place in the original list if we rely on original IPIs for the window.
    # But for the output, we construct a new list.
    
    start_mod_idx = max(args.start_index, args.window_size)
    
    if args.output_pcap:
        for i in range(start_mod_idx):
            modified_packets.append(packets[i])
        last_modified_time = packets[start_mod_idx-1].time
        
    print("Collecting initial packet stats...")
    for i in range(start_mod_idx):
        ipi = compute_ipi(packets, i)
        
        pkt_dir = get_packet_direction(packets[i], args.client_ip)
        if pkt_dir:
            dir_label = "Uplink" if pkt_dir == "UL" else "Downlink"
        else:
            dir_label = "Uplink" if args.direction == "UL" else "Downlink"
            
        csv_rows.append({"IPI": ipi * 1e6, "PS": len(packets[i]), "Direction": dir_label})
        
    print(f"Starting generation at index {start_mod_idx}...")
    
    print("Pre-calculating IPIs...")
    all_ipis = []
    for i in range(1, n_packets):
        all_ipis.append(float(packets[i].time - packets[i-1].time))
    
    X_batch = []
    original_packet_indices = []
    indices_to_predict = []
    
    print("Creating batches...")
    for i in range(start_mod_idx, n_packets):
        start_w = i - args.window_size
        end_w = i
        window = all_ipis[start_w:end_w]
        
        pkt_dir = get_packet_direction(packets[i], args.client_ip)
        
        should_predict = True
        if args.client_ip:
            if pkt_dir != args.direction:
                should_predict = False
        
        X_batch.append(window)
        original_packet_indices.append(i)
        
        if should_predict:
            indices_to_predict.append(len(X_batch) - 1)
        
    X_batch = np.array(X_batch) * 1e6
    
    print("Scaling inputs...")
    N, W = X_batch.shape
    X_flat = X_batch.reshape(-1, 1)
    X_flat_df = pd.DataFrame(X_flat, columns=['IPI'])
    X_scaled_flat = scaler.transform(X_flat_df)
    X_batch_scaled = X_scaled_flat.reshape(N, W, 1)
    
    print("Running batch prediction...")
    preds_scaled = model.predict(X_batch_scaled, batch_size=1024, verbose=1)
    
    print("Inverse scaling predictions...")
    preds_micros = scaler.inverse_transform(preds_scaled)
    
    print("Saving results...")
    for j, idx in enumerate(original_packet_indices):
        original_pkt = packets[idx]
        
        pkt_dir = get_packet_direction(original_pkt, args.client_ip)
        if pkt_dir:
            dir_label = "Uplink" if pkt_dir == "UL" else "Downlink"
        else:
            dir_label = "Uplink" if args.direction == "UL" else "Downlink"

        use_prediction = True
        if args.client_ip:
            if pkt_dir != args.direction:
                use_prediction = False
        
        if use_prediction:
            pred_val = preds_micros[j][0]
            if pred_val < 0: pred_val = 0
            final_ipi_csv = pred_val
            final_ipi_pcap = pred_val / 1e6
        else:
            orig_ipi_sec = float(original_pkt.time - packets[idx-1].time)
            final_ipi_csv = orig_ipi_sec * 1e6
            final_ipi_pcap = orig_ipi_sec

        csv_rows.append({"IPI": final_ipi_csv, "PS": len(original_pkt), "Direction": dir_label})
        
        if args.output_pcap:
            new_pkt = original_pkt.copy()
            new_time = last_modified_time + final_ipi_pcap
            new_pkt.time = new_time
            last_modified_time = new_time
            modified_packets.append(new_pkt)

    if args.output_pcap:
        print(f"\nSaving modified PCAP to: {args.output_pcap}")
        wrpcap(args.output_pcap, modified_packets)
    
    print(f"\nSaving CSV stats to: {args.output_csv}")
    pd.DataFrame(csv_rows).to_csv(args.output_csv, index=False)
        
    print("Done.")

if __name__ == "__main__":
    main()
