# FlowSync

FlowSync is a research-oriented collection of notebooks, models, and helper
functions for experimenting with network traffic traces, with a particular
focus on cloud gaming versus non-cloud-gaming traffic and inter-packet
interval (IPI)–based analysis.

The repository includes:

- Cloud-gaming IPI prediction notebooks (downlink and uplink)
- Traffic classifiers and pre-trained models
- Utilities to modify packet-level features in datasets
- Shared model architectures and helper utilities

## Repository structure

The project has been reorganized into a modular structure:

- `src/flowsync/`
  - `models/`: Model definitions and architectures.
  - `utils/`: Helper functions for data processing, training, and plotting.
  - `scripts/`: Standalone scripts for traffic modification, training, and extraction.
- `notebooks/`
  - `training/`: Notebooks for training classifiers.
  - `prediction/`: Notebooks for IPI prediction (DL/UL).
  - `processing/`: Notebooks for data processing and packet field modification.
- `models/saved/`
  - Saved `.keras` and `.joblib` models.
- `data/`
  - Raw and processed datasets.
- `_backup/`
  - Legacy folders from the previous structure.

## Installation

1. Create and activate a Python virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install the dependencies:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   You may need to install the package in editable mode to use the `src` modules in notebooks/scripts:
   ```bash
   pip install -e .
   ```
   (Create a `setup.py` or `pyproject.toml` if not present, or add `src` to `PYTHONPATH`)

   Example to add to PYTHONPATH:
   ```bash
   export PYTHONPATH=$PYTHONPATH:$(pwd)/src
   ```

Some notebooks and models can optionally leverage a GPU via TensorFlow and
NVIDIA CUDA libraries if available, but a CPU-only setup also works (with
longer runtimes).

## Usage

Most workflows are implemented as Jupyter notebooks or scripts.

### 1. IPI prediction for cloud-gaming traffic

- Open notebooks in `notebooks/prediction/`.
- Update the `file_path` variable in the initial cells so it points to your
  CSV file containing packet-level features.

### 2. Cloud gaming vs. non–cloud-gaming classification

- Open notebooks in `notebooks/training/`.

### 3. Modifying packet-level fields in datasets

- Open notebooks in `notebooks/processing/`.

### 4. Scripts

- Run scripts from `src/flowsync/scripts/`.

  #### Traffic Extraction (PCAP to CSV)
  
  To extract specific traffic (e.g., Uplink/UL) from a full PCAP file:

  1. **Verify Filters Manually**: Open the PCAP file in Wireshark first to identify the correct source/destination IPs and protocols to ensure you are isolating the desired flow.
  2. **Run the Script**: Use `extract_traffic_csv.py`.
  3. **Important**: Always wrap the `--filter` argument in **single quotes** (`'...'`) to prevent shell expansion issues (especially with `!`).

  **Example (Extracting Kombat Uplink Traffic):**
  ```bash
  python3 src/flowsync/scripts/extract_traffic_csv.py \
    --input_pcap "/path/to/out_Kombat_W.pcap" \
    --output_csv DS_Kombat_UL_only.csv \
    --filter 'ip.src == 200.18.102.42 and ip.dst == 13.104.107.204 and !rtp'
  ```
  The script will automatically create an `extracted_csv/` directory and save the file there.

  #### Dataset Extraction Reference

  Use these specific filters for the provided datasets:

  **1. Forza (`out_Forza_W.pcap`)**
  - **UL Filter**: `ip.src == 200.18.102.42 and ip.dst == 13.104.104.176 and !rtp`
  - **DL Filter**: `ip.src == 13.104.104.176 and ip.dst == 200.18.102.42 and rtp`

  **2. Kombat (`out_Kombat_W.pcap`)**
  - **UL Filter**: `ip.src == 200.18.102.42 and ip.dst == 13.104.107.204 and !rtp`
  - **DL Filter**: `ip.src == 13.104.107.204 and ip.dst == 200.18.102.42 and rtp`

  **3. Fortnite (`out_Fortnite_W.pcap`)**
  - **UL Filter**: `ip.src == 192.168.137.21  and ip.dst == 103.6.210.133 and !rtp`
  - **DL Filter**: `ip.src == 103.6.210.133 and ip.dst == 192.168.137.21 and rtp`

  > **Note**: `tshark` may not automatically decode packets as RTP. If your DL extraction results in 0 packets, you may need to force RTP decoding on the dynamic ports used in the trace (e.g., via `-d udp.port==<PORT>,rtp`).
  > For the sample Fortnite trace, use `-d udp.port==10200,rtp`.

  *On `out_Fortnite_W.pcap` (total display packets: 297962):*

  - UL filter: `ip.src == 192.168.137.21  and ip.dst == 103.6.210.133 and !rtp` (10.7% / 31998 packets)
  - DL filter: `ip.src == 103.6.210.133 and ip.dst == 192.168.137.21 and rtp` (97.3% / 604660 packets)
  > Total percentage is less than 100% because the UL filter considers DTLS packets and any other upstream traffic.

  **`extract_traffic_csv.py` argument reference**

  | Argument       | Type   | Required | Default | Description |
  |---------------|--------|----------|---------|-------------|
  | `--input_pcap`| str    | Yes      | -       | Path to input PCAP file. |
  | `--output_csv`| str    | Yes      | -       | Name of the output CSV file (saved under `data/extracted_csv/`). |
  | `--filter`    | str    | Yes      | -       | Wireshark display filter used to select packets. |
  | `--decode_as` | str    | No       | None    | Optional `tshark` `-d` mapping, e.g. `udp.port==1057,rtp` to force RTP decoding. |

  #### Example Commands to Reproduce Datasets

  The following commands were used to generate the reference CSV datasets:

  **DS_Forza_DL_only.csv** (saved to `data/extracted_csv/`)
  ```bash
  python3 src/flowsync/scripts/extract_traffic_csv.py \
    --input_pcap "data/pcap/out_Forza_W.pcap" \
    --output_csv DS_Forza_DL_only.csv \
    --filter 'ip.src == 13.104.104.176 and ip.dst == 200.18.102.42 and rtp' \
    --decode_as 'udp.port==1057,rtp'
  ```

  **DS_Forza_UL_only.csv** (saved to `data/extracted_csv/`)
  ```bash
  python3 src/flowsync/scripts/extract_traffic_csv.py \
    --input_pcap "data/pcap/out_Forza_W.pcap" \
    --output_csv DS_Forza_UL_only.csv \
    --filter 'ip.src == 200.18.102.42 and ip.dst == 13.104.104.176 and !rtp'
  ```

  **DS_Kombat_DL_only.csv** (saved to `data/extracted_csv/`)
  ```bash
  python3 src/flowsync/scripts/extract_traffic_csv.py \
    --input_pcap "data/pcap/out_Kombat_W.pcap" \
    --output_csv DS_Kombat_DL_only.csv \
    --filter 'ip.src == 13.104.107.204 and ip.dst == 200.18.102.42 and rtp' \
    --decode_as 'udp.port==1098,rtp'
  ```

  **DS_Kombat_UL_only.csv** (saved to `data/extracted_csv/`)
  ```bash
  python3 src/flowsync/scripts/extract_traffic_csv.py \
    --input_pcap "data/pcap/out_Kombat_W.pcap" \
    --output_csv DS_Kombat_UL_only.csv \
    --filter 'ip.src == 200.18.102.42 and ip.dst == 13.104.107.204 and !rtp'
  ```

  **DS_Fortnite_DL_only.csv** (saved to `data/extracted_csv/`)
  ```bash
  python3 src/flowsync/scripts/extract_traffic_csv.py \
    --input_pcap "data/pcap/out_Fortnite_W.pcap" \
    --output_csv DS_Fortnite_DL_only.csv \
    --filter 'ip.src == 103.6.210.133 and ip.dst == 192.168.137.21 and rtp' \
    --decode_as 'udp.port==10200,rtp'
  ```

  **DS_Fortnite_UL_only.csv** (saved to `data/extracted_csv/`)
  ```bash
  python3 src/flowsync/scripts/extract_traffic_csv.py \
    --input_pcap "data/pcap/out_Fortnite_W.pcap" \
    --output_csv DS_Fortnite_UL_only.csv \
    --filter 'ip.src == 192.168.137.21  and ip.dst == 103.6.210.133 and !rtp'
  ```

  #### Example Training Commands

  After extracting the CSVs, you can train IPI prediction models for each game and direction
  using `train_ipi_model.py`. These commands assume you have either installed the package
  in editable mode (`pip install -e .`) or added `src` to `PYTHONPATH`.

  **Forza Downlink (DL)**
  ```bash
  python3 src/flowsync/scripts/train_ipi_model.py \
    --input_file data/extracted_csv/DS_Forza_DL_only.csv \
    --output_prefix Forza \
    --direction DL \
    --n_packets 10000 \
    --output_dir models/Forza_DL
  ```

  **Forza Uplink (UL)**
  ```bash
  python3 src/flowsync/scripts/train_ipi_model.py \
    --input_file data/extracted_csv/DS_Forza_UL_only.csv \
    --output_prefix Forza \
    --direction UL \
    --n_packets 10000 \
    --output_dir models/Forza_UL
  ```

  **Kombat Downlink (DL)**
  ```bash
  python3 src/flowsync/scripts/train_ipi_model.py \
    --input_file data/extracted_csv/DS_Kombat_DL_only.csv \
    --output_prefix Kombat \
    --direction DL \
    --n_packets 10000 \
    --output_dir models/Kombat_DL
  ```

  **Kombat Uplink (UL)**
  ```bash
  python3 src/flowsync/scripts/train_ipi_model.py \
    --input_file data/extracted_csv/DS_Kombat_UL_only.csv \
    --output_prefix Kombat \
    --direction UL \
    --n_packets 10000 \
    --output_dir models/Kombat_UL
  ```

  **Fortnite Downlink (DL)**
  ```bash
  python3 src/flowsync/scripts/train_ipi_model.py \
    --input_file data/extracted_csv/DS_Fortnite_DL_only.csv \
    --output_prefix Fortnite \
    --direction DL \
    --n_packets 10000 \
    --output_dir models/Fortnite_DL
  ```

  **Fortnite Uplink (UL)**
  ```bash
  python3 src/flowsync/scripts/train_ipi_model.py \
    --input_file data/extracted_csv/DS_Fortnite_UL_only.csv \
    --output_prefix Fortnite \
    --direction UL \
    --n_packets 10000 \
    --output_dir models/Fortnite_UL
  ```

  **`train_ipi_model.py` argument reference**

  | Argument         | Type | Required | Default | Description |
  |------------------|------|----------|---------|-------------|
  | `--input_file`   | str  | Yes      | -       | Path to input CSV with an `IPI` column. |
  | `--output_prefix`| str  | No       | `model` | Prefix for all output artifacts (model, scaler, splits, remaining CSV). |
  | `--direction`    | str  | Yes      | -       | Traffic direction label, `DL` or `UL`. Used in output filenames only. |
  | `--epochs`       | int  | No       | 50      | Number of training epochs. |
  | `--window_size`  | int  | No       | 10      | Window size used to build GRU input sequences. |
  | `--n_packets`    | int  | No       | 10000   | Number of first packets to use for training. Remaining packets are saved separately if available. |
  | `--output_dir`   | str  | No       | `.`     | Directory where model, scaler, splits, and remaining CSV are written. |

  #### Direction Inference (DS_*_all_packets.csv)

  To infer a `Direction` column (Uplink/Downlink) from a full DL/UL CSV such as
  `DS_Kombat_DL_UL.csv`, and export a compact dataset with only `IPI` and
  `Direction`. For example:

  ```bash
  python3 src/flowsync/scripts/create_direction_column.py \
    --input_csv data/extracted_csv/DS_Kombat_DL_UL.csv
  # Output (by default): data/processed/DS_Kombat_all_packets.csv
  ```

  ```bash
  python3 src/flowsync/scripts/create_direction_column.py \
    --input_csv data/extracted_csv/DS_Fortnite_DL_UL.csv
  # Output (by default): data/processed/DS_Fortnite_all_packets.csv
  ```

  This script mirrors the legacy notebook logic: it inspects the first
  packets to identify server/client IPs, assigns `Direction`, and writes a
  two-column CSV (`IPI,Direction`).

  When you **manually** construct a `DS_*_packets.csv` file (for example, by
  exporting from Wireshark), you must ensure there is an `IPI` column. This
  `IPI` column should correspond to Wireshark's **"Delta time displayed"**
  field (i.e., the inter-packet time difference).

  **`create_direction_column.py` argument reference**

  | Argument        | Type | Required | Default | Description |
  |-----------------|------|----------|---------|-------------|
  | `--input_csv`   | str  | Yes      | -       | Path to a `DS_<Game>_DL_UL.csv` file with `IPI`, `Source`, `Destination`, `Protocol`. |
  | `--output_csv`  | str  | No       | None    | Optional explicit output path. If omitted, saves to `DS_<Game>_all_packets.csv` in the same folder. |
  | `--first_packets`| int | No       | 50      | Number of initial packets used to infer server and client IPs. |

  #### Generating Synthetic IPIs with `modify_ipi_csv.py`

  The script `modify_ipi_csv.py` can either overwrite IPIs in-place or
  generate a CSV containing only synthetic IPIs:

  - `--output_mode modified_full` (default): write a full CSV like the
    input, but with selected `IPI` values replaced by model predictions.
  - `--output_mode synthetic_only`: write a one-column CSV with only the
    generated synthetic IPIs (`IPI`).

  When the input CSV has a `Direction` column, you can restrict
  modification/generation to a single direction using
  `--direction_column` and `--direction_to_modify`.

  **`modify_ipi_csv.py` argument reference**

  | Argument             | Type | Required | Default          | Description |
  |----------------------|------|----------|------------------|-------------|
  | `--input_csv`        | str  | Yes      | -                | Path to input CSV with at least an `IPI` column (and optionally a direction column). |
  | `--remaining_csv`    | str  | Yes      | -                | Path to `*_packets_N_to_end_scaled.csv` produced by `train_ipi_model.py`. |
  | `--model`            | str  | Yes      | -                | Path to trained `.keras` IPI prediction model. |
  | `--scaler`           | str  | Yes      | -                | Path to `.joblib` scaler used during training (for inverse-scaling predictions). |
  | `--window_size`      | int  | No       | 10               | Sliding window size; must match the value used during training. |
  | `--modify_percent`   | int  | No       | 100              | Percentage of packets within each window that are eligible for modification. |
  | `--output_mode`      | str  | No       | `modified_full`  | `modified_full` overwrites IPI in the full CSV; `synthetic_only` writes only generated IPIs. |
  | `--synthetic_count`  | int  | No       | None             | When `output_mode=synthetic_only`, cap the number of synthetic IPIs written. |
  | `--direction_column` | str  | No       | `Direction`      | Name of the column indicating traffic direction, if present. |
  | `--direction_to_modify` | str | No    | None             | Restrict modification to rows whose direction matches this value (e.g., `Downlink`). |
  | `--output_csv`       | str  | Yes      | -                | Path to output CSV (full modified dataset or synthetic-only IPIs). |

  **Example: Forza Downlink (full modified CSV)** (saved to `data/generated/`)
  ```bash
  python3 src/flowsync/scripts/modify_ipi_csv.py \
    --input_csv data/extracted_csv/DS_Forza_DL_only.csv \
    --model models/Forza_DL/Forza_DL.keras \
    --scaler models/Forza_DL/Forza_DL_scaler.joblib \
    --remaining_csv models/Forza_DL/Forza_DL_packets_N_to_end_scaled.csv \
    --window_size 10 \
    --modify_percent 100 \
    --output_mode modified_full \
    --output_csv data/generated/Forza_DL_modified_IPI.csv
  ```

  **Example: Kombat Downlink (synthetic-only IPIs)** (saved to `data/generated/`)

  Using a DL-only input (already filtered to downlink packets):

  ```bash
  python3 src/flowsync/scripts/modify_ipi_csv.py \
    --input_csv data/extracted_csv/DS_Kombat_DL_only.csv \
    --model models/Kombat_DL/Kombat_DL.keras \
    --scaler models/Kombat_DL/Kombat_DL_scaler.joblib \
    --remaining_csv models/Kombat_DL/Kombat_DL_packets_N_to_end_scaled.csv \
    --window_size 10 \
    --modify_percent 100 \
    --output_mode synthetic_only \
    --output_csv data/generated/Kombat_DL_synthetic_only.csv
  ```

  If the input instead mixes directions and has a `Direction` column
  (e.g., `DS_Kombat_all_packets.csv`), you can restrict to Downlink rows
  only:

  ```bash
  python3 src/flowsync/scripts/modify_ipi_csv.py \
    --input_csv data/processed/DS_Kombat_all_packets.csv \
    --model models/Kombat_DL/Kombat_DL.keras \
    --scaler models/Kombat_DL/Kombat_DL_scaler.joblib \
    --remaining_csv models/Kombat_DL/Kombat_DL_packets_N_to_end_scaled.csv \
    --window_size 10 \
    --modify_percent 100 \
    --direction_column Direction \
    --direction_to_modify Downlink \
    --output_mode synthetic_only \
    --output_csv data/generated/Kombat_DL_synthetic_downlink_only.csv
  ```

  **Example: Fortnite Downlink (synthetic-only IPIs)** (saved to `data/generated/`)

  ```bash
  python3 src/flowsync/scripts/modify_ipi_csv.py \
    --input_csv data/processed/DS_Fortnite_all_packets.csv \
    --model models/Fortnite_DL/Fortnite_DL.keras \
    --scaler models/Fortnite_DL/Fortnite_DL_scaler.joblib \
    --remaining_csv models/Fortnite_DL/Fortnite_DL_packets_N_to_end_scaled.csv \
    --window_size 10 \
    --modify_percent 100 \
    --direction_column Direction \
    --direction_to_modify Downlink \
    --output_mode synthetic_only \
    --output_csv data/generated/Fortnite_DL_synthetic_downlink_only.csv
  ```

  **Example: Fortnite Uplink (synthetic-only IPIs)** (saved to `data/generated/`)

  ```bash
  python3 src/flowsync/scripts/modify_ipi_csv.py \
    --input_csv data/processed/DS_Fortnite_all_packets.csv \
    --model models/Fortnite_UL/Fortnite_UL.keras \
    --scaler models/Fortnite_UL/Fortnite_UL_scaler.joblib \
    --remaining_csv models/Fortnite_UL/Fortnite_UL_packets_N_to_end_scaled.csv \
    --window_size 10 \
    --modify_percent 100 \
    --direction_column Direction \
    --direction_to_modify Uplink \
    --output_mode synthetic_only \
    --output_csv data/generated/Fortnite_UL_synthetic_uplink_only.csv
  ```

  #### Evaluating Traffic Preservation with `evaluate_classifier.py`

  To verify that generated synthetic traces preserve cloud gaming timing signatures using the pre-trained Decision Tree and Random Forest classifiers:

  ```bash
  python3 src/flowsync/scripts/evaluate_classifier.py \
    --trace_dir data/generated \
    --original_dir data/extracted_csv \
    --model_dir models/saved/classifiers
  ```

  #### Extracting Model Diagnostics and Reproducing Evaluation Tables

  To re-extract training/test MAE metrics and generate manuscript-ready LaTeX tables (`table.tex`, `table_wide.tex`):

  ```bash
  python3 acm_tomm_metadata/extract_training_metadata.py
  python3 acm_tomm_metadata/make_table.py
  ```

### 5. Using shared models and utilities

- The module `src.flowsync` contains model-building utilities and training helpers.
- Model architecture documentation and diagrams are in `docs/README.md`.
- In notebooks, ensure `src` is in the Python path (`pip install -e .` or `export PYTHONPATH=$PYTHONPATH:$(pwd)/src`).

## Datasets

The `data/` directory contains example CSV files.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
