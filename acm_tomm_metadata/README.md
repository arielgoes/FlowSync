# Model Generalization & Training Diagnostics

This directory contains scripts and metadata for extracting evaluation metrics, reproducing the generalization diagnostic table, and inspecting saved IPI prediction models.

## Overview

The IPI prediction component models inter-packet interval (IPI) dynamics using a hybrid 1D-CNN-GRU architecture. For each game and traffic direction, models were trained on 799 non-overlapping windows of length 10 and evaluated on 199 held-out test windows.

Evaluation metrics are computed directly from the saved `.keras` models and their corresponding held-out `.npz` test partitions without retraining.

## Directory Contents

| File | Description |
|---|---|
| `extract_training_metadata.py` | Standalone script that loads saved `.keras` models and `.npz` splits to compute train/test MAE and optimizer iterations. |
| `make_table.py` | Formats extracted metadata into LaTeX tables (`table.tex`, `table_wide.tex`). |
| `training_metadata.json` | Complete machine-readable record of model configurations, iterations, and evaluation errors. |
| `training_metadata.csv` | Flat CSV representation of model evaluation diagnostics. |
| `table.tex` | LaTeX table reporting per-model train and held-out test MAE and error gap. |
| `table_wide.tex` | Two-column formatted LaTeX table. |

## Reproducing Evaluation Diagnostics

To re-extract model diagnostics and generate the evaluation tables:

```bash
# Extract metrics from saved models into JSON and CSV
python acm_tomm_metadata/extract_training_metadata.py

# Generate LaTeX tables
python acm_tomm_metadata/make_table.py
```

## Summary of Results

MAE evaluated on the $[-1, 1]$ min-max scaled IPI series (799 train windows, 199 test windows):

| Trace | Direction | Train MAE | Test MAE | Error Increase |
|---|---|---|---|---|
| Mortal Kombat | Downlink (DL) | 0.0767 | 0.0790 | +3% |
| Forza Horizon 5 | Downlink (DL) | 0.0897 | 0.1030 | +15% |
| Fortnite | Downlink (DL) | 0.1197 | 0.1555 | +30% |
| Fortnite | Uplink (UL) | 0.0953 | 0.1174 | +23% |
| Forza Horizon 5 | Uplink (UL) | 0.1416 | 0.2122 | +50% |
| Mortal Kombat | Uplink (UL) | 0.1337 | 0.2427 | +82% |
