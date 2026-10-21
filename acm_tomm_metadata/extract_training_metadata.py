"""Extract verifiable training metadata from the saved FlowSync .keras models.

Reads only; never retrains, re-splits, or re-fits a scaler, so the published
80/20 test split and MinMaxScaler are untouched.

What is recoverable from a .keras archive:
  - optimizer `iterations` (gradient steps) and current learning rate, from
    model.weights.h5 -> optimizer/vars/0 and optimizer/vars/1
  - epochs completed, derived as iterations / ceil(n_train / batch_size)
  - loss on any split we still have, by calling model.evaluate()

What is NOT recoverable:
  - the per-epoch loss trajectory (Keras never persists History in .keras)
  - validation loss, because no validation split was held out at training time
    (see README.md)

Usage:  ./flowsync/bin/python acm_tomm_metadata/extract_training_metadata.py
"""

import glob
import io
import json
import math
import os
import zipfile

import h5py
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, 'acm_tomm_metadata')
BATCH_SIZE = 32  # matches batch_size=32 in the prediction notebooks
MAX_EPOCHS = 50  # matches EPOCHS=50 in the prediction notebooks


def read_optimizer_state(keras_path):
    """Return (iterations, learning_rate) from a .keras archive, or (None, None)."""
    with zipfile.ZipFile(keras_path) as z:
        weights = io.BytesIO(z.read('model.weights.h5'))
        meta = json.loads(z.read('metadata.json').decode())
    with h5py.File(weights, 'r') as h:
        if 'optimizer' not in h:
            return None, None, meta
        variables = h['optimizer']['vars']
        iterations = int(np.asarray(variables['0'][()]).item())
        learning_rate = float(np.asarray(variables['1'][()]).item())
    return iterations, learning_rate, meta


def evaluate_losses(keras_path, splits_path):
    """MAE on each split we still have. Inference mode, so dropout is off.

    NOTE: model.evaluate() returns [loss, compile_metrics]. The compiled *loss*
    includes a weight-regularization penalty (~1.64) on top of the MAE, and that
    penalty is a function of the weights alone -- identical on train and test.
    Reporting it would add a large constant to both sides and compress the
    apparent generalization gap toward 1.0. We therefore report the MAE metric,
    and additionally compute MAE directly from predict() to confirm it.
    """
    from tensorflow import keras

    model = keras.models.load_model(keras_path)
    data = np.load(splits_path)
    results = {}
    for split in ('train', 'test'):
        x_key, y_key = 'X_%s' % split, 'y_%s' % split
        if x_key not in data.files:
            continue
        x, y = data[x_key], data[y_key]
        raw = model.evaluate(x, y, batch_size=BATCH_SIZE, verbose=0)
        # Shapes must match exactly; (n,) vs (n,1) would broadcast and inflate.
        pred = model.predict(x, batch_size=BATCH_SIZE, verbose=0)
        assert pred.shape == y.shape, 'shape mismatch %s vs %s' % (pred.shape, y.shape)
        manual_mae = float(np.mean(np.abs(pred - y)))
        results[split] = {
            'mae': float(raw[1]) if isinstance(raw, list) and len(raw) > 1 else manual_mae,
            'regularized_loss': float(raw[0]) if isinstance(raw, list) else float(raw),
            'mae_from_predict': manual_mae,
        }
        # Naive baselines, for context on whether the model learned anything.
        y_train_mean = float(data['y_train'].mean())
        results[split]['baseline_mean_mae'] = float(np.mean(np.abs(y - y_train_mean)))
        results[split]['baseline_persistence_mae'] = float(
            np.mean(np.abs(y.ravel() - x[:, -1, 0])))
    return results, {k: list(data[k].shape) for k in data.files}


def main():
    records = []
    pairs = [(p, p.replace('.keras', '_train_test_splits.npz'))
             for p in sorted(glob.glob(os.path.join(REPO, 'models', '*', '*.keras')))]

    for keras_path, splits_path in pairs:
        name = os.path.basename(keras_path).replace('.keras', '')
        if not os.path.exists(splits_path):
            print('  skip %s: no splits file' % name)
            continue

        iterations, learning_rate, meta = read_optimizer_state(keras_path)
        results, shapes = evaluate_losses(keras_path, splits_path)

        n_train = shapes['X_train'][0]
        steps_per_epoch = math.ceil(n_train / BATCH_SIZE)
        epochs = iterations / steps_per_epoch

        record = {
            'model': name,
            'date_saved': meta.get('date_saved'),
            'keras_version': meta.get('keras_version'),
            'n_train': n_train,
            'n_test': shapes['X_test'][0],
            'has_validation_split': 'X_val' in shapes,
            'batch_size': BATCH_SIZE,
            'steps_per_epoch': steps_per_epoch,
            'optimizer_iterations': iterations,
            'epochs_completed': epochs,
            'ran_to_max_epochs': abs(epochs - MAX_EPOCHS) < 1e-6,
            'final_learning_rate': learning_rate,
            'train_mae': results['train']['mae'],
            'test_mae': results['test']['mae'],
            'train_regularized_loss': results['train']['regularized_loss'],
            'test_regularized_loss': results['test']['regularized_loss'],
            'baseline_mean_mae_test': results['test']['baseline_mean_mae'],
            'baseline_persistence_mae_test': results['test']['baseline_persistence_mae'],
        }
        record['test_over_train_mae'] = record['test_mae'] / record['train_mae']
        record['skill_vs_mean'] = record['baseline_mean_mae_test'] / record['test_mae']
        record['skill_vs_persistence'] = (
            record['baseline_persistence_mae_test'] / record['test_mae'])
        records.append(record)
        print('  %-14s trainMAE=%.4f testMAE=%.4f ratio=%.3f | mean=%.4f persist=%.4f' % (
            name, record['train_mae'], record['test_mae'],
            record['test_over_train_mae'],
            record['baseline_mean_mae_test'],
            record['baseline_persistence_mae_test']))

    out_path = os.path.join(OUT_DIR, 'training_metadata.json')
    with open(out_path, 'w') as fh:
        json.dump(records, fh, indent=2)

    columns = ['model', 'n_train', 'n_test', 'has_validation_split',
               'epochs_completed', 'ran_to_max_epochs', 'final_learning_rate',
               'train_mae', 'test_mae', 'test_over_train_mae',
               'baseline_mean_mae_test', 'baseline_persistence_mae_test',
               'skill_vs_mean', 'skill_vs_persistence']
    csv_path = os.path.join(OUT_DIR, 'training_metadata.csv')
    with open(csv_path, 'w') as fh:
        fh.write(','.join(columns) + '\n')
        for record in records:
            fh.write(','.join(str(record.get(c, '')) for c in columns) + '\n')

    print('\nwrote %s' % out_path)
    print('wrote %s' % csv_path)


if __name__ == '__main__':
    main()
