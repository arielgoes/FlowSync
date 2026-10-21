"""Recover per-epoch training curves from saved Jupyter cell outputs.

Keras does not persist the History object into a .keras file, but fit(verbose=1)
writes per-epoch progress to stdout, and Jupyter stores stdout in the notebook
JSON. Where a notebook was saved after training, the curve is recoverable
without retraining.

Only training loss and training MAE are recoverable here: these runs used no
validation split, so no val_loss was ever computed.

Usage:  ./flowsync/bin/python acm_tomm_metadata/extract_curves.py
"""

import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

SOURCES = {
    'UL': '_backup/FlowSync_legacy/Cloud Gaming/DS_CG_IPI/UL/predict_IPI_UL.ipynb',
    'DL': '_backup/FlowSync_legacy/Cloud Gaming/DS_CG_IPI/DL/predict_IPI_DL.ipynb',
}

# Final line of each epoch: "25/25 ... - loss: X - mae: Y"
EPOCH_LINE = re.compile(r'(\d+)/(\d+).*?- loss: ([0-9.]+) - mae: ([0-9.]+)')


def extract(path):
    with open(path) as fh:
        nb = json.load(fh)
    epochs = []
    for cell in nb['cells']:
        for out in cell.get('outputs', []):
            text = ''.join(out.get('text', [])) if 'text' in out else ''
            if 'step - loss:' not in text:
                continue
            for line in text.split('\n'):
                m = EPOCH_LINE.search(line)
                if not m:
                    continue
                done, total = int(m.group(1)), int(m.group(2))
                if done != total:
                    continue  # mid-epoch progress line, not the epoch summary
                epochs.append({'loss': float(m.group(3)), 'mae': float(m.group(4))})
    return epochs


def main():
    out = {}
    for direction, rel in SOURCES.items():
        path = os.path.join(REPO, rel)
        if not os.path.exists(path):
            print('missing: %s' % rel)
            continue
        epochs = extract(path)
        out[direction] = epochs
        if not epochs:
            continue
        print('%s: %d epochs recovered' % (direction, len(epochs)))
        print('    loss %.4f -> %.4f   mae %.4f -> %.4f' % (
            epochs[0]['loss'], epochs[-1]['loss'],
            epochs[0]['mae'], epochs[-1]['mae']))
        tail = [e['mae'] for e in epochs[-10:]]
        print('    last 10 epochs mae: min %.4f max %.4f spread %.4f' % (
            min(tail), max(tail), max(tail) - min(tail)))

    path = os.path.join(HERE, 'training_curves.json')
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=2)
    print('\nwrote %s' % path)
    print('NOTE: training loss only -- no validation split existed for these runs.')


if __name__ == '__main__':
    main()
