"""Training diagnostics helpers for FlowSync IPI models.

Provides the standard training callbacks (early stopping, learning-rate
reduction, checkpointing) and a helper to report the epoch selected by
early stopping. Together with ``plot_train_and_val_loss`` in
``plotting.py``, these let the prediction notebooks show train/validation
loss curves across epochs and document the selected epoch -- e.g. to
demonstrate that validation loss tracks training loss without diverging
(no severe overfitting).

The Keras import is deferred into ``make_training_callbacks`` so this
module can be imported -- and ``report_early_stopping_epoch`` used or
tested -- without TensorFlow installed.
"""

import os

import numpy as np


def make_training_callbacks(checkpoint_path,
                            monitor='val_loss',
                            mode='min',
                            patience_early_stopping=10,
                            patience_reduce_lr=5,
                            factor_reduction=0.1,
                            min_lr=1e-6,
                            verbose=1):
    """Build the standard callbacks used to train FlowSync IPI models.

    Returns ``(callbacks, early_stopping, checkpoint)`` where ``callbacks``
    is the list to pass to ``model.fit(callbacks=...)``.

    ``early_stopping`` uses ``restore_best_weights=True``, so after training
    the model already reflects the selected (best) epoch. ``checkpoint`` is
    returned separately because existing notebook cells reference its
    ``.best`` attribute (e.g. ``cp10.best``).
    """
    from tensorflow.keras.callbacks import (
        EarlyStopping,
        ModelCheckpoint,
        ReduceLROnPlateau,
    )

    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    early_stopping = EarlyStopping(
        monitor=monitor,
        mode=mode,
        patience=patience_early_stopping,
        restore_best_weights=True,
        verbose=verbose,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor=monitor,
        mode=mode,
        factor=factor_reduction,
        patience=patience_reduce_lr,
        min_lr=min_lr,
        verbose=verbose,
    )
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor=monitor,
        mode=mode,
        save_best_only=True,
        verbose=verbose,
    )

    callbacks = [early_stopping, reduce_lr, checkpoint]
    return callbacks, early_stopping, checkpoint


def report_early_stopping_epoch(history, early_stopping=None,
                                monitor='val_loss', verbose=True):
    """Report the epoch selected by early stopping (best ``monitor`` value).

    Returns the 1-indexed best epoch. When ``verbose`` is True it also prints
    a short summary: the selected epoch, the best monitored value, the
    training loss at that epoch, how many epochs actually ran, and whether
    early stopping triggered before the maximum number of epochs.
    """
    hist = history.history if hasattr(history, 'history') else history

    if monitor not in hist:
        raise KeyError(
            "'{0}' not found in history. Available keys: {1}. "
            "Did you pass validation_data to model.fit()?".format(
                monitor, list(hist.keys())
            )
        )

    monitored = np.asarray(hist[monitor], dtype=float)
    best_idx = int(np.argmin(monitored))
    best_epoch = best_idx + 1
    total_epochs = int(len(monitored))
    best_value = float(monitored[best_idx])

    train_loss_at_best = None
    if 'loss' in hist:
        train_loss_at_best = float(np.asarray(hist['loss'], dtype=float)[best_idx])

    stopped_early = None
    if early_stopping is not None:
        stopped_epoch = int(getattr(early_stopping, 'stopped_epoch', 0) or 0)
        stopped_early = stopped_epoch > 0

    if verbose:
        print('Early stopping / model selection summary')
        print('-' * 44)
        print('  Selected epoch (best {0}): {1} of {2} run'.format(
            monitor, best_epoch, total_epochs))
        print('  Best {0}: {1:.6f}'.format(monitor, best_value))
        if train_loss_at_best is not None:
            print('  Training loss at selected epoch: {0:.6f}'.format(
                train_loss_at_best))
        if stopped_early is not None:
            if stopped_early:
                print('  Early stopping triggered (halted before max epochs).')
            else:
                print('  Early stopping did NOT trigger (ran to max epochs).')

    return best_epoch
