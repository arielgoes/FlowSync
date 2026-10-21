import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# output: single features (i.e., 'rts')
def plot_predictions1(model, X, y, start=0, end=100, step=1):
    predictions = model.predict(X).flatten()
    y = y.flatten()  # Flatten y if it's not 1D
    df = pd.DataFrame(data={'Predictions':predictions, 'Actuals':y})
    plt.plot(df['Predictions'][start:end:step], label='Predictions', color='orange')
    plt.plot(df['Actuals'][start:end:step], label='Actuals', color='blue', linestyle='--')

    
    # Add legend to show labels
    plt.legend()

    # Show the plot
    plt.show()
    return df


# outputs: Target 1: position [0], Target 2: position [1], ...)
def plot_predictions2(model, X, y, start=0, end=100, step=1):
    predictions = model.predict(X)
    target1_preds, target2_preds = predictions[:, 0], predictions[:, 1]
    target1_actuals, target2_actuals = y[:, 0], y[:, 1]
    df = pd.DataFrame(data={'Target1 Predictions':target1_preds,
                            'Target1 Actuals':target1_actuals,
                            'Target2 Predictions':target2_preds,
                            'Target2 Actuals':target2_actuals})
    plt.plot(df['Target1 Predictions'][start:end:step], label='Target1 Predictions', color='orange')
    plt.plot(df['Target1 Actuals'][start:end:step], label='Target1 Actuals', color='blue', linestyle='--')
    plt.plot(df['Target2 Predictions'][start:end:step], label='Target2 Predictions', color='red')
    plt.plot(df['Target2 Actuals'][start:end:step], label='Target2 Actuals', color='green', linestyle='--')

    # Add legend to show labels
    plt.legend()

    # Show the plot
    plt.show()
    return df


# outputs: Target 1: position [0], Target 2: position [1], ...
def plot_predictions3(model, X, y, start=0, end=100, step=1):
    # Make predictions
    predictions = model.predict(X)
    print("predictions shape", predictions.shape)
    
    # Extract predictions and actuals
    target1_preds, target2_preds = predictions[:, 0], predictions[:, 1]
    target_actual, target2_actuals = y[:, 0], y[:, 1]
    
    # Create a DataFrame for better visualization
    df = pd.DataFrame(data={
        'Target1 Predictions': target1_preds,
        'Target1 Actuals': target_actual,
        'Target2 Predictions': target2_preds,
        'Target2 Actuals': target2_actuals
    })
    
    # Plot Target1 Predictions vs Actuals
    plt.figure(figsize=(10, 5))
    plt.plot(df['Target1 Predictions'][start:end:step], label='Target1 Predictions', color='orange')
    plt.plot(df['Target1 Actuals'][start:end:step], label='Target1 Actuals', color='blue', linestyle='--')
    
    # Add title and labels
    plt.title('Target1 Predictions vs Actuals')
    plt.xlabel('Sample Index')
    plt.ylabel('Target1 Value')
    plt.legend()
    plt.grid()
    
    # Show the plot
    plt.show()
    
    # Plot Target2 Predictions vs Actuals
    plt.figure(figsize=(10, 5))
    plt.plot(df['Target2 Predictions'][start:end], label='Target2 Predictions', color='red')
    plt.plot(df['Target2 Actuals'][start:end], label='Target2 Actuals', color='green', linestyle='--')
    
    # Add title and labels
    plt.title('Target2 Predictions vs Actuals')
    plt.xlabel('Sample Index')
    plt.ylabel('Target2 Value')
    plt.legend()
    plt.grid()
    
    # Show the plot
    plt.show()

    return df


def plot_predictions_generic(model, X, y, start=0, end=100, step=1):
    """
    Plots predictions and actuals for any number of output features.

    Args:
    - model: The trained model used for predictions.
    - X: Input features for predictions.
    - y: Ground truth values.
    - start: Start index for plotting.
    - end: End index for plotting.
    - step: Step size for plotting.

    Returns:
    - df: DataFrame containing predictions and actuals for all output features.
    """
    # Make predictions
    predictions = model.predict(X)
    print("Predictions shape:", predictions.shape)
    print("Ground truth shape:", y.shape)

    # Create a DataFrame for visualization
    num_outputs = predictions.shape[1]
    data = {}
    
    for i in range(num_outputs):
        data[f'Target{i+1} Predictions'] = predictions[:, i]
        data[f'Target{i+1} Actuals'] = y[:, i]

    df = pd.DataFrame(data)

    # Plot predictions vs actuals for each output feature
    for i in range(num_outputs):
        plt.figure(figsize=(10, 5))
        plt.plot(df[f'Target{i+1} Predictions'][start:end:step], label=f'Target{i+1} Predictions', color='orange')
        plt.plot(df[f'Target{i+1} Actuals'][start:end:step], label=f'Target{i+1} Actuals', color='blue', linestyle='--')
        
        # Add title and labels
        plt.title(f'Target{i+1} Predictions vs Actuals')
        plt.xlabel('Sample Index')
        plt.ylabel(f'Target{i+1} Value')
        plt.legend()
        plt.grid()
        
        # Show the plot
        plt.show()

    return df



# output: single features (i.e., 'rts')
def plot_only_predictions(model, X, y, start=0, end=100, step=1):
    predictions = model.predict(X).flatten()
    df = pd.DataFrame(data={'Predictions':predictions, 'Actuals':y})
    plt.plot(df['Predictions'][start:end:step], label='Predictions', color='orange')
    plt.plot(df['Actuals'][start:end:step], label='Actuals', color='blue')
    
    # Add legend to show labels
    plt.legend()

    # Show the plot
    plt.show()
    return df


def plot_train_and_val_loss(history, best_epoch=None, save_path=None,
                            title='Model Loss', mark_best=True, show=True):
    """Plot training (and validation) loss across epochs.

    Backward compatible: ``plot_train_and_val_loss(history)`` still works.

    Parameters
    ----------
    history : keras ``History`` (or any object/dict exposing a ``history``
        mapping with ``'loss'`` and optionally ``'val_loss'``).
    best_epoch : int, optional
        1-indexed epoch to mark as the one selected by early stopping. If
        None and validation loss is present, it defaults to
        ``argmin(val_loss) + 1``.
    save_path : str, optional
        If given, the figure is saved there (parent directories are created).
        For non-PNG paths a ``.png`` copy is also written alongside, so the
        notebook can keep a vector PDF for the paper and a PNG for quick view.
    title : str
        Plot title (use it to encode game and direction, e.g.
        "Training vs Validation Loss (Kombat, Downlink)").
    mark_best : bool
        Draw a marker and annotation at the selected epoch.
    show : bool
        Call ``plt.show()`` so the figure renders inline in the notebook.

    Returns
    -------
    matplotlib.figure.Figure
    """
    hist = history.history if hasattr(history, 'history') else history

    train_loss = hist['loss']
    epochs = range(1, len(train_loss) + 1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, train_loss, label='Training Loss')

    if 'val_loss' in hist:
        val_loss = hist['val_loss']
        ax.plot(epochs, val_loss, label='Validation Loss')

        if mark_best:
            if best_epoch is None:
                best_epoch = int(np.argmin(val_loss)) + 1
            best_val = val_loss[best_epoch - 1]
            ax.axvline(best_epoch, color='gray', linestyle='--', linewidth=1)
            ax.scatter([best_epoch], [best_val], color='red', zorder=5,
                       label='Selected epoch ({0})'.format(best_epoch))
            ax.annotate(
                'Selected epoch (early stopping): {0}'.format(best_epoch),
                xy=(best_epoch, best_val),
                xytext=(0.97, 0.95), textcoords='axes fraction',
                ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.85),
            )

    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        base, ext = os.path.splitext(save_path)
        if ext.lower() != '.png':
            fig.savefig(base + '.png', dpi=300, bbox_inches='tight')
        print('Saved loss curve to: {0}'.format(save_path))

    if show:
        plt.show()

    return fig
