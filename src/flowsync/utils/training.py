import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.callbacks import Callback, ModelCheckpoint
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

# Import from other modules in the package
from src.flowsync.models.architectures import (
    create_model1, create_model2, create_model3, create_model4, 
    create_model5, create_model6, create_model7, create_model8, 
    create_model9, create_model10
)
from src.flowsync.utils.data_processing import df_to_X_y_version2
from src.flowsync.utils.plotting import plot_train_and_val_loss, plot_predictions1

# Custom callback function to print a single bar for the whole training, instead of relying on a per-epoch basis bar
class ProgressBarCallback(Callback):
    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']  # Total number of epochs
        self.progress_bar = tqdm(total=self.epochs, desc='Training Progress', unit='epoch')
        
    def on_epoch_end(self, epoch, logs=None):
        # Update the progress bar after each epoch
        self.progress_bar.update(1)

    def on_train_end(self, logs=None):
        # Close the progress bar at the end of training
        self.progress_bar.close()


def train_and_evaluate_models(original_data, train_size, val_size, models, output_size, num_features, 
                              window_sizes, num_neurons_list, batch_sizes, epochs, loss_func, learning_rates=0.0001):
    
    best_model = None
    #best_val_loss = float('inf')
    best_val_loss_difference = float('inf')
    best_params = tuple()
    best_history = 0
    history_results = []

    for model_id in models:
        for window_size in window_sizes:
            X, y = df_to_X_y_version2(original_data, window_size=window_size, target='rts')
            X_train, y_train = X[:train_size], y[:train_size]
            X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
            X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

            for num_neurons in num_neurons_list:
                for batch_size in batch_sizes:
                    for learning_rate in learning_rates:
                        print(f"Training model with: model={model_id}, window_size={window_size}, neurons={num_neurons}, batch_size={batch_size}, learning_rate={learning_rate}")
                        
                        # Create the model
                        if model_id == 1:
                            model, cp = create_model1(output_size=output_size, window_size=window_size, num_features=num_features,
                                                      num_neurons=num_neurons, loss_func=loss_func, learning_rate=learning_rate)
                        elif model_id == 2:
                            model, cp = create_model2(output_size=output_size, window_size=window_size, num_features=num_features,
                                                      num_neurons=num_neurons, loss_func=loss_func, learning_rate=learning_rate)
                        elif model_id == 3:
                            model, cp = create_model3(output_size=output_size, window_size=window_size, num_features=num_features, 
                                                      num_neurons=num_neurons, loss_func=loss_func, learning_rate=learning_rate)
                        elif model_id == 4:
                            model, cp = create_model4(output_size=output_size, window_size=window_size, num_features=num_features,
                                                      num_neurons=num_neurons, loss_func=loss_func, learning_rate=learning_rate)
                        elif model_id == 5:
                            model, cp = create_model5(output_size=output_size, window_size=window_size, num_features=num_features,
                                                      num_neurons=num_neurons, loss_func=loss_func, learning_rate=learning_rate)
                        elif model_id == 6:
                            model, cp = create_model6(output_size=output_size, window_size=window_size, num_features=num_features,
                                                      num_neurons=num_neurons, loss_func=loss_func, learning_rate=learning_rate)
                        elif model_id == 7:
                            model, cp = create_model7(output_size=output_size, window_size=window_size, num_features=num_features, 
                                                      loss_func=loss_func)
                        elif model_id == 8:
                            model, cp = create_model8(output_size=output_size, window_size=window_size, num_features=num_features,
                                                      num_neurons=num_neurons, loss_func=loss_func)
                        elif model_id == 9:
                            model, cp = create_model9(window_size=window_size, num_features=num_features, output_size=output_size, 
                                                      learning_rate=learning_rate)
                        elif model_id == 10:
                            model, cp = create_model10(output_size=output_size, window_size=window_size, num_features=num_features,
                                                       num_neurons=num_neurons, loss_func=loss_func, learning_rate=learning_rate)
                        else:
                            raise ValueError("Invalid model number!")

                        # Train the model with validation data
                        history = model.fit(
                            X_train, y_train,
                            epochs=epochs,
                            validation_data=(X_val, y_val),
                            batch_size=batch_size,
                            callbacks=[cp, ProgressBarCallback()],
                            verbose=0  # Suppress output
                        )

                        # Record the validation loss
                        first_val_loss = history.history['val_loss'][0]  # Get the validation loss from the first epoch
                        last_val_loss = history.history['val_loss'][-1]  # Get the validation loss from the last epoch

                        # Calculate the difference in validation loss
                        val_loss_difference = first_val_loss - last_val_loss

                        # Append model results including the loss difference
                        history_results.append((model_id, window_size, num_neurons, batch_size, learning_rate, val_loss_difference))

                        # Record the validation loss
                        #val_loss = min(history.history['val_loss'])  # Get the minimum validation loss
                        #history_results.append((model_id, window_size, num_neurons, batch_size, learning_rate, val_loss))

                        # Save the best model
                        #if val_loss < best_val_loss:
                        # Save the best model based on the difference in validation loss
                        if val_loss_difference < best_val_loss_difference:
                            best_val_loss_difference = val_loss_difference
                            #best_val_loss = val_loss
                            best_model = model
                            best_history = history
                            #best_params = (window_size, num_neurons, batch_size, learning_rate, val_loss)
                            best_params = (window_size, num_neurons, batch_size, learning_rate, best_val_loss_difference)
                            plot_train_and_val_loss(best_history)
                            best_test_predictions = plot_predictions1(best_model, X_test, y_test, start=0, end=1000, step=10)
                            print("Original correspoding cumulative sum")
                            print("rows (y_test):", len(y_test))
                            print(np.cumsum(y_test)[-1])
                            print()
                            print("Generated/Predicted cumulative sum")
                            print("rows:", len(best_test_predictions))
                            print(np.cumsum(best_test_predictions).iloc[-1])
                            print(f"Best hyperparameters so far: model={model_id} window_size={window_size}, num_neurons={num_neurons}, batch_size={batch_size}, learning_rate={learning_rate}")

    return best_model, best_params, best_history, history_results, X, y


# Blocked K-folded Cross Validation
# Function for applying Blocked K-Fold Cross-Validation to Time Series
def block_kfold_time_series(X, y, n_splits=5, input_shape=(5, 64), num_filters=64, kernel_size=3, dilation_rates=[1, 2, 4, 8], output_size=1, batch_size=32, epochs=50):
    # Initialize TimeSeriesSplit for blocked k-fold
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    fold_no = 1
    val_losses = []
    
    # Iterate through each split
    for train_index, val_index in tscv.split(X):
        print(f"Fold {fold_no}")
        
        # Split the dataset into training and validation based on the current fold
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Create model instance for the current fold
        model, _ = create_model9(input_shape=input_shape, num_filters=num_filters, kernel_size=kernel_size, dilation_rates=dilation_rates, output_size=output_size)

        # Define a checkpoint for saving the best model during training
        cp = ModelCheckpoint(f'models/model9_fold{fold_no}.keras', save_best_only=True, monitor='val_loss', mode='min')

        # Train the model on the current fold
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[cp],
            verbose=1
        )
        
        # Evaluate validation loss for the current fold
        val_loss = np.min(history.history['val_loss'])
        print(f"Validation loss for fold {fold_no}: {val_loss}")
        val_losses.append(val_loss)
        
        fold_no += 1
    
    # Calculate the average validation loss across all folds
    avg_val_loss = np.mean(val_losses)
    print(f"Average Validation Loss across all folds: {avg_val_loss}")
    
    return val_losses


def single_target_xgboost(df, target_col, threshold=0.02, test_size=0.2, random_state=42):
    """
    Train an XGBoost model on the provided dataframe and return selected features
    based on feature importance threshold.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing features and target column.
        target_col (str): Name of the target column in df.
        threshold (float): Feature importance threshold for selection. Default is 0.02.
        test_size (float): Proportion of the data to use as test set. Default is 0.2.
        random_state (int): Random seed for reproducibility. Default is 42.

    Returns:
        selected_features_xgb (list): List of selected features based on the importance threshold.
    """
    # Split data into features (X) and target (y)
    X = df[df.columns.difference([target_col])]  # Features
    y = df[target_col]  # Target variable

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Train XGBoost model
    model = xgb.XGBRegressor(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)

    # Get feature importances
    importances = model.feature_importances_
    features = X.columns

    # Combine features and their importances, then sort in descending order
    feature_importance_pairs = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)

    # Unzip sorted pairs into separate lists for plotting
    sorted_features, sorted_importances = zip(*feature_importance_pairs)

    # Plot feature importance
    plt.figure(figsize=(10, 5))
    plt.barh(sorted_features, sorted_importances, color='royalblue')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Name')
    plt.title('XGBoost Feature Importance for Predicting {}'.format(target_col))
    plt.gca().invert_yaxis()  # Invert y-axis to show highest importance on top
    plt.show()

    # Select features based on a threshold
    selected_features_xgb = [feature for feature, importance in feature_importance_pairs if importance >= threshold]

    # Print the selected features in descending order of importance
    print("Selected features based on importance (in descending order):", selected_features_xgb)

    # Return the selected features
    return selected_features_xgb


def multi_target_xgboost(df, target_cols, threshold=0.02, test_size=0.2, random_state=42):
    """
    Train an XGBoost model on the provided dataframe and return selected features
    based on feature importance threshold for each target column in a multi-target scenario.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing features and target columns.
        target_cols (list): List of target column names in df for multi-target regression.
        threshold (float): Feature importance threshold for selection. Default is 0.02.
        test_size (float): Proportion of the data to use as test set. Default is 0.2.
        random_state (int): Random seed for reproducibility. Default is 42.

    Returns:
        selected_features_xgb (dict): Dictionary of selected features for each target column 
        based on the importance threshold.
    """
    selected_features_xgb = {}

    # Iterate over each target column for multi-target regression
    for target_col in target_cols:
        print(f"Training XGBoost for target column: {target_col}")

        # Split data into features (X) and target (y)
        X = df[df.columns.difference(target_cols)]  # Features (excluding all target columns)
        y = df[target_col]  # Target variable (current target column)

        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Train XGBoost model
        model = xgb.XGBRegressor(n_estimators=100, random_state=random_state)
        model.fit(X_train, y_train)

        # Get feature importances
        importances = model.feature_importances_
        features = X.columns

        # Combine features and their importances, then sort in descending order
        feature_importance_pairs = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)

        # Unzip sorted pairs into separate lists for plotting
        sorted_features, sorted_importances = zip(*feature_importance_pairs)

        # Plot feature importance for this target column
        plt.figure(figsize=(10, 5))
        plt.barh(sorted_features, sorted_importances, color='royalblue')
        plt.xlabel('Feature Importance')
        plt.ylabel('Feature Name')
        plt.title(f'XGBoost Feature Importance for Predicting {target_col}')
        plt.gca().invert_yaxis()  # Invert y-axis to show highest importance on top
        plt.show()

        # Select features based on the threshold
        selected_features = [feature for feature, importance in feature_importance_pairs if importance >= threshold]

        # Store selected features for this target column
        selected_features_xgb[target_col] = selected_features

        # Print the selected features in descending order of importance
        print("Selected features based on importance (in descending order):", selected_features_xgb)

    # Return the dictionary of selected features for all target columns
    return selected_features_xgb


def single_target_rfr(df, target_col, threshold=0.02, test_size=0.2, random_state=42, scale=False):
    """
    Train a RandomForestRegressor model on the provided dataframe and return selected features
    based on feature importance threshold.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing features and target column.
        target_col (str): Name of the target column in df.
        threshold (float): Feature importance threshold for selection. Default is 0.02.
        test_size (float): Proportion of the data to use as test set. Default is 0.2.
        random_state (int): Random seed for reproducibility. Default is 42.
        scale (bool): Whether to scale features using MinMaxScaler. Default is False.

    Returns:
        selected_features_rfr (list): List of selected features based on the importance threshold.
    """
    # Split data into features (X) and target (y)
    X_rfr = df[df.columns.difference([target_col])]  # Features
    y_rfr = df[target_col]  # Target variable

    # Optionally scale features
    if scale:
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_rfr = scaler.fit_transform(X_rfr)

    # Split into training and test sets
    X_train_rfr, X_test_rfr, y_train_rfr, y_test_rfr = train_test_split(X_rfr, y_rfr, test_size=test_size, random_state=random_state)

    # Train RandomForestRegressor
    rfr = RandomForestRegressor(n_estimators=100, random_state=random_state)
    rfr.fit(X_train_rfr, y_train_rfr)

    # Get feature importances
    importances = rfr.feature_importances_
    features = df[df.columns.difference([target_col])].columns

    # Plot feature importance
    plt.figure(figsize=(10, 5))
    plt.barh(features, importances, color='royalblue')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Name')
    plt.title(f'Random Forest Feature Importance for Predicting {target_col}')
    plt.show()

    # Create a list of tuples containing (feature, importance)
    feature_importance_list = [(feature, importance) for feature, importance in zip(features, importances) if importance >= threshold]

    # Sort the list in descending order by importance
    sorted_features = sorted(feature_importance_list, key=lambda x: x[1], reverse=True)

    # Extract the feature names from the sorted list
    selected_features_rfr = [feature for feature, importance in sorted_features]

    # Print the selected features in descending order of importance
    print("Selected features based on importance (in descending order):", selected_features_rfr)

    # Optionally, print each feature with its importance
    for feature, importance in sorted_features:
        print(f'{feature}: {importance}')

    # Return the selected features
    return selected_features_rfr


def multi_target_random_forest(df, target_cols, threshold=0.02, test_size=0.2, random_state=42, scale=False):
    """
    Train a MultiOutputRegressor using RandomForestRegressor on a DataFrame with arbitrary target columns
    and return selected important features based on feature importance threshold for the first target.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing features and target columns.
        target_cols (list of str): List of target column names in df.
        threshold (float): Feature importance threshold for selection. Default is 0.02.
        test_size (float): Proportion of the data to use as test set. Default is 0.2.
        random_state (int): Random seed for reproducibility. Default is 42.
        scale (bool): Whether to scale features using MinMaxScaler. Default is False.

    Returns:
        selected_features (list): List of selected features based on the importance threshold for the first target.
    """
    # Split data into features (X) and targets (y)
    X_rfr = df[df.columns.difference(target_cols)]  # Features
    y_rfr = df[target_cols]  # Target variables

    # Optionally scale features
    if scale:
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_rfr = scaler.fit_transform(X_rfr)

    # Split into training and test sets
    X_train_rfr, X_test_rfr, y_train_rfr, y_test_rfr = train_test_split(X_rfr, y_rfr, test_size=test_size, random_state=random_state)

    # Train MultiOutputRegressor with RandomForestRegressor
    rfr = RandomForestRegressor(n_estimators=150, random_state=random_state)
    multi_target_rfr = MultiOutputRegressor(rfr)

    # Fit the model
    multi_target_rfr.fit(X_train_rfr, y_train_rfr)

    # Predict both targets
    predictions = multi_target_rfr.predict(X_test_rfr)

    # Evaluate predictions for each target feature
    for i, target in enumerate(target_cols):
        plt.figure(figsize=(10, 5))
        
        # Scatter plot for actual vs predicted
        plt.scatter(y_test_rfr[target], predictions[:, i], alpha=0.5, label='Predicted Values')
        
        # Diagonal line for perfect predictions
        plt.plot([y_test_rfr[target].min(), y_test_rfr[target].max()], 
                 [y_test_rfr[target].min(), y_test_rfr[target].max()], 
                 'r--', label='Perfect Prediction')  # Add label for the line
        
        plt.xlabel(f'Actual {target}')
        plt.ylabel(f'Predicted {target}')
        plt.title(f'Actual vs Predicted for {target}')
        plt.legend()  # Add legend to the plot
        plt.show()

    # Get feature importances for the first target model (RandomForestRegressor for the first target)
    importances = multi_target_rfr.estimators_[0].feature_importances_
    features = X_rfr.columns

    # Plot feature importance
    plt.figure(figsize=(10, 5))
    plt.barh(features, importances, color='royalblue')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Name')
    plt.title(f'Random Forest Feature Importance for Predicting {target_cols[0]}')
    plt.show()

    # Print and save feature importance values for the first target
    selected_features = []
    
    # Create a list of tuples containing (feature, importance)
    feature_importance_list = [(feature, importance) for feature, importance in zip(features, importances) if importance >= threshold]

    # Sort the list in descending order by importance
    sorted_features_rfr = sorted(feature_importance_list, key=lambda x: x[1], reverse=True)

    # Extract the feature names from the sorted list
    selected_features = [feature for feature, importance in sorted_features_rfr]

    # Print the selected features in descending order of importance
    print(f"Selected features for {target_cols[0]} based on importance (in descending order):", selected_features)

    # Optionally, print each feature with its importance
    for feature, importance in sorted_features_rfr:
        print(f'{feature}: {importance}')

    # Return the selected features for the first target
    return selected_features
