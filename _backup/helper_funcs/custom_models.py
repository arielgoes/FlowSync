import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import RMSprop, AdamW, Adam
from tensorflow.keras.metrics import *
from keras.callbacks import ModelCheckpoint
from tcn import TCN
from tensorflow.keras.models import Model
import tensorflow as tf  # Add this import for TensorFlow
from sklearn.multioutput import MultiOutputRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import xgboost as xgb
from sklearn.feature_selection import RFE
from tqdm import tqdm
from keras.callbacks import Callback
from tensorflow.keras.losses import Huber, KLDivergence
import re
import joblib


eth_prefixes = {'eth_dhost_':48,
                'eth_shost_':48,
                'eth_ethertype_':16}

ipv4_prefixes = {'ipv4_ver_':4,
                    'ipv4_hl_':4,
                    'ipv4_tos_':8,
                    'ipv4_tl_':16,
                    'ipv4_id_':16,
                    'ipv4_rbit_':1, 'ipv4_dfbit_':1, 'ipv4_mfbit_':1,
                    'ipv4_foff_':13,
                    'ipv4_ttl_':8,
                    'ipv4_proto_':8,
                    'ipv4_cksum_':16,
                    'ipv4_src_':32,
                    'ipv4_dst_':32,
                    'ipv4_opt_':320}

ipv6_prefixes = {'ipv6_ver_':4,
                    'ipv6_tc_':8,
                    'ipv6_fl_':20,
                    'ipv6_len_':16,
                    'ipv6_nh_':8,
                    'ipv6_hl_':8,
                    'ipv6_src_':128,
                    'ipv6_dst_':128}

tcp_prefixes = {'tcp_sprt_':16,
                'tcp_dprt_':16,
                'tcp_seq_':32,
                'tcp_ackn_':32,
                'tcp_doff_':4,
                'tcp_res_':3, 'tcp_ns_':1, 'tcp_cwr_':1,
                'tcp_ece_':1, 'tcp_urg_':1,
                'tcp_ackf_':1, 'tcp_psh_':1,
                'tcp_rst_':1, 'tcp_syn_':1,
                'tcp_fin_':1,
                'tcp_wsize_':16,
                'tcp_cksum_':16,
                'tcp_urp_':16,
                'tcp_opt_':320}

udp_prefixes = {'udp_sport_':16,
                'udp_dport_':16,
                'udp_len_':16,
                'udp_cksum_':16}

icmp_prefixes = {'icmp_type_':8,
                    'icmp_code_':8,
                    'icmp_cksum_':8,
                    'icmp_roh_':32}

payload = {} # if "payload_<#>" columns exist, this dictionary will be dynamically populated.



# Model 1 - Single-layer LSTM
def create_model1(output_size, window_size, num_features, num_neurons, print_summary=False, loss_func=Huber(), learning_rate=0.001):
    model = Sequential()
    model.add(LSTM(units=num_neurons, activation='relu', input_shape=(window_size, num_features)))  # Adjust input shape as needed
    model.add(Dense(8, activation='relu'))
    model.add(Dense(output_size))  # Single output for next timestamp
    #
    if print_summary:
        model.summary()
    
    # save/compile model
    cp = ModelCheckpoint('models/model1/model.keras', save_best_only=True)
    model.compile(loss=loss_func, optimizer=Adam(learning_rate=learning_rate), metrics=[RootMeanSquaredError()])

    return model, cp

# Model 2 (Conv1D) - Single Conv Layer
def create_model2(output_size, window_size, num_features, num_neurons, print_summary=False, loss_func=Huber(), learning_rate=0.001):
    model = Sequential()
    model.add(InputLayer((window_size, num_features)))
    model.add(Conv1D(64, kernel_size=2, activation='relu'))
    model.add(Flatten())
    model.add(Dense(num_neurons, 'relu'))
    model.add(Dense(output_size))
    #
    if print_summary:
        model.summary()
    
    # save/compile model
    cp = ModelCheckpoint('models/model2/model.keras', save_best_only=True)
    model.compile(loss=loss_func, optimizer=Adam(learning_rate=learning_rate), metrics=[RootMeanSquaredError()])

    return model, cp

#model, cp = create_model3(output_size, window_size, num_features, gru_units, dense_units, learning_rate)

# Model 3 (GRU)
def create_model3(output_size, window_size, num_features, num_neurons, print_summary=False, loss_func=Huber(), 
                  learning_rate=0.001, activation='linear', dropout=0, use_batch_norm=False):
    model = Sequential()
    model.add(InputLayer((window_size, num_features)))
    model.add(GRU(num_neurons, return_sequences=True))  # Optional: if stacking layers later

    if use_batch_norm:
        model.add(BatchNormalization())
    
    if dropout > 0:
        model.add(Dropout(dropout))

    model.add(GRU(num_neurons))  # Adding a second GRU layer
    if use_batch_norm:
        model.add(BatchNormalization())
    
    if dropout > 0:
        model.add(Dropout(dropout))

    model.add(Dense(output_size, activation=activation))

    if print_summary:
        model.summary()

    # save/compile model
    cp = ModelCheckpoint('models/model3/model.keras', save_best_only=True)
    model.compile(loss=loss_func, optimizer=Adam(learning_rate=learning_rate), metrics=[RootMeanSquaredError()])

    return model, cp


# Model 4 - Double LSTM
def create_model4(output_size, window_size, num_features, num_neurons, print_summary=False, loss_func=Huber(), learning_rate=0.001):
    model = Sequential()
    model.add(InputLayer((window_size, num_features)))
    model.add(LSTM(int(num_neurons/2), return_sequences=True))
    model.add(LSTM(num_neurons))
    model.add(Dense(int(num_neurons/2), 'relu'))
    model.add(Dense(output_size), 'relu')
    #
    if print_summary:
        model.summary()

    # save/compile model
    cp = ModelCheckpoint('models/model4/model.keras', save_best_only=True)
    model.compile(loss=loss_func, optimizer=Adam(learning_rate=learning_rate), metrics=[RootMeanSquaredError()])

    return model, cp

# Model 5 - LSTM with Attention Mechanism
def create_model5(output_size, window_size, num_features, num_neurons, print_summary=False, loss_func=Huber(), learning_rate=0.001):
    # Attention Mechanism
    def attention_layer(inputs):
        attention = Dense(1, activation='tanh')(inputs)
        attention = Permute((2, 1))(attention)
        attention = Softmax(axis=-1)(attention)  # Using Keras Softmax layer
        attention = Permute((2, 1))(attention)
        return Multiply()([inputs, attention])

    # LSTM with Attention Model
    input_seq = Input(shape=(window_size, num_features)) 
    lstm_out = LSTM(num_neurons, return_sequences=True)(input_seq)
    attention_out = attention_layer(lstm_out)

    # Using Lambda to compute the reduce sum
    output = Dense(output_size)(Lambda(lambda x: tf.reduce_sum(x, axis=1))(attention_out)) # by default output_size = 1

    # Create the model
    model = Model(inputs=input_seq, outputs=output)
    #
    if print_summary:
        model.summary()

    # save/compile model
    cp = ModelCheckpoint('models/model5/model.keras', save_best_only=True)
    model.compile(loss=loss_func, optimizer=Adam(learning_rate=learning_rate), metrics=[RootMeanSquaredError()])

    return model, cp


# Model 6 - Bidirecitonal LSTM (BiLSTM)
def create_model6(output_size, window_size, num_features, num_neurons, print_summary=False, loss_func=Huber(), learning_rate=0.001):
    # Create a simple Bidirectional LSTM model
    model = Sequential([
        Bidirectional(LSTM(num_neurons, return_sequences=True), input_shape=(window_size, num_features)),
        Bidirectional(LSTM(num_neurons)),
        Dense(8),
        Dense(output_size)
    ])

    if print_summary:
        model.summary()

    # save/compile model
    cp = ModelCheckpoint('models/model6/model.keras', save_best_only=True)
    model.compile(loss=loss_func, optimizer=Adam(learning_rate=learning_rate), metrics=[RootMeanSquaredError()])

    return model, cp


# Model 7 - Conv1D with multi layers
def create_model7(output_size, window_size, num_features, print_summary=False, loss_func=Huber(), learning_rate=0.001):
    # Create a Conv1D model for timestamp prediction
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=(window_size, num_features)),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(50, activation='relu'),
        Dense(output_size)  # Output layer for predicting the timestamp
    ])

    if print_summary:
        model.summary()

    # save model
    cp = ModelCheckpoint('models/model7/model.keras', save_best_only=True)
    model.compile(loss=loss_func, optimizer=Adam(learning_rate=learning_rate), metrics=[RootMeanSquaredError()])

    return model, cp


# Model 8 - Multi-Layer Conv1D + Single-Layer LSTM
def create_model8(output_size, window_size, num_features, num_neurons, print_summary=False, loss_func=Huber(), learning_rate=0.001):
    # Create a Conv1D + LSTM model for timestamp prediction
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=(window_size, num_features)),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        
        # Adding LSTM layer after Conv1D layers
        LSTM(num_neurons, return_sequences=False),  # Set return_sequences=True if stacking more LSTMs
        Dropout(0.2),  # Optional: Dropout for regularization
        Dense(output_size)  # Output layer for predicting the timestamp
    ])

    if print_summary:
        model.summary()

    # save model
    cp = ModelCheckpoint('models/model8/model.keras', save_best_only=True)
    model.compile(loss=loss_func, optimizer=Adam(learning_rate=learning_rate), metrics=[RootMeanSquaredError()])

    return model, cp


# Model 9 - TCN model
def create_model9(num_filters=64, window_size=10, num_features=10, kernel_size=3, dilation_rates=[1, 2, 4, 8], output_size=1, print_summary=False,
                  loss_func=Huber(), learning_rate=0.001):
    inputs = tf.keras.Input(shape=(window_size, num_features))
    x = inputs
    
    for dilation_rate in dilation_rates:
        # Dilated Conv1D layer
        conv = tf.keras.layers.Conv1D(
            filters=num_filters, 
            kernel_size=kernel_size, 
            padding='causal',  # Ensures the model does not look into the future
            dilation_rate=dilation_rate,
            activation='relu'
        )(x)

        # Residual connection (1x1 Conv to ensure the shapes match)
        if conv.shape[-1] != x.shape[-1]:
            x = tf.keras.layers.Conv1D(filters=conv.shape[-1], kernel_size=1, padding='same')(x)
        
        x = tf.keras.layers.Add()([conv, x])  # Add residual connection

    # Global Average Pooling to reduce the output to a single value
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    # Dense layer for the final output (predicting one value per sequence)
    output = tf.keras.layers.Dense(output_size)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=output)

    if print_summary:
        model.summary()

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss_func, metrics=[RootMeanSquaredError()])

    # save model
    cp = ModelCheckpoint('models/model9/model.keras', save_best_only=True)
    model.compile(loss=loss_func, optimizer=Adam(learning_rate=learning_rate), metrics=[RootMeanSquaredError()])

    return model, cp

# Model 10 - GRU with Dropout and Huber loss function
def create_model10(output_size, window_size, num_features, num_neurons, print_summary=False, loss_func=Huber(),
                   learning_rate=0.001, dropout=0):
    model = Sequential()
    model.add(InputLayer((window_size, num_features)))
    
    # GRU with tanh activation (default) and optional dropout
    if dropout > 0:
        model.add(GRU(num_neurons, activation='relu', dropout=dropout, recurrent_dropout=dropout))
    else:
        model.add(GRU(num_neurons, activation='relu'))  # Default
    
    # Output layer with no activation (linear output) for regression
    model.add(Dense(output_size, activation=None))
    
    if print_summary:
        model.summary()

    # Compile the model
    cp = ModelCheckpoint('models/model10/model.keras', save_best_only=True)
    model.compile(loss=loss_func, optimizer=Adam(learning_rate=learning_rate), metrics=[RootMeanSquaredError()])

    return model, cp




def create_model11(output_size, window_size, num_features, num_neurons, print_summary=False, loss_func=Huber(), learning_rate=0.0001):
    # Input layer
    inputs = Input(shape=(window_size, num_features))

    # Bidirectional LSTM
    lstm_out = Bidirectional(LSTM(units=num_neurons, return_sequences=True, activation='relu'))(inputs)

    # Second LSTM layer
    lstm_out = LSTM(units=num_neurons, return_sequences=True)(lstm_out)

    # Attention mechanism
    # The query and value are both taken from lstm_out; the key can also be lstm_out if desired.
    attention = Attention()([lstm_out, lstm_out])

    # Flatten the attention output if necessary
    flatten = Flatten()(attention)

    # Dense layers
    dense_out = Dense(16, activation='relu')(flatten)
    output = Dense(output_size)(dense_out)  # Final output for next timestamp prediction

    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=output)

    if print_summary:
        model.summary()

    # Save the best model
    cp = ModelCheckpoint('models/model1/model.keras', save_best_only=True)

    # Compile the model using a finely-tuned Adam optimizer
    optimizer = Adam(learning_rate=learning_rate, beta_1=0.8, beta_2=0.999, decay=1e-6, clipnorm=1.0)

    # Compile the model with Huber loss and Adam optimizer
    model.compile(loss=Huber(delta=0.9), optimizer = optimizer, metrics=[RootMeanSquaredError()])

    return model, cp




#####################
####### UTILS #######
#####################


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

    # Example usage
    # window_sizes = [10, 20, 30]
    # gru_units_list = [64, 128]
    # learning_rates = [0.0001, 0.001]
    # epochs = 50
    # batch_size = 32

    # Assume X_train, y_train, X_val, y_val are defined
    # best_model, best_params, history_results = train_and_evaluate_models(
    #     X_train=X2_train, 
    #     y_train=y2_train, 
    #     X_val=X2_val, 
    #     y_val=y2_val, 
    #     output_size=1, 
    #     num_features=1089, 
    #     window_sizes=window_sizes, 
    #     gru_units_list=gru_units_list, 
    #     dense_units_list=dense_units_list, 
    #     learning_rates=learning_rates, 
    #     epochs=epochs, 
    #     batch_size=batch_size
    # )

    # print(f"Best model parameters: {best_params}")


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


def train_val_test_data_split_size(original_data, TRAIN_SPLIT=0.8, VAL_SPLIT=0.1):
    # Split the data into training, validation, and testing sets
    train_size = int(len(original_data) * TRAIN_SPLIT)  # 80% for training (default)
    val_size = int(len(original_data) * VAL_SPLIT)    # 10% for validation (default)
    test_size = len(original_data) - train_size - val_size  # Remaining 10% for testing (default)

    train_size, val_size, 

# works for a single feature predicting itself
def df_to_X_y_version1(df, window_size):
    df_as_np = df.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np)-window_size):
        row = [[a] for a in df_as_np[i:i+window_size]]
        X.append(row)
        label = df_as_np[i+window_size]
        y.append(label)
    return np.array(X), np.array(y)

    # X                      y 
    # [[1],[2],[3],[4],[5]] [6]
    # [[2],[3],[4],[5],[6]] [7]
    # [[3],[4],[5],[6],[7]] [8]


# works for multiple features predicting a single feature
def df_to_X_y_version2(df, window_size, target='rts'):
    # Find the position (index) of the 'rts' column dynamically
    rts_index = df.columns.get_loc(target)

    df_as_np = df.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np)-window_size):
        row = [r for r in df_as_np[i:i+window_size]]
        X.append(row)
        label = df_as_np[i+window_size][rts_index] # just the 'rts'
        y.append(label) # this is the actual value
    return np.array(X), np.array(y)

    # X                                                                                      y 
    # [[rts1, ipv4_tl1],[rts2, ipv4_tl2],[rts3, ipv4_tl3],[rts4, ipv4_tl4],[rts5, ipv4_tl5]] [rts6].  In this case, we are using a subset of features to predict 'rts'
    # [[rts2, ipv4_tl2],[rts3, ipv4_tl3],[rts4, ipv4_tl4],[rts5, ipv4_tl5],[rts6, ipv4_tl6]] [rts7]
    # [[rts3, ipv4_tl3],[rts4, ipv4_tl4],[rts5, ipv4_tl5],[rts6, ipv4_tl6],[rts7, ipv4_tl7]] [rts8]


def df_to_X_y_version3(df, window_size, target=['rts', 'ipv4_tl']):
    # Dynamically find the indices of the target columns
    target_indices = [df.columns.get_loc(col) for col in target]
    
    # Convert the DataFrame to NumPy for efficient sliding window processing
    df_as_np = df.to_numpy()
    
    X = []
    y = []
    
    # Loop through the data to create sliding windows
    for i in range(len(df_as_np) - window_size):
        # Select the window for X
        row = df_as_np[i:i+window_size]
        X.append(row)
        
        # Dynamically select the target columns from the (i + window_size)-th row for y
        label = tuple(df_as_np[i + window_size][target_indices])  # Select the columns based on target indices
        y.append(label)
        
    return np.array(X), np.array(y)

    # X                                                                                      y 
    # [[rts1, ipv4_tl1],[rts2, ipv4_tl2],[rts3, ipv4_tl3],[rts4, ipv4_tl4],[rts5, ipv4_tl5]] [rts6]
    # [[rts2, ipv4_tl2],[rts3, ipv4_tl3],[rts4, ipv4_tl4],[rts5, ipv4_tl5],[rts6, ipv4_tl6]] [rts7]
    # [[rts3, ipv4_tl3],[rts4, ipv4_tl4],[rts5, ipv4_tl5],[rts6, ipv4_tl6],[rts7, ipv4_tl7]] [rts8]


# Function to perform RFE and return the selected features and labels
def feature_selection_with_rfe(df, window_size, base_estimator=RandomForestRegressor(verbose=2), n_features_to_select=10):
    # Call the data transformation function
    X, y = df_to_X_y_version2(df, window_size, target='rts')
    
    # Reshape X to 2D for RFE (required by RFE)
    n_samples, n_time_steps, n_features = X.shape
    X_flat = X.reshape(n_samples, n_time_steps * n_features)
    
    # Use RandomForestRegressor as the base estimator for RFE
    rfe = RFE(estimator=base_estimator, n_features_to_select=n_features_to_select, step=1)
    rfe.fit(X_flat, y)
    
    # Get the selected features from RFE
    selected_features = rfe.support_  # Boolean array indicating selected features
    
    # Transform the data to keep only selected features
    X_rfe = X_flat[:, selected_features]
    
    # Reshape back to 3D if needed
    X_rfe = X_rfe.reshape(X_rfe.shape[0], n_time_steps, -1)
    
    # Split the reduced dataset into train/test sets
    X_train_rfe, X_test_rfe, y_train, y_test = train_test_split(X_rfe, y, test_size=0.2, random_state=42)
    
    # Return selected feature data and the corresponding labels
    return X_train_rfe, X_test_rfe, y_train, y_test

    # Usage:
    # # First, create and compile a model (e.g., model9)
    # X_train_rfe, X_test_rfe, y_train, y_test = feature_selection_with_rfe(df, WINDOW_SIZE, base_estimator=RandomForestRegressor(),
    #                                                                       n_features_to_select=4)
    # history9 = model9.fit(X_train_rfe, y_train, epochs=EPOCHS, validation_data=(X2_val, y2_val), batch_size=16)


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

    # Example usage:
    # selected_features = get_important_features(df, target_col='rts', threshold=0.02)
    # print("Selected features based on importance (in descending order):", selected_features)



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

    # Example usage:
    # target_columns = ['rts', 'another_target']  # List of multiple target columns
    # selected_features = get_important_features_multi_target(df, target_cols=target_columns, threshold=0.02)
    # print("Selected features based on importance (in descending order) for each target:", selected_features)


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

    # Example usage:
    # selected_features = get_important_features_rfr(df, target_col='rts', threshold=0.02, scale=True)
    # print("Selected features:", selected_features)


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

    # Example usage:
    # selected_features = multi_target_random_forest(df, target_cols=['rts', 'stream_id'], threshold=0.02, scale=True)
    # print("Selected features:", selected_features)


def convert_columns_to_int(input_pcap_dir, sorted_npt_files, prefixes_list):
    all_dataframes = []  # To hold DataFrames for concatenation

    for file in sorted_npt_files:
        file_path = os.path.join(input_pcap_dir, file)
        print(f"Processing file: {file_path}")

        # Load your DataFrame
        df = pd.read_csv(file_path)

        ignored_columns = set()  # To track columns to be ignored

        for column_prefixes in prefixes_list:
            for column_prefix, bit_length in column_prefixes.items():
                # Gather all columns with the given prefix
                cols = [col for col in df.columns if col.startswith(column_prefix)]

                # Check if any of the columns contain -1
                if any(df[col].isin([-1]).any() for col in cols):
                    print(f"Ignoring prefix '{column_prefix}' due to presence of -1 in columns.")
                    ignored_columns.update(cols)  # Mark all columns of this prefix to be ignored
                    continue  # Skip this prefix if -1 is present

                # Ensure we only process the right number of columns
                if len(cols) == bit_length:
                    # Concatenate bit columns into a single binary string and convert to an integer
                    df[column_prefix.rstrip('_')] = df[cols].astype(str).agg(''.join, axis=1).apply(lambda x: int(x, 2))

                    # Optionally drop the individual bit columns after conversion
                    df.drop(columns=cols, inplace=True)

        # Drop all ignored columns after checking all prefixes
        df.drop(columns=ignored_columns, inplace=True, errors='ignore')

        # Append the modified DataFrame to the list if it has columns after conversion
        if not df.empty:
            all_dataframes.append(df)

    return all_dataframes  # Return all DataFrames with relevant columns converted to int



def process_npt_files(input_pcap_dir, prefixes_list, SINGLE_FILE=0):
    """
    Processes .npt files in the input directory, converts columns to integers, and concatenates them.
    
    Parameters:
        input_pcap_dir (str): Directory containing .npt files.
        prefixes_list (list of dict): List of dictionaries containing prefixes and their corresponding columns.
        SINGLE_FILE (int): If greater than 0, process only the SINGLE_FILE-th file from the sorted list.
    
    Returns:
        pd.DataFrame: Concatenated DataFrame of the extracted columns from the .npt files.
    """

    # List all .npt files in the input_pcap_dir
    npt_files = [f for f in os.listdir(input_pcap_dir) if f.endswith('.npt')]
    print("path:", npt_files)

    # Delete any .npt files that end with '_all'
    for file in npt_files:
        if file.endswith('_all.npt'):
            file_path = os.path.join(input_pcap_dir, file)
            try:
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
            except OSError as e:
                print(f"Error deleting file {file}: {e}")

    # Sort the list based on the numerical part of the filenames
    sorted_npt_files = sorted(
        [file for file in npt_files if file.split('_')[1].split('.')[0].isdigit()],
        key=lambda x: int(x.split('_')[1].split('.')[0])
    )

    # If SINGLE_FILE is specified, select only the SINGLE_FILE-th file from the sorted list
    if SINGLE_FILE > 0:
        sorted_npt_files = [sorted_npt_files[SINGLE_FILE-1]]

    # Call the modified convert_columns_to_int function (Assuming this function is defined elsewhere)
    dataframes = convert_columns_to_int(input_pcap_dir, sorted_npt_files, prefixes_list)

    # Concatenate all extracted DataFrames side by side (column-wise)
    if dataframes:
        df = pd.concat(dataframes, ignore_index=True)

        # Optionally save the final DataFrame to a new .npt file
        output_file = os.path.join(input_pcap_dir, "Discord_all.npt")
        df.to_csv(output_file, index=False, header=True, sep=',')  # Adjust delimiter if necessary

        print(f"Extracted columns from .npt files and saved to {output_file}")
    else:
        print("No data was extracted. Please check the column names or file contents.")
        return None

    # Print extracted columns
    for prefix_dict in prefixes_list:
        for col in prefix_dict.keys():
            print(f"Extracted column '{col}' from .npt files and saved to {output_file}")

    return df

    # Example usage:
    # ipv4_prefixes = {'ipv4_src': 32, 'ipv4_dst': 32, ... }
    # tcp_prefixes = {'tcp_sport': 16, 'tcp_dport': 16, ... }
    # udp_prefixes = {'udp_sport': 16, 'udp_dport': 16, ... }
    # prefixes_list = [ipv4_prefixes, tcp_prefixes, udp_prefixes]

    # df = process_npt_files(input_pcap_dir="/path/to/pcap/dir", prefixes_list=prefixes_list, SINGLE_FILE=0)


# Function to create a canonical form of the flow
def canonical_flow(flow_str):
    # Remove any trailing .0 from ports
    flow_str = re.sub(r'(\d+)\.0_', r'\1_', flow_str)
    
    # Split the flow string into its components
    try:
        src_ip, dst_ip, src_port, dst_port, proto = flow_str.split('_')
    except ValueError:
        print(f"Error: '{flow_str}' is not in the correct format")
        return None  # Or handle the error as needed

    # Sort the IP-port pairs lexicographically
    if (src_ip, src_port) > (dst_ip, dst_port):
        # Swap the source and destination if needed
        src_ip, dst_ip = dst_ip, src_ip
        src_port, dst_port = dst_port, src_port
    
    # Return the canonical flow string
    return f"{src_ip}_{dst_ip}_{src_port}_{dst_port}_{proto}"


def calculate_vif(df, target_column):
    """
    Calculate the Variance Inflation Factor (VIF) for each feature in the DataFrame, 
    excluding the target variable.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the features and target variable.
        target_column (str): The name of the target column to be excluded from VIF calculation.

    Returns:
        pd.DataFrame: A DataFrame containing the feature names and their corresponding VIF values.
    """
    # Drop the target column from the feature set
    X_vif = df.drop(target_column, axis=1)
    
    # Create a DataFrame to hold the VIF data
    vif_data = pd.DataFrame()
    vif_data['feature'] = X_vif.columns
    
    # Calculate VIF for each feature
    vif_data['VIF'] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
    
    # Return the DataFrame with VIF values
    return vif_data

    # Example usage:
    # vif_result = calculate_vif(df, 'rts')
    # print(vif_result)


def set_inbound_outbound_stream_as_unique(df):
    flows = df['flow']

    # Unique flows (original data)
    unique_flows = set()

    for flow in flows:
        unique_flows.add(canonical_flow(flow))  # Canonicalize the original flows

    print("Unique flows:", len(unique_flows))  # Inbound traffic == Outbound traffic

    # Convert the set to a dictionary with IDs
    flow_dict = {flow: idx for idx, flow in enumerate(unique_flows, start=0)}

    # Print the resulting dictionary
    for flow, flow_id in flow_dict.items():
        print(f"Flow: {flow}, ID: {flow_id}")
    
    # Add a new column 'stream_id' to df by mapping the canonical form of the flow to flow_dict
    df['stream_id'] = df['flow'].apply(lambda flow: flow_dict.get(canonical_flow(flow), -1))

    return df, len(unique_flows)


def set_inbound_outbound_stream_as_distinct(df):
    flows = df['flow']

    # distinct flows (original data)
    distinct_flows = set()

    for flow in flows:
        distinct_flows.add(flow)

    print("Distinct flows:", len(distinct_flows))  

    # Convert the set to a dictionary with IDs
    flow_dict = {flow: idx for idx, flow in enumerate(distinct_flows, start=0)}

    # Print the resulting dictionary
    for flow, flow_id in flow_dict.items():
        print(f"Flow: {flow}, ID: {flow_id}")
    
    # Add a new column 'stream_id' to df by mapping the canonical form of the flow to flow_dict
    df['stream_id'] = df['flow'].map(flow_dict)

    return df

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


    import pandas as pd
import matplotlib.pyplot as plt

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


def plot_train_and_val_loss(history):
    # Plot training & validation loss values
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')

    # Adding labels and title
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')

    # Show the plot
    plt.show()