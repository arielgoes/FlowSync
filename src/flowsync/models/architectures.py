import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, Dense, InputLayer, Conv1D, Flatten, GRU, BatchNormalization, 
    Dropout, Input, Permute, Softmax, Multiply, Lambda, Bidirectional, 
    MaxPooling1D, Attention, LayerNormalization, MultiHeadAttention
)
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop, AdamW
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import Huber
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.initializers import GlorotUniform, HeNormal, Orthogonal

def mish(x):
    return x * K.tanh(K.softplus(x))

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


def create_complex_gru_model(output_size=1, window_size=10, num_features=1, num_rnn_neurons=[128, 128], 
                        filter_sizes=[128, 128, 128], print_summary=False, 
                        loss_func=['mae'], learning_rate=0.0001, dropout_rnn=0.2, dropout_window=0.2,
                        activation_rnn=mish, activation_conv='tanh', 
                        activation_dense='tanh', activation_out='tanh', 
                        l1_reg=0.001, l2_reg=0.001, conv_kernels=[3, 5, 7], num_dense_neurons=[], optimizer='Adam',
                        use_layer_norm=True, use_lstm=False, use_attention=False, num_attention_heads=0, stateful=False):
    
    # Define the input layer
    inputs = Input(shape=(window_size, num_features))

    # Add Conv1D and MaxPooling1D layers if conv_kernels list is provided
    x = inputs
    if conv_kernels:
        for i, (kernel_size, num_filters) in enumerate(zip(conv_kernels, filter_sizes)):
            if activation_conv == 'relu':
                x = Conv1D(filters=num_filters, kernel_size=kernel_size, padding='same', 
                           activation=activation_conv, kernel_initializer=HeNormal())(x)
            elif activation_conv == 'tanh':
                x = Conv1D(filters=num_filters, kernel_size=kernel_size, padding='same', 
                           activation=activation_conv, kernel_initializer=GlorotUniform())(x)
            else:
                x = Conv1D(filters=num_filters, kernel_size=kernel_size, padding='same', 
                           activation=activation_conv)(x)
            x = BatchNormalization()(x)  # Add batch normalization
            x = MaxPooling1D(pool_size=2)(x)  # Reduce sequence length by half
    else:
        x = inputs  # Skip convolution if add_conv_layers is False

    # Choose between GRU and LSTM based on the use_lstm flag
    rnn_layer = LSTM if use_lstm else GRU

    # Add RNN layers (GRU or LSTM), progressively reducing neuron counts if specified
    for i, neurons in enumerate(num_rnn_neurons[:len(num_rnn_neurons)]):
        return_seq = True if (i < len(num_rnn_neurons) - 1 or use_attention) else False
        if activation_rnn == 'relu':
            # Note: mish is handled if passed as object, but here we expect string 'relu' or 'tanh' usually, 
            # or the object itself. If string, standard keras activations.
            x = rnn_layer(neurons, activation=activation_rnn, return_sequences=return_seq,  
                        dropout=dropout_rnn if dropout_rnn > 0 else 0, recurrent_dropout=dropout_window if dropout_window > 0 else 0,
                        kernel_regularizer=regularizers.L1L2(l1=l1_reg, l2=l2_reg) if l1_reg > 0 or l2_reg > 0 else None,
                        kernel_initializer=HeNormal(), stateful=stateful)(x)
        elif activation_rnn == 'tanh':
            x = rnn_layer(neurons, activation=activation_rnn, return_sequences=return_seq,  
                        dropout=dropout_rnn if dropout_rnn > 0 else 0, recurrent_dropout=dropout_window if dropout_window > 0 else 0,
                        kernel_regularizer=regularizers.L1L2(l1=l1_reg, l2=l2_reg) if l1_reg > 0 or l2_reg > 0 else None,
                        kernel_initializer=GlorotUniform(), stateful=stateful)(x)
        else:
            x = rnn_layer(neurons, activation=activation_rnn, return_sequences=return_seq,  
                        dropout=dropout_rnn if dropout_rnn > 0 else 0, recurrent_dropout=dropout_window if dropout_window > 0 else 0,
                        kernel_regularizer=regularizers.L1L2(l1=l1_reg, l2=l2_reg) if l1_reg > 0 or l2_reg > 0 else None,
                        stateful=stateful)(x)
        
        # Apply Layer Normalization after each RNN layer if enabled
        if use_layer_norm:
            x = LayerNormalization()(x)
    
    if use_attention and num_attention_heads > 0 and num_attention_heads <= 1:
        # Use the last output from the GRU as both the query and value
        attention_output = Attention()([x, x])
        x = Flatten()(attention_output)  # Flatten if you plan to use Dense layer

    elif use_attention and num_attention_heads > 1:
        # Use MultiHeadAttention for attention mechanism
        attention_output = MultiHeadAttention(num_heads=num_attention_heads, 
                                              kernel_initializer=Orthogonal(), key_dim=num_rnn_neurons[-1])(x, x)
        
        # You might want to add a skip connection or layer normalization after attention
        x = x + attention_output  # Skip connection
        x = LayerNormalization()(x)  # Optional: add layer normalization

    x = Flatten()(x)  # Flatten if you plan to use Dense layer

    # Add dense layers specified in num_dense_neurons list
    for neurons in num_dense_neurons:
        x = Dense(neurons, activation=activation_dense, kernel_initializer=GlorotUniform())(x)
        x = BatchNormalization()(x)  # Add batch normalization

    # Final Dense output layer
    if activation_out == None:
        outputs = Dense(output_size, activation=None, kernel_initializer=GlorotUniform())(x)
    else:
        outputs = Dense(output_size, activation=activation_out, kernel_initializer=GlorotUniform())(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    if print_summary:
        model.summary()

    if optimizer == 'Adam':
        optimizer_obj = Adam(learning_rate=learning_rate, amsgrad=False)
    elif optimizer == 'AMSgrad':
        optimizer_obj = Adam(learning_rate=learning_rate, amsgrad=True)
    elif optimizer == 'Nadam':
        optimizer_obj = Nadam(learning_rate=learning_rate)
    elif optimizer == 'RMSprop':
        optimizer_obj = RMSprop(learning_rate=learning_rate)
    elif optimizer == 'AdamW':
        optimizer_obj = AdamW(learning_rate=learning_rate)
    else:
        optimizer_obj = Adam(learning_rate=learning_rate)

    model.compile(loss=loss_func, 
                optimizer=optimizer_obj, 
                metrics=['mae'])

    return model
