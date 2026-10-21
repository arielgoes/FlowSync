import pandas as pd
import numpy as np
import os
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor

def train_val_test_data_split_size(original_data, TRAIN_SPLIT=0.8, VAL_SPLIT=0.1):
    # Split the data into training, validation, and testing sets
    train_size = int(len(original_data) * TRAIN_SPLIT)  # 80% for training (default)
    val_size = int(len(original_data) * VAL_SPLIT)    # 10% for validation (default)
    test_size = len(original_data) - train_size - val_size  # Remaining 10% for testing (default)

    return train_size, val_size, test_size

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

    # Delete any .npt files that end with '_all.npt'
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
