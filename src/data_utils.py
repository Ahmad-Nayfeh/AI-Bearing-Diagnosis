# src/data_utils.py

import os
import scipy.io
import numpy as np
import pandas as pd

# Define the base directory for your data
# This assumes data_utils.py is in 'src' and data is in 'data/raw'
RAW_DATA_DIR = '../data/raw/'

# This configuration is adapted directly from your '01_data_understanding_and_eda.ipynb'
# It accurately reflects your filenames, keys, and handles missing RPMs.
SIGNAL_CONFIGURATIONS = {
    # 0 HP Load
    'Normal_0HP':    {'file_name': 'Normal_0.mat',    'rpm_key_actual': 'X097RPM', 'signal_key_actual': 'X097_DE_time', 'fault_type': 'Normal', 'motor_load_hp': 0},
    'IRF_0HP_007':   {'file_name': 'IR007_0.mat',     'rpm_key_actual': 'X105RPM', 'signal_key_actual': 'X105_DE_time', 'fault_type': 'IRF',    'motor_load_hp': 0},
    'BF_0HP_007':    {'file_name': 'B007_0.mat',      'rpm_key_actual': 'X118RPM', 'signal_key_actual': 'X118_DE_time', 'fault_type': 'BF',     'motor_load_hp': 0},
    'ORF_0HP_007_6': {'file_name': 'OR007@6_0.mat',   'rpm_key_actual': 'X130RPM', 'signal_key_actual': 'X130_DE_time', 'fault_type': 'ORF',    'motor_load_hp': 0},

    # 1 HP Load
    'Normal_1HP':    {'file_name': 'Normal_1.mat',    'rpm_key_actual': None,      'signal_key_actual': 'X098_DE_time', 'fault_type': 'Normal', 'motor_load_hp': 1, 'expected_rpm': 1772},
    'IRF_1HP_007':   {'file_name': 'IR007_1.mat',     'rpm_key_actual': 'X106RPM', 'signal_key_actual': 'X106_DE_time', 'fault_type': 'IRF',    'motor_load_hp': 1},
    'BF_1HP_007':    {'file_name': 'B007_1.mat',      'rpm_key_actual': 'X119RPM', 'signal_key_actual': 'X119_DE_time', 'fault_type': 'BF',     'motor_load_hp': 1},
    'ORF_1HP_007_6': {'file_name': 'OR007@6_1.mat',   'rpm_key_actual': 'X131RPM', 'signal_key_actual': 'X131_DE_time', 'fault_type': 'ORF',    'motor_load_hp': 1},

    # 2 HP Load
    'Normal_2HP':    {'file_name': 'Normal_2.mat',    'rpm_key_actual': None,      'signal_key_actual': 'X099_DE_time', 'fault_type': 'Normal', 'motor_load_hp': 2, 'expected_rpm': 1750},
    'IRF_2HP_007':   {'file_name': 'IR007_2.mat',     'rpm_key_actual': 'X107RPM', 'signal_key_actual': 'X107_DE_time', 'fault_type': 'IRF',    'motor_load_hp': 2},
    'BF_2HP_007':    {'file_name': 'B007_2.mat',      'rpm_key_actual': 'X120RPM', 'signal_key_actual': 'X120_DE_time', 'fault_type': 'BF',     'motor_load_hp': 2},
    'ORF_2HP_007_6': {'file_name': 'OR007@6_2.mat',   'rpm_key_actual': 'X132RPM', 'signal_key_actual': 'X132_DE_time', 'fault_type': 'ORF',    'motor_load_hp': 2},
}

def load_and_label_data(raw_data_dir=RAW_DATA_DIR, config=SIGNAL_CONFIGURATIONS):
    """
    Loads .mat files specified in the config, extracts DE signals and RPM,
    and associates them with fault type and motor load labels.

    Args:
        raw_data_dir (str): Path to the directory containing raw .mat files.
        config (dict): Configuration mapping descriptive names to file details and labels.

    Returns:
        list: A list of dictionaries, where each dictionary contains:
              'config_name': str, the descriptive name from the config
              'signal': np.array of the DE time-series data
              'rpm': float, the motor speed
              'fault_type': str, e.g., 'Normal', 'IRF'
              'motor_load_hp': int, e.g., 0, 1, 2
              'original_filename': str, the name of the .mat file
    """
    all_data = []

    for config_name, details in config.items():
        file_path = os.path.join(raw_data_dir, details['file_name'])
        try:
            mat_contents = scipy.io.loadmat(file_path)

            signal_key = details['signal_key_actual']
            if signal_key not in mat_contents:
                print(f"Warning: Signal key '{signal_key}' not found in {details['file_name']}. Skipping.")
                continue
            signal_data = mat_contents[signal_key].flatten()

            rpm_value = None
            if details['rpm_key_actual']:
                rpm_key = details['rpm_key_actual']
                if rpm_key in mat_contents:
                    rpm_candidate = mat_contents[rpm_key]
                    if isinstance(rpm_candidate, np.ndarray):
                        rpm_value = float(rpm_candidate.item(0)) if rpm_candidate.size > 0 else None
                    elif isinstance(rpm_candidate, (int, float)):
                        rpm_value = float(rpm_candidate)
                else:
                    print(f"Warning: RPM key '{rpm_key}' not found in {details['file_name']}. Checking for 'expected_rpm'.")
            
            if rpm_value is None and 'expected_rpm' in details:
                rpm_value = details['expected_rpm']
                print(f"Info: Using expected_rpm ({rpm_value}) for {details['file_name']}.")
            
            if rpm_value is None:
                 print(f"Warning: RPM could not be determined for {details['file_name']}. Setting RPM to None.")

            all_data.append({
                'config_name': config_name,
                'signal': signal_data,
                'rpm': rpm_value,
                'fault_type': details['fault_type'],
                'motor_load_hp': details['motor_load_hp'],
                'original_filename': details['file_name']
            })
            print(f"Successfully loaded and processed: {details['file_name']}")

        except FileNotFoundError:
            print(f"Error: File not found - {file_path}")
        except KeyError as e:
            print(f"Error: Key not found in {details['file_name']} - {e}")
        except Exception as e:
            print(f"An unexpected error occurred with {details['file_name']}: {e}")

    return all_data

def segment_signals(loaded_data_df, window_size=2048, overlap_ratio=0.5):
    """
    Segments the time-series signals from the loaded data into windows.

    Args:
        loaded_data_df (pd.DataFrame): DataFrame from load_and_label_data(),
                                       where each row has a 'signal' (np.array)
                                       and its corresponding labels/metadata.
        window_size (int): The number of samples in each segment.
        overlap_ratio (float): The fraction of overlap between consecutive windows (e.g., 0.5 for 50%).

    Returns:
        pd.DataFrame: A new DataFrame where each row represents a single segment,
                      containing the segment data ('signal_segment') and inherited labels.
    """
    all_segments_data = []
    step_size = int(window_size * (1 - overlap_ratio))
    if step_size <= 0: 
        step_size = 1
        print(f"Warning: Overlap ratio {overlap_ratio} with window size {window_size} results in non-positive step. Setting step to 1.")

    for index, row in loaded_data_df.iterrows():
        signal = row['signal']
        num_samples = len(signal)
        
        for i in range(0, num_samples - window_size + 1, step_size):
            segment = signal[i : i + window_size]
            
            if len(segment) == window_size:
                segment_info = {
                    'config_name': row['config_name'],
                    'original_filename': row['original_filename'],
                    'fault_type': row['fault_type'],
                    'motor_load_hp': row['motor_load_hp'],
                    'rpm': row['rpm'],
                    'signal_segment': segment 
                }
                all_segments_data.append(segment_info)
    
    if not all_segments_data:
        print("Warning: No segments were generated. Check input data, window size, and overlap settings.")
        return pd.DataFrame(columns=['config_name', 'original_filename', 'fault_type', 'motor_load_hp', 'rpm', 'signal_segment'])

    return pd.DataFrame(all_segments_data)


if __name__ == '__main__':
    # Check if the raw data directory exists and has files
    if not os.path.exists(RAW_DATA_DIR) or not os.listdir(RAW_DATA_DIR):
        print(f"Error: Raw data directory '{RAW_DATA_DIR}' is empty or does not exist.")
        print(f"Please ensure your .mat files are in '{os.path.abspath(RAW_DATA_DIR)}'.")
    else:
        # --- 1. Load the data ---
        print("======================================================")
        print("      Stage 1: Data Loading and Labeling        ")
        print("======================================================")
        loaded_data_list = load_and_label_data() 
        
        if loaded_data_list:
            loaded_df = pd.DataFrame(loaded_data_list)
            print(f"\nSuccessfully loaded {len(loaded_df)} full signal files.")
            
            if loaded_df.empty:
                print("Loaded DataFrame is empty. Cannot proceed with segmentation.")
            else:
                loaded_df['signal_length'] = loaded_df['signal'].apply(len)
                print("\nSummary of Loaded Raw Signals:")
                print(f"  Total signals: {len(loaded_df)}")
                print(f"  Total data points: {loaded_df['signal_length'].sum():,}")
                print("  Signal counts per fault type:")
                print(loaded_df['fault_type'].value_counts().to_string())
                print("  Signal counts per motor load (HP):")
                print(loaded_df['motor_load_hp'].value_counts().sort_index().to_string())
                print(f"  Min signal length: {loaded_df['signal_length'].min():,}")
                print(f"  Max signal length: {loaded_df['signal_length'].max():,}")
                print(f"  Mean signal length: {loaded_df['signal_length'].mean():,.2f}")
            
                # --- 2. Segment the loaded signals ---
                print("\n======================================================")
                print("      Stage 2: Signal Segmentation (Windowing)    ")
                print("======================================================")
                WINDOW_SIZE = 2048
                OVERLAP_RATIO = 0.5
                print(f"Segmenting signals with Window Size = {WINDOW_SIZE}, Overlap Ratio = {OVERLAP_RATIO*100}% ...")
                
                segmented_df = segment_signals(loaded_df, window_size=WINDOW_SIZE, overlap_ratio=OVERLAP_RATIO)
                
                if segmented_df.empty:
                    print("No segments were generated. Check segmentation parameters or input data.")
                else:
                    print(f"\nSuccessfully generated {len(segmented_df):,} segments.")
                    
                    print("\n--- Segmented Dataset Statistics ---")
                    print(f"Total Number of Segments: {len(segmented_df):,}")

                    # Verify segment length
                    segmented_df['segment_length'] = segmented_df['signal_segment'].apply(len)
                    if (segmented_df['segment_length'] == WINDOW_SIZE).all():
                        print(f"All segments have the correct length of {WINDOW_SIZE} samples.")
                    else:
                        print("Error: Not all segments have the correct length!")
                        print(segmented_df['segment_length'].value_counts().to_string())

                    print("\nNumber of Segments per Fault Type:")
                    print(segmented_df['fault_type'].value_counts().sort_index().to_string())
                    
                    print("\nNumber of Segments per Motor Load (HP):")
                    print(segmented_df['motor_load_hp'].value_counts().sort_index().to_string())
                    
                    print("\nNumber of Segments per Original Configuration (File):")
                    # Display more rows if needed, or sort by count
                    with pd.option_context('display.max_rows', None): 
                        print(segmented_df['config_name'].value_counts().sort_index().to_string())

                    print("\nRPM Distribution in Segments:")
                    print("Unique RPM values found in segments (count):")
                    print(segmented_df['rpm'].value_counts(dropna=False).sort_index().to_string())
                    if segmented_df['rpm'].isnull().any():
                        print(f"  Note: {segmented_df['rpm'].isnull().sum():,} segments have RPM as None (likely from files with expected RPM).")
                
                    print("\n--- Sample of Segmented Data (First 5 rows) ---")
                    # Displaying head without the actual signal_segment for brevity
                    print(segmented_df.drop(columns=['signal_segment', 'segment_length']).head())
        else:
            print("No data was loaded from raw files. Aborting.")
            
    print("\n======================================================")
    print("             Script Execution Finished              ")
    print("======================================================")