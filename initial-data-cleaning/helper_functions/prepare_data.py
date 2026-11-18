import json
import os
import pandas as pd
import numpy as np
import warnings
import torch
from scipy.signal import savgol_filter
from sklearn import preprocessing

def get_tension_data(processed_data_path, stage, layers=["G", "X", "U", "V"], extract_all_tensions=True):
    """
    Extract tension data from the processed JSON file for specified layers and store it in a flat list of records.

    Args:
        processed_data_path (str): Path to the processed JSON file.
        stage (str): Stage the data corresponds to (e.g., "Winding", "DWA Development" etc.).
        layers (list): List of layer names to extract tension data from. If left empty, assumes data contains all layers.
        extract_all_tensions (bool): If True, extracts all tension readings for each wire segment. 
                                      If False, only extracts the minimum non-zero tension reading for each wire segment.
    
    Returns:
        list: A list of dictionaries containing tension data for each wire in the specified layers. 
        Each element contains the following keys:
            - uid: A unique identifier for the wire segment, layer + side + segment number.
            - tension: The tension value extracted from the processed data.
    """
    
    full_data = json.load(open(processed_data_path))
    stage_data = full_data[stage]
    flat_records = []
    
    for layer in layers:
        layer_data = stage_data[layer]

        for side, side_data in layer_data.items():  
            for segment_id, wire_data in side_data.items(): # gets first two indices of dict
                    
                if len(wire_data['tension']) == 0:
                    continue

                if extract_all_tensions:
                    for parent_folder in wire_data['tension'].keys():  
                        tension = extract_tension_from_reading(wire_data["tension"][parent_folder])

                        if tension is not None:
                            flat_records.append({
                                            "uid": str(layer)+str(side)+str(int(segment_id)), # int(segment_id) to remove leading zeros
                                            "parent_folder": parent_folder,
                                            "tension": tension
                                        })
                
                else: # select min non zero tension from all readings
                    for parent_folder in wire_data['tension'].keys():  
                        tensions = []
                        tension = extract_tension_from_reading(wire_data["tension"][parent_folder])

                        if tension is not None:
                            tensions.append(tension)

                    if len(tensions) > 0:
                        flat_records.append({
                            "uid": str(layer)+str(side)+str(int(segment_id)), # int(segment_id) to remove leading zeros
                            "parent_folder": parent_folder,
                            "tension": min(tension)
                        })

    return flat_records

def extract_tension_from_reading(reading):
    """
    Extracts the tension value from a reading dictionary, prioritizing manual readings over auto readings.

    Args:
        reading (dict): A dictionary containing all tension readings for a given segment, which may include both manual and auto readings. 
        Should equal side_data[segment_id]['tension'].keys()[0] e.g 20230208T185357_U_A_<segment>
    
    Returns:
        float or None: The extracted tension value, or None if no valid tension is found.
    """

    try:                            
        if len(reading) == 2: # manually corrected tensions also include auto calculations
            tension = reading["Manual"]["tension"]
        elif len(reading) == 1:
            tension = reading["Auto"]["tension"]

        if type(tension) is str or tension <= 0:
            tension = None

    except IndexError as e:
        return e

    return tension
    

def get_raw_data(path, keys_to_remove):
    """
    Returns a list of dictionaries containing the cleaned scan data from all subdirectories of the given path.
    
    Args:
        path (str): The root directory containing all the subdirectories with scan data.
        keys_to_remove (list): A list of keys to remove from each scan dictionary.
    
    Returns:
        list: A list of dictionaries containing the cleaned scan data.
        """
    
    list_of_dir = []
    [list_of_dir.append(x[0]) for x in os.walk(path)] # first element is current path, rest are subdirectories

    list_of_scans = []
    for dir in list_of_dir:
        if "amplitudeData.json" in os.listdir(dir):
            rel_path = dir.removeprefix(list_of_dir[0]) # remove the root path from the current path to get relative path
            scan = json.load(open(dir + "/amplitudeData.json"))
            clean_scan = {k: v for k, v in scan.items() if k not in keys_to_remove}
            final_scan = map_wire_segments_to_channels(clean_scan, rel_path)
            list_of_scans.extend(final_scan)

    return list_of_scans


def map_wire_segments_to_channels(scan, folder_path):
    """
    Maps wire segments to their corresponding channel indices based on the APA channels in the scan data.
    
    Args:
        scan (dict): A dictionary containing the cleaned scan data. 
    
    Returns:
        list: A list of dictionaries containing the wire ID, channel index, frequency and amplitude data
        """

    apa_channels = scan['apaChannels']
    wire_segments = scan['wireSegments']

    # Create a map of channels corresponding to frequency/amplitude indexing,
    # e.g. 0 -> channel 0.
    channel_map = {}
    
    for idx, val in enumerate(apa_channels):
        if val is not None:

            try:
                channel_map[int(val) % 100] = int(idx) # sometimes values are stored as strings instead of ints

            except TypeError:
                warnings.warn(f"TypeError: Invalid value in apa_channels with value ({val}), with index ({idx})")
            except ValueError: # val = "None" should be ignored, but if it's anything else, raise warning as to not interrupt code 
                if val != "None":
                    warnings.warn(f"ValueError: Invalid value in apa_channels with value ({val}), with index ({idx})")

    extracted_scans = []

    for segment in wire_segments:
        tar_digits = segment % 100
        if tar_digits in channel_map:
            channel_idx = channel_map[tar_digits]
            channel_data = scan.get(str(channel_idx), {})
            
            # Only append if there is actual frequency data in that channel
            if channel_data.get("freq") and channel_data.get("ampl"):
                extracted_scans.append({
                    "uid": str(scan["layer"])+str(scan["side"])+str(int(segment)),
                    "segment_id": int(segment),
                    "channel_index": channel_idx,
                    "freq": channel_data["freq"],
                    "ampl": channel_data["ampl"],
                    "layer": scan["layer"],
                    "stage": scan["stage"],
                    "headboardNumber": scan["headboardNum"],
                    "side": scan["side"],
                    "parent_folder": folder_path
                })

    return extracted_scans

def get_lengths(lengths_path, layer):
    """
    Reads the lengths data of specific layer from the specified Excel file and 
    returns a DataFrame with the relevant columns.
    
    Args:
        lengths_path (str): Path to the Excel file containing lengths data. 
        layer (str): The layer for which to extract lengths data. Should be either "U", "V", "X", or "G".
        
    Returns:
        pd.DataFrame: A DataFrame containing the segment lengths for the specified layer for "U" and "V" layers, or a constant length for "X" and "G" layers.
    """
    sheet_name = layer + "-layer"

    if layer == "X": # wire lengths are constant for X and G layers
        lengths = 1.273 * 1000  # convert to mm

    elif layer == "G":
        lengths = 1.282 * 1000  # convert to mm

    elif layer == "U" or layer == "V":
        side_a = pd.read_excel(lengths_path, sheet_name=sheet_name, header=11, usecols="A, V")
        side_b = pd.read_excel(lengths_path, sheet_name=sheet_name, header=11, usecols="AJ")

        seg_index_label_a = side_a.columns[0]
        measurement_label = side_a.columns[1]
        seg_index_label_b = side_b.columns[0]

        # rows with no measurement label are classed as too short
        side_a = side_a.dropna(subset=[measurement_label]) 
        side_a[seg_index_label_a] = side_a[seg_index_label_a].astype(int)

        # side b segments have the same length as side a so just copy across
        side_b[measurement_label] = side_a[measurement_label]
        side_b = side_b.dropna(subset=[measurement_label])
        side_b[seg_index_label_b] = side_b[seg_index_label_b].astype(int)

        # create uid = layer + side + segment_index
        side_a["uid"] = layer + "A" + side_a[seg_index_label_a].astype(str)
        side_b["uid"] = layer + "B" + side_b[seg_index_label_b].astype(str)

        # dropping seg index columns so they don't clutter final dataframe
        side_a = side_a.drop(columns=[seg_index_label_a])
        side_b = side_b.drop(columns=[seg_index_label_b])

        lengths = pd.concat([side_a, side_b])
        lengths = lengths.rename(columns={measurement_label: "length"})

    else:
        raise ValueError("Invalid layer specified. Must be 'U', 'V', 'X', or 'G'.")

    return lengths


def calculate_res_freq(tensions, lengths, rho):
    """ 
    Calcualtes the resonant frequency based on fundamental frequency formula

    Args:
        tensions (pd.Series): A series containing the tension data.
        lengths (pd.Series): A series containing the length data.
        rho (float): The linear density of the wire.

    Returns:
        pd.Series: A series containing the resonant frequency data.
    """

    return np.sqrt(tensions / rho) * (1 / (2*lengths))

def create_torch_datasets(df, sig2sig = False, include_lengths = False):
    """
    Creates PyTorch datasets for the input features and target variable from the given DataFrame.
    
    Args:
        df (pd.DataFrame): A DataFrame containing the input features and target variable.
        sig2sig (bool): A flag indicating whether the target variable is a signal (True) or a scalar value (False). Default is False.

    Returns:
        tuple: A tuple containing the input features tensor (X_tensor) and target variable tensor (y_tensor).
    """

    max_len = df["freq"].apply(len).max()
    target_len = ((max_len + 63) // 64) * 64
    num_samples = len(df)

    padded_freq = torch.zeros((num_samples, target_len), dtype=torch.float32)
    padded_ampl = torch.zeros((num_samples, target_len), dtype=torch.float32)

    if sig2sig:
        padded_y = torch.zeros((num_samples, target_len), dtype=torch.float32) 

        if include_lengths:
            wire_lengths = df["length"].values
            padded_lengths = torch.zeros((num_samples, target_len), dtype=torch.float32)

    if sig2sig:

        if include_lengths:
            for i, (f_lst, a_lst, y_lst) in enumerate(zip(df["freq"], df["ampl"], df["res_freq_map"])):
                padded_freq[i, :len(f_lst)] = torch.tensor(f_lst, dtype=torch.float32)
                padded_ampl[i, :len(a_lst)] = torch.tensor(a_lst, dtype=torch.float32)
                padded_lengths[i, :len(f_lst)] = torch.tensor([wire_lengths[i]] * len(f_lst), dtype=torch.float32) # scalar repeated for each freq data point
                padded_y[i, :len(y_lst)] = torch.tensor(y_lst, dtype=torch.float32)
        else:
            for i, (f_lst, a_lst, y_lst) in enumerate(zip(df["freq"], df["ampl"], df["res_freq_map"])):
                padded_freq[i, :len(f_lst)] = torch.tensor(f_lst, dtype=torch.float32)
                padded_ampl[i, :len(a_lst)] = torch.tensor(a_lst, dtype=torch.float32)
                padded_y[i, :len(y_lst)] = torch.tensor(y_lst, dtype=torch.float32)
                
    else:
        for i, (f_lst, a_lst) in enumerate(zip(df["freq"], df["ampl"])):
            padded_freq[i, :len(f_lst)] = torch.tensor(f_lst, dtype=torch.float32)
            padded_ampl[i, :len(a_lst)] = torch.tensor(a_lst, dtype=torch.float32)

    # Stack into the final 3D tensor (N, target_len, 2)
    if include_lengths:
        X_padded = torch.stack((padded_freq, padded_ampl, padded_lengths), dim=-1)
    else:
        X_padded = torch.stack((padded_freq, padded_ampl), dim=-1)

    if sig2sig:
        y_tensor = padded_y
    else:
        y_tensor = torch.tensor(df["res_freq"].values, dtype=torch.float32).view(-1, 1)

    return (X_padded, y_tensor)

def train_scaler(feature, scaler):
    """
    Scales the specified features in the DataFrame using Min-Max scaling.

    Args:
        df (pd.DataFrame): A DataFrame containing the features to be scaled.
        features (list): A list of feature names to be scaled.
        scalaer (sklearn.preprocessing): A fitted Min-Max scaler object.

    Returns:
        pd.DataFrame: A DataFrame with the specified features scaled to the range [0, 1].
    """
    return scaler.fit(feature.reshape(-1, 1))

def baseline_subtracted(freq: np.ndarray, ampl: np.ndarray) -> np.ndarray:
    """
    Subtracts the baseline from the data using a savgol filter.

    Args:
        freq (np.ndarray): An array of frequency values.
        ampl (np.ndarray): An array of amplitude values.

    Returns:
        np.ndarray: An array of baseline-subtracted amplitude values.
    """
    num_bins = _get_num_savgol_bins(freq)
    smooth_curve = savgol_filter(ampl, num_bins, 3)
    signal = ampl - smooth_curve
    return signal

def _get_num_savgol_bins(freq: np.ndarray) -> int:
    """
    Determines the appropriate number of bins to use for the savgol filter.

    Args:
        freq (np.ndarray): An array of frequency values.

    Returns:
        int: The number of bins to use for the savgol filter.
    """
    step_size = freq[1] - freq[0]
    num_bins = int(round(27 / (8 * step_size)))
    if (num_bins % 2) == 0:
        num_bins += 1
    return num_bins

def smooth_data(freq, ampl):
    bg_sub = baseline_subtracted(freq, ampl)
    bg_sub_norm = bg_sub / np.max(bg_sub)

    # Get the derivative/slope of the trace
    bg_sub_norm_dfda = np.array([slope_near_bin(freq, bg_sub_norm, i, 5) for i in range(len(freq))])
    bg_sub_norm_dfda /= np.max(bg_sub_norm_dfda)

    # The ringing resonances appear to be sin-like. So we add the square of the trace to the square of the derivative, which results in peaks where 
    # the trace is sin-like due to sin^2 + cos^2 being maximized.
    pythag = bg_sub_norm**2+bg_sub_norm_dfda**2
    # Smooth out the processed trace
    pythag_smooth = savgol_filter(pythag, _get_num_savgol_bins(freq), 3)
    
    # Normalize the processed trace
    pythag_smooth /= np.max(pythag_smooth)
    return pythag_smooth

def slope_near_bin(freqs, amps, i, window):
    """
    Calculates the slope of the amplitude data near a specific frequency bin. 
    """
    df = freqs[1]-freqs[0]
    startBin = i-int((window-1)/2)
    endBin = i+int((window-1)/2)
    if startBin < 0:
        endBin = endBin - startBin
        startBin = 0
    length = len(freqs)
    if endBin > length - 1:
        endBin = length - 1
        startBin = startBin - (endBin - (length - 1))
    return (amps[endBin]-amps[startBin])/df

def create_res_freq_map(freq, res_freq, step, window_values):
    """
    Creates a map of resonance frequencies for each frequency reading.

    Args:
        freq (list): A list of frequency readings.
        res_freq (list): A list of resonance frequencies corresponding to each frequency reading.
        step (float): The step size between frequency readings.
        window_values (list): The dynamic values to place around the center index 
                              (e.g., [0.2, 0.8, 1.0, 0.8, 0.2]). Should be an odd length.

    Returns:
        list: A binary list of arrays    
    
    """
    res_freq_map = []
    
    # Convert to numpy array to ensure we can slice it easily
    window = np.array(window_values, dtype=np.float32)
    half_width = len(window) // 2
    
    for i in range(len(freq)):
        start_freq = freq[i][0]
        curr_res_freq_map = np.zeros((len(freq[i]),), dtype=np.float32)
        
        center_idx = int(round((res_freq[i] - start_freq) / step))

        # tracks index range in freq array where the window will be placed
        map_start = center_idx - half_width 
        map_end = center_idx + half_width + 1

        # tracks window range to be inserted into the map (in case of edge cases)
        win_start = 0
        win_end = len(window)
        
        # if the peak is too close to the start or end of the array
        if map_start < 0:
            win_start = abs(map_start)  # Crop the left side of the window
            map_start = 0
            
        if map_end > len(curr_res_freq_map):
            win_end -= (map_end - len(curr_res_freq_map)) # Crop the right side
            map_end = len(curr_res_freq_map)
            
        if map_start < map_end: # Ensure it hasn't been completely cropped out
            curr_res_freq_map[map_start:map_end] = window[win_start:win_end]
            
        res_freq_map.append(curr_res_freq_map)
        
    return res_freq_map
