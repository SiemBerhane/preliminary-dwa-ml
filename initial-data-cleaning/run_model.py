import os

from sklearn import preprocessing
# 1. Disable cuDNN autotuning BEFORE TensorFlow loads
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# 2. Disable XLA compiler flags BEFORE TensorFlow loads
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
os.environ['TF_XLA_AUTO_JIT'] = '0'

from helper_functions import prepare_data
from helper_functions import training_testing
from models import sig2sig_unet

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import random

from scipy.signal import windows

import torch
import tensorflow as tf
from sklearn.model_selection import train_test_split

# code from gemini to bypass specific error
gpus = tf.config.list_physical_devices('GPU')
tf.config.optimizer.set_jit(False)

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, random_split


def prepare_scan_data(raw_path, proc_path, stage, dataset_name):
    """
    Prepares the dataset by merging processed and raw data, calculating resonant frequencies, and saving the final DataFrame to a pickle file.
    
    Args:
        raw_path (str): Path to the raw data.
        proc_path (str): Path to the processed data.
        stage (str): Stage of the data collection.
        dataset_name (str): Name of the dataset to be saved.
        
    Returns:
        pd.DataFrame: The final combined DataFrame with calculated resonant frequencies.
    """

    proc_scans = prepare_data.get_tension_data(proc_path, stage)
    keys_to_remove = {"scanEndMode", "measuredBy", "apaUuid", "type"}
    raw_scans = prepare_data.get_raw_data(raw_path, keys_to_remove)

    proc_df = pd.DataFrame(proc_scans)
    raw_df = pd.DataFrame(raw_scans)

    comb_df = pd.merge(raw_df, proc_df, on=["uid", "parent_folder"], how="outer")
    final_comb_df = comb_df.dropna(subset=['tension'])

    u_side_lengths = prepare_data.get_lengths("/vols/dune/sb3423/DWA/training-data/tensions/U_Layer_Tension_Template.xlsx", "U")
    v_side_lengths = prepare_data.get_lengths("/vols/dune/sb3423/DWA/training-data/tensions/V_Layer_Tension_Template.xlsx", "V")
    x_side_lengths = prepare_data.get_lengths("/vols/dune/sb3423/DWA/training-data/tensions/X_Layer_Tension_Template.xlsx", "X")
    g_side_lengths = prepare_data.get_lengths("/vols/dune/sb3423/DWA/training-data/tensions/G_Layer_Tension_Template.xlsx", "G")

    full_u_df = pd.merge(final_comb_df, u_side_lengths, on="uid", how="inner") # this only includes U layer wires
    full_v_df = pd.merge(final_comb_df, v_side_lengths, on="uid", how="inner")

    full_x_df = final_comb_df[final_comb_df["uid"].str.startswith("X", na=False)]
    full_x_df["length"] = x_side_lengths

    full_g_df = final_comb_df[final_comb_df["uid"].str.startswith("G", na=False)]
    full_g_df["length"] = g_side_lengths

    df_list = [full_u_df, full_v_df, full_x_df, full_g_df]
    final_df = pd.concat(df_list, ignore_index=True)
    
    rho = 1.52e-4
    final_df['res_freq'] = res_freq(final_df, rho)

    final_df['baseline_subtracted_ampl'] = baseline_subtract(final_df['freq'].values, final_df['ampl'].values)
    final_df['smoothed_ampl'] = smooth_data(final_df['freq'].values, final_df['baseline_subtracted_ampl'].values)

    final_df = final_df.drop(columns=["segment_id", "channel_index", "layer", "stage", "headboardNumber", "side"])
    final_df.to_pickle(f"datasets/{dataset_name}.pkl")
    
    return final_df

def res_freq(df, rho):
    """
    Calculates the resonant frequency for each wire in the DataFrame based on tension and length.
    
    Calls the calculate_res_freq function from the helper_functions.prepare_data module.
    
    Args:
        df (pd.DataFrame): DataFrame containing wire data with 'tension' and 'length' columns.
        rho (float): Linear mass density of the wire.

    Returns:
        np.ndarray: Array of resonant frequencies.
    """

    tension = df['tension'].values
    lengths = df['length'].values / 1000 # in meters
    return prepare_data.calculate_res_freq(tension, lengths, rho)

def add_res_freq_map(df, window_values=[0.2, 0.8, 1.0, 0.8, 0.2]):
    """
    Adds a resonant frequency map to the DataFrame.

    Calls the create_res_freq_map function from the helper_functions.prepare_data module

    Args:
        df (pd.DataFrame): DataFrame containing wire data.
        window_values (list): List of window values for the frequency map.

    Returns:
        pd.DataFrame: DataFrame with the added resonant frequency map.
    """
    res_freq = df['res_freq'].values
    freq = df['freq'].values
    rounded_res_freq = np.round(res_freq * 8, 0) / 8
    step = 0.125

    return prepare_data.create_res_freq_map(freq, rounded_res_freq, step, window_values)

def split_data(df, test_frac=0.2):
    """
    Splits the DataFrame into training and testing sets based on the specified test fraction.
    
    Args:
        df (pd.DataFrame): The input DataFrame to be split.
        test_frac (float): Fraction of the data to be used for testing. Default is 0.2 (80% test, 20% train).
    """

    df_train, df_test = train_test_split(df, test_size=test_frac, random_state=42)
    return (df_train, df_test)

def baseline_subtract(freqs, ampls):
    """
    Performs background subtraction on the amplitude data by fitting a polynomial to the data and subtracting it.
    Calls the baseline_subtract function from the helper_functions.prepare_data module.
     Args:
        freq (list): 2D list of all frequency data for each scan.
        ampl (list): 2d list of all amplitude data for each scan.
    Returns:
        list: 2D list of smoothed amplitude data for each scan.
    """
    smoothed_ampls = [prepare_data.baseline_subtracted(freq, np.cumsum(ampl)) for freq, ampl in zip(freqs, ampls)]
    return smoothed_ampls

def smooth_data(freqs, ampls):
    """
    Smooths the amplitude data using a Savitzky-Golay filter.
    Calls the smooth_data function from the helper_functions.prepare_data module.
    Args:
        freq (list): 2D list of all frequency data for each scan.
        ampl (list): 2d list of all amplitude data for each scan.
    Returns:
        list: 2D list of smoothed amplitude data for each scan.
    """
    smoothed_ampls = [prepare_data.smooth_data(freq, ampl) for freq, ampl in zip(freqs, ampls)]
    return smoothed_ampls

def scale_data(train, test, scaler):
    """
    Scales data using the provided scaler. The scaler is fitted on the training data 
    and then applied to both training and testing data. Inputs should be dataframe columns.

    Args:
        train (pd.Series): Series containing the training data.
        test (pd.Series): Series containing the testing data.
        scaler (sklearn.preprocessing): Sklearn scaler object (e.g., StandardScaler, MinMaxScaler) to scale the data.
        None: If no scaling is desired.

    Returns:
        train (pd.Series): Scaled training Series.
        test (pd.Series): Scaled testing Series.
    """

    if scaler is None:
        return (train, test)

    train_values = train.to_numpy()
    test_values = test.to_numpy()

    # Scalar columns (for example, wire length) need a 2-D matrix, while
    # amplitude/frequency columns contain variable-length arrays.
    first_train_value = np.asarray(train_values[0])
    if first_train_value.ndim == 0: # if scalar, first element will be a 0-D array, otherwise it will be a 1-D array
        train_flat = train_values.reshape(-1, 1)
        scaler.fit(train_flat)

        train = pd.Series(
            scaler.transform(train_values.reshape(-1, 1)).flatten(),
            index=train.index,
            name=train.name,
        )
        test = pd.Series(
            scaler.transform(test_values.reshape(-1, 1)).flatten(),
            index=test.index,
            name=test.name,
        )
        return (train, test)

    flat_train = np.concatenate(
        [np.asarray(value).reshape(-1) for value in train_values]
    ).reshape(-1, 1)
    scaler.fit(flat_train)

    train = train.apply(
        lambda x: scaler.transform(np.asarray(x).reshape(-1, 1)).flatten().tolist()
    )
    test = test.apply(
        lambda x: scaler.transform(np.asarray(x).reshape(-1, 1)).flatten().tolist()
    )

    return (train, test)

def normalise_lengths(lengths):
    """"
    Normalises the lengths to the range [0, 1] by dividing by the maximum length.

    Args:
        lengths (np.ndarray): Array of lengths to be normalised.

    Returns:
        np.ndarray: Array of normalised lengths in the range [0, 1].
    """
    return lengths / np.max(lengths)

def prepare_torch_datasets(df, max_len, is_sig2sig=True, include_lengths=False, ampl_proc="raw", is_freq_scaled=False, is_ampl_scaled=False, is_length_scaled=False):
    """
    Prepares PyTorch datasets and dataloaders from a pandas DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        max_len (int): The maximum length of the input data.
        include_lengths (bool): Whether to include lengths in the dataset.
        is_sig2sig (bool): Whether to create a signal-to-signal dataset.
        ampl_proc (str): Type of amplitude data to use ("raw", "bg_sub", "smooth" ).
        is_freq_scaled (bool): Whether the frequency data is scaled.
        is_ampl_scaled (bool): Whether the amplitude data is scaled.

    Returns:
        dloader (torch.utils.data.DataLoader): PyTorch DataLoader.
        win_size (int): The window size of the input data.
        channel (int): The number of channels in the input data.
    """
    X_padded, y_tensor, sample_weights = prepare_data.create_torch_datasets(
        df,
        max_len,
        sig2sig=is_sig2sig,
        include_lengths=include_lengths,
        ampl_proc=ampl_proc,
        is_freq_scaled=is_freq_scaled,
        is_ampl_scaled=is_ampl_scaled,
        is_length_scaled=is_length_scaled
    )
    dataset = TensorDataset(X_padded, y_tensor, sample_weights)
    dloader = DataLoader(dataset, batch_size=32, shuffle=True)
    return (dloader, X_padded.shape[1], X_padded.shape[2])

def train_model(model, dloader_train, epochs, save_model_path, optim, loss, batch_size, patience):
    
    history, training_time = training_testing.train_keras_model(model, dloader_train, epochs, save_model_path, optim, loss, batch_size, patience)
    return (history, training_time)

def test_model(model_path, dloader_test, batch_size):
    return training_testing.test_keras_model(model_path, dloader_test, batch_size)

def plot_history(key, history, save_path):
    fig, ax = plt.subplots()

    try:
        ax.plot(history[key], label=f'Training {key}')
    except KeyError:
        print(f"Key '{key}' not found in history. Available keys: {list(history.keys())}")
        return
    
    ax.grid(True)
    ax.set_title(f'Training {key} over Epochs')
    ax.set_xlabel("Epochs")
    ax.set_ylabel(key)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    if len(tf.config.list_physical_devices("GPU")) == 0:
        print("No GPU found...")
        # exit(1)
    else:
        print("GPU found. Proceeding with training...")
        
    raw_path = "/vols/dune/amh24/APA-DWA-measurements/4ml/scanData/raw/APA_APA30_PostCold_27_01_26/"
    proc_path = "/vols/dune/amh24/APA-DWA-measurements/4ml/scanData/processed/APA30_PostCold_27_01_26.json"
    stage = "DWA Development"
    dataset_name = "APA30_PostCold_27_01_26"

    if os.path.exists(f"datasets/{dataset_name}.pkl"):
        print(f"Loading dataset from datasets/{dataset_name}.pkl")
        df = pd.read_pickle(f"datasets/{dataset_name}.pkl")
    else:
        print(f"Preparing dataset and saving to datasets/{dataset_name}.pkl")
        df = prepare_scan_data(raw_path, proc_path, stage, dataset_name)
    

    #add experimentation methods
    low, high, n = 3, 51, 5  # inclusive range
    win_lens = random.sample(range(low | 1, high + 1, 2), k=n) # length values for gaussian window, must be odd
    win_stds = random.sample(range(low, high), k=n)
    # win_values = [windows.gaussian(win_len, std=win_std) for win_len in win_lens for win_std in win_stds]
    include_lengths_options = True
    scaler = preprocessing.MinMaxScaler
    len_normalisation = True
    ampl_smoothing = "smooth"
    
    count = 0 # for labelling model iterations

    batch_size = 32
    patience = 8
    epochs = 100 # keep it large, early stopping will handle the actual number of epochs
    loss_class = tf.keras.losses.BinaryCrossentropy
    optimiser_class = tf.keras.optimizers.AdamW
    # optimisers = [tf.keras.optimizers.AdamW, tf.keras.optimizers.Adam, tf.keras.optimizers.SGD]
    # losses=[tf.keras.losses.BinaryCrossentropy, 
    #         tf.keras.losses.Dice, tf.keras.losses.BinaryFocalCrossentropy]

    results_df = pd.DataFrame(columns=["Name", "Training Loss", "Training Mean IOU", 
                                       "Testing Loss", "Testing Mean IOU", "Window length", 
                                       "Window standard deviation", "Training Time (s)"])

    max_len = df["freq"].apply(len).max()
    count = 0
    # winlens, scalers, lennorm
    for win_len in win_lens:
        for win_std in win_stds:
            win_values = windows.gaussian(win_len, std=win_std)

            if scaler is not None:
                freq_scaler = scaler() 
                scaler_name = type(freq_scaler).__name__
            else:
                freq_scaler = None
                scaler_name = "None"

            if len_normalisation:
                df['length'] = normalise_lengths(df['length'].values)

            df['res_freq_map'] = add_res_freq_map(df, window_values=win_values)
            df_train, df_test = split_data(df, test_frac=0.2)
            df_train["freq"], df_test["freq"] = scale_data(df_train["freq"], df_test["freq"], freq_scaler)

            dloader_train, win_size, channel = prepare_torch_datasets(df_train, max_len, is_sig2sig=True, include_lengths=include_lengths_options, ampl_proc=ampl_smoothing)
            dloader_test, _, _ = prepare_torch_datasets(df_test, max_len, is_sig2sig=True, include_lengths=include_lengths_options, ampl_proc=ampl_smoothing)
            
            model = sig2sig_unet(win_size, channel)
                                            
            model_name = f"UNet_model_{count}_win_len_{win_len}_win_std_{win_std}"
            save_model_path = f"UNet/models/prelim_testing/{model_name}.keras"
            
            # Train the model
            optimiser = optimiser_class()
            loss = loss_class()
            history, training_time = train_model(model, dloader_train, epochs, save_model_path, optimiser, loss, batch_size, patience)

            hist_keys = list(history.keys())
            loss_key = hist_keys[0]
            metric_key = hist_keys[1]
            # Plot training history
            plot_history(loss_key, history, f"UNet/figures/loss/training_loss_plot_{count}.png")
            plot_history(metric_key, history, f"UNet/figures/metric/meaniou_plot_{count}.png")
            
            print(f"Training completed in {training_time:.2f} seconds for experiment {count}.")
            
            # Test the model
            testing = test_model(save_model_path, dloader_test, batch_size)
            print(f"Testing results for experiment {count}\n Loss: {testing[0]} \n Mean IOU: {testing[1]}")

            results_dict = {
                "Name": model_name,
                "Training Loss": history[loss_key][-1],
                "Training Mean IOU": history[metric_key][-1],
                "Testing Loss": testing[0],
                "Testing Mean IOU": testing[1],
                "Window length": win_len,
                "Window standard deviation": win_std,
                "Training Time (s)": training_time
            }
            results_df = pd.concat([results_df, pd.DataFrame([results_dict])], ignore_index=True)
            count += 1

    results_df.to_csv("UNet/metrics/prelim_testing_results.csv", index=False)
                                                                            