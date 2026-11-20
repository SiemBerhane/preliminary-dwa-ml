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

def split_data(df, test_frac=0.8):
    """
    Splits the DataFrame into training and testing sets based on the specified test fraction.
    
    Args:
        df (pd.DataFrame): The input DataFrame to be split.
        test_frac (float): Fraction of the data to be used for testing. Default is 0.8 (80% test, 20% train).
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
    smoothed_ampls = [prepare_data.baseline_subtracted(freq, ampl) for freq, ampl in zip(freqs, ampls)]
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

    # amplitude data has variable length so will scale each amplitude array individually 
    flat_train = np.concatenate(train.values).reshape(-1, 1)
    scaler.fit(flat_train)

    train = train.apply(
        lambda x: scaler.transform(np.array(x).reshape(-1, 1)).flatten().tolist())

    test = test.apply(
        lambda x: scaler.transform(np.array(x).reshape(-1, 1)).flatten().tolist())

    return (train, test)

def prepare_torch_datasets(df, max_len, is_sig2sig=True, include_lengths=False, ampl_proc="raw"):
    """
    Prepares PyTorch datasets and dataloaders from a pandas DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        is_sig2sig (bool): Whether to create a signal-to-signal dataset.
        ampl_proc (str): Type of amplitude data to use ("raw", "bg_sub", "smooth" ).

    Returns:
        dloader_train (torch.utils.data.DataLoader): PyTorch DataLoader for the training set.
        dloader_test (torch.utils.data.DataLoader): PyTorch DataLoader for the testing set.
        win_size (int): The window size of the input data.
        channel (int): The number of channels in the input data.
    """
    X_padded, y_tensor = prepare_data.create_torch_datasets(df, max_len, sig2sig=is_sig2sig, include_lengths=include_lengths, ampl_proc=ampl_proc)
    dataset = TensorDataset(X_padded, y_tensor)
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
        print("No GPU found. Exiting...")
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
    win_std = np.linspace(1, 20, 5) # standard deviation values for gaussian window
    win_length = np.linspace(1, 31, 2, dtype=int) # length values for gaussian window, must be odd
    win_values = windows.gaussian(win_length, std=win_std)
    include_lengths_options = [False, True]
    freq_scalers = [preprocessing.MinMaxScaler, None]
    len_normalisation = [True, False]
    ampl_smoothing = ["raw", "bg_sub", "smooth"]
    
    count = 0 # for labelling model iterations

    batch_size = [32]
    patience = 8
    epochs = [100] # keep it large, early stopping will handle the actual number of epochs
    optimisers = [tf.keras.optimizers.BinaryCrossentropy]
    losses = [tf.keras.losses.Adam]
    # optimisers = [tf.keras.optimizers.AdamW, tf.keras.optimizers.Adam, tf.keras.optimizers.SGD]
    # losses=[tf.keras.losses.BinaryCrossentropy, 
    #         tf.keras.losses.Dice, tf.keras.losses.BinaryFocalCrossentropy]

    results_df = pd.DataFrame(columns=["Name", "Training Loss", "Training Mean IOU", 
                                       "Testing Loss", "Testing Mean IOU", "Window std", "Window length", "Include Lengths", 
                                       "Length normalisation", "Frequency Scaler",  "Amplitude Scaler", "Amplitude smoothing",
                                       "Epochs", "Batch Size", "Optimizer", "Loss Function", "Training Time (s)"])

    max_len = df["freq"].apply(len).max()

    for win_values in win_values_array:
        for include_lengths in include_lengths_options:
            for scaler_class in scalers:
                for epoch in epochs:
                    for batch in batch_size:
                        for loss_class in losses:
                            for optim_class in optimisers:
                                count += 1
                                print(f"Running experiment {count} with window values: {win_values}, include_lengths: {include_lengths}, scaler: {scaler}, epochs: {epoch}, batch size: {batch}, optimizer: {optim_class.__name__}, loss function: {loss_class.__name__}")
                                
                                optim = optim_class()
                                loss = loss_class()
                                scaler = scaler_class() if scaler_class is not None else None
                                if scaler is not None:
                                    scaler_name = type(scaler).__name__
                                else:
                                    scaler_name = "None"
                                
                                # Scale the data
                                df['res_freq_map'] = add_res_freq_map(df, window_values=win_values)
                                df_train, df_test = split_data(df, test_frac=0.8)
                                scaled_df_train, scaled_df_test = scale_data(df_train, df_test, scaler)
                                
                                # Prepare datasets
                                dloader_train, win_size, channel = prepare_torch_datasets(scaled_df_train, max_len, is_sig2sig=True, include_lengths=include_lengths)
                                dloader_test, _, _ = prepare_torch_datasets(scaled_df_test, max_len, is_sig2sig=True, include_lengths=include_lengths)
                                
                                # Create model
                                model = sig2sig_unet(win_size, channel)
                                
                                model_name = f"UNet_model_{count}_win_{len(win_values)}_len_{include_lengths}_scaler_{scaler_name}"
                                save_model_path = f"UNet/models/{model_name}.keras"
                                
                                # Train the model
                                history, training_time = train_model(model, dloader_train, epoch, save_model_path, optim, loss, batch, patience)

                                hist_keys = list(history.keys())
                                loss_key = hist_keys[0]
                                metric_key = hist_keys[1]
                                # Plot training history
                                plot_history(loss_key, history, f"UNet/figures/training_loss_plot_{count}.png")
                                plot_history(metric_key, history, f"UNet/figures/meaniou_plot_{count}.png")
                                
                                print(f"Training completed in {training_time:.2f} seconds for experiment {count}.")
                                
                                # Test the model
                                testing = test_model(save_model_path, dloader_test, batch)
                                print(f"Testing results for experiment {count}\n Loss: {testing[0]} \n Mean IOU: {testing[1]}")
                                
                                # Save metrics to file
                                results_dict = {
                                    "Training Loss": history[loss_key][-1],
                                    "Training Mean IOU": history[metric_key][-1],
                                    "Testing Loss": testing[0],
                                    "Testing Mean IOU": testing[1],
                                    "Window Values": win_values,
                                    "Include Lengths": include_lengths,
                                    "Scaler": scaler_name,
                                    "Epochs": epoch,
                                    "Batch Size": batch,
                                    "Optimizer": optim_class.__name__,
                                    "Loss Function": loss_class.__name__,
                                    "Training Time (s)": training_time

                                }

                                results_df = pd.concat([results_df, pd.DataFrame([results_dict])], ignore_index=True)
    results_df.to_csv("UNet/metrics/experiment_results.csv", index=False)

    
    
    # dloader_train, dloader_test, win_size, channel = prepare_torch_datasets(df, is_sig2sig=True)

    # model = sig2sig_unet(win_size, channel)
    
    # model_name = "UNet_model01"
    # save_model_path = "UNet/" + model_name + ".keras"
    
    

    # history, training_time = train_model(model, dloader_train, epochs, save_model_path, optim, loss, batch_size, patience)
    # plot_history('loss', history, "UNet/figures/training_loss_plot01.png")
    # plot_history('mean_io_u', history, "UNet/figures/meaniou_plot01.png")

    # print(f"Training completed in {training_time:.2f} seconds.")

    # testing = test_model(save_model_path, dloader_test, batch_size)
    # print(f"Testing results\n Loss: {testing[0]} \n Mean IOU: {testing[1]}")

    # with open("UNet/metrics/"+model_name+"_metrics.txt", "w") as f:
    #     f.write(f"Training completed in {training_time:.2f} seconds.\n")
    #     f.write(f"Training results\n Loss: {history['loss'][-1]} \n Mean IOU: {history['mean_io_u'][-1]}\n")
    #     f.write(f"Testing results\n Loss: {testing[0]} \n Mean IOU: {testing[1]}\n")
