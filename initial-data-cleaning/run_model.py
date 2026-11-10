from read_training_data import get_tension_data, get_raw_data, get_lengths, calculate_res_freq, create_torch_datasets, create_res_freq_map
from training_testing import train_keras_model
from models import sig2sig_unet

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

import torch
import tensorflow as tf

# code from gemini to bypass specific error
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, random_split

# code from gemini to bypass specific error
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
os.environ['TF_XLA_AUTO_JIT'] = '0'
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'

def prepare_data(raw_path, proc_path, stage):
    
    proc_scans = get_tension_data(proc_path, stage)
    keys_to_remove = {"scanEndMode", "measuredBy", "apaUuid", "type"}
    raw_scans = get_raw_data(raw_path, keys_to_remove)

    proc_df = pd.DataFrame(proc_scans)
    raw_df = pd.DataFrame(raw_scans)

    comb_df = pd.merge(raw_df, proc_df, on=["uid", "parent_folder"], how="outer")
    final_comb_df = comb_df.dropna(subset=['tension'])

    u_side_lengths = get_lengths("/vols/dune/sb3423/DWA/training-data/tensions/U_Layer_Tension_Template.xlsx", "U")
    v_side_lengths = get_lengths("/vols/dune/sb3423/DWA/training-data/tensions/V_Layer_Tension_Template.xlsx", "V")
    x_side_lengths = get_lengths("/vols/dune/sb3423/DWA/training-data/tensions/X_Layer_Tension_Template.xlsx", "X")
    g_side_lengths = get_lengths("/vols/dune/sb3423/DWA/training-data/tensions/G_Layer_Tension_Template.xlsx", "G")

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

    return final_df.drop(columns=["segment_id", "channel_index", "layer", "stage", "headboardNumber", "side"])

def res_freq(df, rho):
    tension = df['tension'].values
    lengths = df['length'].values / 1000 # in meters
    return calculate_res_freq(tension, lengths, rho)

def add_res_freq_map(df):
    res_freq = df['res_freq'].values
    freq = df['freq'].values
    rounded_res_freq = np.round(res_freq * 8, 0) / 8
    step = 0.125

    return create_res_freq_map(freq, rounded_res_freq, step)

def prepare_torch_datasets(df, is_sig2sig=True):
    X_padded, y_tensor = create_torch_datasets(df, sig2sig=is_sig2sig)

    dataset = TensorDataset(X_padded[0:200], y_tensor[0:200]) # testing on small subset
    dataset_train, dataset_test = random_split(dataset, lengths = [0.8, 0.2], generator=torch.Generator().manual_seed(42))  # 80% train, 20% test

    dloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)
    dloader_test = DataLoader(dataset_test, batch_size=32, shuffle=True)

    return (dloader_train, dloader_test, X_padded.shape[1], X_padded.shape[2])

def train_model(model, dloader_train, epochs, save_model_path, optim, loss, batch_size, patience):
    
    history, training_time = train_keras_model(model, dloader_train, epochs, save_model_path, optim, loss, batch_size, patience)
    return (history, training_time)

def plot_history(key, history, save_path):
    fig, ax = plt.subplots()
    ax.plot(history[key], label=f'Training {key}')
    ax.grid(True)
    ax.set_title(f'Training {key} over Epochs')
    ax.set_xlabel("Epochs")
    ax.set_ylabel(key)
    plt.tight_layout()
    plt.savefig(save_path)


if __name__ == "__main__":
    raw_path = "/vols/dune/amh24/APA-DWA-measurements/4ml/scanData/raw/APA_APA30_PostCold_27_01_26/"
    proc_path = "/vols/dune/amh24/APA-DWA-measurements/4ml/scanData/processed/APA30_PostCold_27_01_26.json"
    stage = "DWA Development"

    df = prepare_data(raw_path, proc_path, stage)
    df['res_freq_map'] = add_res_freq_map(df)
    dloader_train, dloader_test, win_size, channel = prepare_torch_datasets(df, is_sig2sig=True)

    model = sig2sig_unet(win_size, channel)
    epochs = 50
    save_model_path = "UNet/model01.keras"
    batch_size = 32
    patience = 6
    optim = 'adam'
    loss=tf.keras.losses.BinaryCrossentropy()

    history, training_time = train_model(model, dloader_train, epochs, save_model_path, optim, loss, batch_size, patience)
    plot_history('loss', history, "UNet/figures/training_loss_plot01.png")
    plot_history('mean_io_u', history, "UNet/figures/meaniou_plot01.png")


