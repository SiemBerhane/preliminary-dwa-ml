from models import sig2sig_unet
from models import (
    MaskedPeakRecall,
    MaskedPeakPrecision,
    MaskedPeakBinError,
    MaskedWeightedBinaryCrossentropy,
)
import tensorflow as tf
import torch
import numpy as np
import time

def train_keras_model(model, data_loader, epochs, save_model_path, optim, loss,
                      batch_size, patience, positive_weight=200.0):

    """"
    Trains keras model. Implements early stopping and saves the model to the specified path.

    Args:
        model: Keras model to be trained.
        data_loader: Data loader for training data.
        epochs: Number of epochs to train the model.
        save_model_path: Path to save the trained model.
        optim: Optimizer to use for training.
        loss: Loss function to use for training.
        batch_size: Batch size for training.
        patience: Number of epochs with no improvement after which training will be stopped.
    
    Returns:
        Tuple containing:
            history: Training history.
            training_time: Time taken to train the model.

        history[0] contains loss, history[1] contains metric
    
    """

    if isinstance(loss, tf.keras.losses.BinaryCrossentropy):
        loss = MaskedWeightedBinaryCrossentropy(
            positive_weight=positive_weight
        )

    model.compile(optimizer=optim,
                loss=loss,
                weighted_metrics=[
                    MaskedPeakRecall(threshold=0.5),
                    MaskedPeakPrecision(threshold=0.5),
                    MaskedPeakBinError(),
                ])

    # checkpoint = tf.keras.callbacks.ModelCheckpoint(save_model_path, monitor='loss', verbose=1, save_freq="epoch", mode='min')
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience)

    print("Starting training...")
    t1 = time.time()
    history = model.fit(data_loader ,epochs = epochs, batch_size = batch_size,
                            callbacks=[callback], shuffle=False)
    t2 = time.time()

    model.save(save_model_path)
    return (history.history, t2-t1)

def test_keras_model(model_path, data_loader, batch_size):
    model = tf.keras.models.load_model(model_path)
    testing = model.evaluate(data_loader, batch_size=batch_size)
    return testing


def evaluate_peak_metrics(model_path, data_loader, threshold=0.5):
    model = tf.keras.models.load_model(model_path, compile=False)

    target_positives = 0
    true_positives = 0
    predicted_positives = 0
    bin_errors = []

    for inputs, y_true, sample_weights in data_loader:
        y_pred = model.predict(inputs, verbose=0)
        y_true = y_true.numpy() if hasattr(y_true, "numpy") else np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        sample_weights = (
            sample_weights.numpy()
            if hasattr(sample_weights, "numpy")
            else np.asarray(sample_weights)
        )

        y_true = np.squeeze(y_true, axis=-1) if y_true.ndim == 3 else y_true
        y_pred = np.squeeze(y_pred, axis=-1) if y_pred.ndim == 3 else y_pred
        sample_weights = (
            np.squeeze(sample_weights, axis=-1)
            if sample_weights.ndim == 3
            else sample_weights
        )

        target_mask = y_true >= threshold
        predicted_mask = y_pred >= threshold
        valid_mask = sample_weights > 0

        target_positives += np.count_nonzero(target_mask & valid_mask)
        true_positives += np.count_nonzero(
            target_mask & predicted_mask & valid_mask
        )
        predicted_positives += np.count_nonzero(predicted_mask & valid_mask)

        for i in range(len(y_true)):
            valid_length = int(np.count_nonzero(sample_weights[i]))
            if valid_length == 0:
                continue
            y_true_valid = y_true[i, :valid_length]
            y_pred_valid = y_pred[i, :valid_length]
            bin_errors.append(
                abs(
                    int(np.argmax(y_pred_valid))
                    - int(np.argmax(y_true_valid))
                )
            )

    return {
        "peak_recall": true_positives / max(target_positives, 1),
        "peak_precision": true_positives / max(predicted_positives, 1),
        "mean_peak_bin_error": (
            float(np.mean(bin_errors)) if bin_errors else None
        ),
        "max_peak_bin_error": (
            int(np.max(bin_errors)) if bin_errors else None
        ),
    }

def keras_model_inference(model_path, x_freq, x_ampl, x_lengths=None, edge_bins=50):
    """
    Performs inference using a trained Keras model on the provided frequency and amplitude data.
    Pads the input data to match the model's expected input shape, suppresses edge and padded
    predictions, and returns arrays matching the model's input length.

    Args:
        model_path (str): Path to the trained Keras model.
        x_freq (np.ndarray): Input frequency data.
        x_ampl (np.ndarray): Input amplitude data.
        x_lengths (np.ndarray, optional): Input lengths data.
        edge_bins (int): Number of valid scan bins at each end to suppress.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Padded frequency and prediction arrays, both flattened to 1D.
    """
    if not isinstance(edge_bins, (int, np.integer)) or edge_bins < 0:
        raise ValueError("edge_bins must be a non-negative integer.")

    model = tf.keras.models.load_model(model_path, compile=False)
    input_length = model.get_config()["layers"][0]['config']['batch_shape'][1]

    x_freq = np.asarray(x_freq).reshape(-1)
    x_ampl = np.asarray(x_ampl).reshape(-1)

    data_length = min(input_length, len(x_freq), len(x_ampl))
    freq = np.zeros(input_length, dtype=np.float32)
    ampl = np.zeros(input_length, dtype=np.float32)
    freq[:data_length] = x_freq[:data_length]
    ampl[:data_length] = x_ampl[:data_length]

    freq_tensor = torch.tensor(freq, dtype=torch.float32)
    ampl_tensor = torch.tensor(ampl, dtype=torch.float32)

    if x_lengths is not None:
        lengths = np.zeros(input_length, dtype=np.float32)
        lengths[:data_length] = float(x_lengths)
        lengths_tensor = torch.tensor(lengths, dtype=torch.float32)
        final_tensor = torch.stack((freq_tensor, ampl_tensor, lengths_tensor), dim=1).unsqueeze(0)
    else:
        final_tensor = torch.stack((freq_tensor, ampl_tensor), dim=1).unsqueeze(0)  # Concatenate along the channel dimension

    output = model.predict(final_tensor, verbose=0).flatten()
    output[data_length:] = 0.0

    edge_bins = min(edge_bins, data_length)
    output[:edge_bins] = 0.0
    output[max(edge_bins, data_length - edge_bins):data_length] = 0.0

    return (freq, output)
