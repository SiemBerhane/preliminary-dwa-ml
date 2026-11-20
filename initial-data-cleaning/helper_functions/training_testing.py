from models import sig2sig_unet
import tensorflow as tf
import torch
import numpy as np
import time

def train_keras_model(model, data_loader, epochs, save_model_path, optim, loss, batch_size, patience):

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
    
    """

    model.compile(optimizer=optim,
                loss=loss,
                metrics=[tf.keras.metrics.MeanIoU(num_classes = 2)])

    # checkpoint = tf.keras.callbacks.ModelCheckpoint(save_model_path, monitor='loss', verbose=1, save_best_only=True, mode='min')
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

def keras_model_inference(model_path, x_freq, x_ampl, x_lengths=None):
    """
    Performs inference using a trained Keras model on the provided frequency and amplitude data.
    Pads the input data to match the model's expected input shape and returns the model's output.

    Args:
        model_path (str): Path to the trained Keras model.
        x_freq (np.ndarray): Input frequency data.
        x_ampl (np.ndarray): Input amplitude data.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The padded frequency [0] and the output of the model after inference, flattened to a 1D array [1]
    """
    model = tf.keras.models.load_model(model_path, compile=True)
    input_length = model.get_config()["layers"][0]['config']['batch_shape'][1]

    x_freq = np.asarray(x_freq).reshape(-1)
    x_ampl = np.asarray(x_ampl).reshape(-1)

    freq = np.zeros(input_length, dtype=np.float32)
    ampl = np.zeros(input_length, dtype=np.float32)
    freq[:min(input_length, len(x_freq))] = x_freq[:input_length]
    ampl[:min(input_length, len(x_ampl))] = x_ampl[:input_length]

    freq_tensor = torch.tensor(freq, dtype=torch.float32)
    ampl_tensor = torch.tensor(ampl, dtype=torch.float32)

    if x_lengths is not None:
        lengths = np.zeros(input_length, dtype=np.float32)
        data_length = min(input_length, len(x_freq))
        lengths[:data_length] = float(x_lengths)
        lengths_tensor = torch.tensor(lengths, dtype=torch.float32)
        final_tensor = torch.stack((freq_tensor, ampl_tensor, lengths_tensor), dim=1).unsqueeze(0)
    else:
        final_tensor = torch.stack((freq_tensor, ampl_tensor), dim=1).unsqueeze(0)  # Concatenate along the channel dimension

    output = model.predict(final_tensor)
    return (freq, output.flatten())
