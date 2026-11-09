from models import sig2sig_unet
import tensorflow as tf
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
