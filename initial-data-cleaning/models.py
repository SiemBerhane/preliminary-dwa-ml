from keras.layers import Conv1D, Dropout, MaxPooling1D, Input, Concatenate, Average,Conv1DTranspose
from keras.models import Model
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import os


def _prepare_mask(sample_weight):
    sample_weight = tf.cast(sample_weight, tf.float32)
    if sample_weight.shape.rank == 3:
        return tf.squeeze(sample_weight, axis=-1) # removes channel dimension if present
    return sample_weight


@tf.keras.utils.register_keras_serializable(package="DWA")
class MaskedWeightedBinaryCrossentropy(tf.keras.losses.Loss):
    """Binary cross-entropy that can emphasize target peak bins.

    Padding is excluded by the sample_weight supplied as the third item of each
    DataLoader batch. The positive_weight term controls how strongly target
    bins contribute compared with background bins.
    """
    def __init__(self, positive_weight=200.0, name="masked_weighted_binary_crossentropy",
                 **kwargs):
        kwargs.pop("reduction", None)
        super().__init__(
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
            name=name,
            **kwargs,
        )
        self.positive_weight = float(positive_weight)

    def get_config(self):
        config = super().get_config()
        config["positive_weight"] = self.positive_weight
        return config

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        epsilon = tf.keras.backend.epsilon()

        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        positive_loss = (
            self.positive_weight
            * y_true
            * -tf.math.log(y_pred)
        )
        negative_loss = (1.0 - y_true) * -tf.math.log1p(-y_pred)
        return positive_loss + negative_loss

    def __call__(self, y_true, y_pred, sample_weight=None, **kwargs):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        per_bin_loss = self.call(y_true, y_pred)

        if sample_weight is not None:
            mask = _prepare_mask(tf.cast(sample_weight, tf.float32))
            per_bin_loss = tf.squeeze(per_bin_loss, axis=-1)
            per_bin_loss *= mask
            denominator = tf.maximum(tf.reduce_sum(mask), 1e-7)
            return tf.reduce_sum(per_bin_loss) / denominator

        return tf.reduce_mean(per_bin_loss)

#%% UNET MODEL

#_______________________________________________
# Encoder (Downsampler) 
#_______________________________________________
def downsample(filters, size, apply_batchnorm=True):

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv1D(filters, size, strides=2, padding='same'))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU(0.25))

    return result


#_______________________________________________
# Decoder (Upsampler) 
#_______________________________________________
def upsample(filters, size, apply_dropout=False):

    result = tf.keras.Sequential()
    result.add(Conv1DTranspose(filters, size, strides=2 , padding='same'))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.25))

    result.add(tf.keras.layers.LeakyReLU(0.25))

    return result


################################################
#_______________________________________________
# Signal to Signal Model UNET APPROACH (Deep CNN - 12 layers)
# Trainable params: 79,121
#_______________________________________________
################################################
def sig2sig_unet(win_size, channel = 1):
    inputs = tf.keras.layers.Input(shape=[win_size,channel])

    down_stack = [
    downsample(16, 9, apply_batchnorm=False), # (bs, 8000, 1)
    downsample(16, 9), # (bs, 4000, 1)
    downsample(32, 6), # (bs, 2000, 1)
    downsample(32, 6), # (bs, 1000, 1)
    downsample(64, 3), # (bs, 500, 1)
    downsample(64, 3), # (bs, 250, 1)
    ]

    up_stack = [
    upsample(64, 3, apply_dropout=True), # (bs, 250, 1)
    upsample(32, 3), # (bs, 500, 1)
    upsample(32, 6), # (bs, 1000, 1)
    upsample(16, 6), # (bs, 2000, 1)
    upsample(16, 9), # (bs, 4000, 1)
    ]

    OUTPUT_CHANNELS = 1
    last = Conv1DTranspose(OUTPUT_CHANNELS, 9,
                                            strides=2,
                                            padding='same',
        
                                            activation='sigmoid') 

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])
    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])
    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)

@tf.keras.utils.register_keras_serializable(package="DWA")
class MaskedPeakRecall(tf.keras.metrics.Metric):
    def __init__(self, threshold=0.5, name="peak_recall", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.true_positives = self.add_variable(
            name="true_positives", shape=(), initializer="zeros"
        )
        self.target_peaks = self.add_variable(
            name="target_peaks", shape=(), initializer="zeros"
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.squeeze(y_true, axis=-1)
        y_pred = tf.squeeze(y_pred, axis=-1)
        if sample_weight is not None:
            sample_weight = _prepare_mask(sample_weight)
        else:
            sample_weight = tf.ones_like(y_true)

        # values >= threshold are set to 1, values < threshold are set to 0
        target_mask = tf.cast(y_true >= self.threshold, tf.float32)
        predicted_mask = tf.cast(y_pred >= self.threshold, tf.float32)
        weighted_target = target_mask * sample_weight
        weighted_true_positive = (
            target_mask * predicted_mask * sample_weight
        ) # true positives are where both target and predicted are 1 and 
        # are not in a padded region (sample_weight = 1)

        # sum of true positives and target peaks are accumulated across batches
        self.true_positives.assign_add(tf.reduce_sum(weighted_true_positive))
        self.target_peaks.assign_add(tf.reduce_sum(weighted_target))

    def result(self):
        return tf.math.divide_no_nan(
            self.true_positives,
            self.target_peaks,
        )

    def reset_states(self):
        self.true_positives.assign(0.0)
        self.target_peaks.assign(0.0)


@tf.keras.utils.register_keras_serializable(package="DWA")
class MaskedPeakPrecision(tf.keras.metrics.Metric):
    def __init__(self, threshold=0.5, name="peak_precision", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.true_positives = self.add_variable(
            name="true_positives", shape=(), initializer="zeros"
        )
        self.predicted_peaks = self.add_variable(
            name="predicted_peaks", shape=(), initializer="zeros"
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.squeeze(y_true, axis=-1)
        y_pred = tf.squeeze(y_pred, axis=-1)
        if sample_weight is not None:
            sample_weight = _prepare_mask(sample_weight)
        else:
            sample_weight = tf.ones_like(y_true)

        target_mask = tf.cast(y_true >= self.threshold, tf.float32)
        predicted_mask = tf.cast(y_pred >= self.threshold, tf.float32)

        self.true_positives.assign_add(
            tf.reduce_sum(target_mask * predicted_mask * sample_weight)
        )
        self.predicted_peaks.assign_add(
            tf.reduce_sum(predicted_mask * sample_weight)
        )

    def result(self):
        return tf.math.divide_no_nan(
            self.true_positives,
            self.predicted_peaks,
        )

    def reset_states(self):
        self.true_positives.assign(0.0)
        self.predicted_peaks.assign(0.0)


@tf.keras.utils.register_keras_serializable(package="DWA")
class MaskedPeakBinError(tf.keras.metrics.Metric):
    def __init__(self, name="peak_bin_error", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total_error = self.add_variable(
            name="total_error", shape=(), initializer="zeros"
        )
        self.sample_count = self.add_variable(
            name="sample_count", shape=(), initializer="zeros"
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.squeeze(y_true, axis=-1)
        y_pred = tf.squeeze(y_pred, axis=-1)
        if sample_weight is not None:
            weights = _prepare_mask(sample_weight)
        else:
            weights = tf.ones_like(y_true)

        valid_samples = tf.cast(tf.reduce_max(weights, axis=-1) > 0.0, tf.bool)
        masked_predicted = tf.where(weights > 0.0, y_pred, tf.constant(-1e9, y_pred.dtype))
        masked_target = tf.where(weights > 0.0, y_true, tf.constant(-1e9, y_true.dtype))

        predicted_indices = tf.cast(tf.argmax(masked_predicted, axis=-1), tf.float32)
        target_indices = tf.cast(tf.argmax(masked_target, axis=-1), tf.float32)
        bin_errors = tf.abs(predicted_indices - target_indices)
        bin_errors = tf.boolean_mask(bin_errors, valid_samples)

        self.total_error.assign_add(tf.reduce_sum(bin_errors))
        self.sample_count.assign_add(tf.cast(tf.shape(bin_errors)[0], tf.float32))

    def result(self):
        return tf.math.divide_no_nan(
            self.total_error,
            self.sample_count,
        )

    def reset_states(self):
        self.total_error.assign(0.0)
        self.sample_count.assign(0.0)


class EpochInferenceCallback(tf.keras.callbacks.Callback):
    """Perform inference on fixed samples and save target/prediction plots."""

    def __init__(self, sample_inputs, target, frequencies, sample_weights=None,
                 every=10, output_dir="epoch_predictions"):
        super().__init__()
        self.sample_inputs = np.asarray(sample_inputs)
        self.target = np.asarray(target).squeeze(-1)
        self.frequencies = [np.asarray(frequency).reshape(-1) for frequency in frequencies]
        if sample_weights is None:
            self.sample_weights = np.ones_like(self.target)
        else:
            sample_weights = np.asarray(sample_weights)
            self.sample_weights = (
                sample_weights.squeeze(-1)
                if sample_weights.ndim == 3
                else sample_weights
            )
        self.every = every
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every != 0:
            return

        prediction = self.model.predict(
            self.sample_inputs,
            verbose=0,
        )

        prediction = np.asarray(prediction).squeeze(-1)
        evaluation = self.model.evaluate(
            self.sample_inputs,
            self.target[..., np.newaxis],
            sample_weight=self.sample_weights,
            verbose=0,
            return_dict=True,
        )
        print(
            f"Epoch {epoch + 1}, fixed-batch evaluate: "
            f"loss={evaluation.get('loss', float('nan')):.6f}, "
            f"peak_bin_error={evaluation.get('peak_bin_error', float('nan')):.4f}"
        )

        fig, axes = plt.subplots(
            len(self.sample_inputs),
            1,
            figsize=(10, 3 * len(self.sample_inputs)),
            sharex=True,
        )
        if len(self.sample_inputs) == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            valid_length = int(np.count_nonzero(self.sample_weights[i]))
            valid_length = min(valid_length, len(self.frequencies[i]),
                               len(prediction[i]))
            frequency = self.frequencies[i][:valid_length]
            target = self.target[i, :valid_length]
            predicted = prediction[i, :valid_length]

            ax.plot(frequency, target, label="Target")
            ax.plot(frequency, predicted, label="Prediction")
            ax.set_title(f"Epoch {epoch + 1}, sample {i}")
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Amplitude")
            ax.legend()

            target_bin = int(np.argmax(target))
            predicted_bin = int(np.argmax(predicted))
            print(
                f"Epoch {epoch + 1}, sample {i}: "
                f"target={target_bin}, predicted={predicted_bin}, "
                f"bin_error={abs(target_bin - predicted_bin)}, "
                f"predicted_max={predicted.max():.4f}"
            )

        plt.tight_layout()
        plt.savefig(
            f"{self.output_dir}/epoch_{epoch + 1:03d}.png",
            dpi=150,
        )
        plt.close()
#%%
