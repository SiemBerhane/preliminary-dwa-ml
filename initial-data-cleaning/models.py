from keras.layers import Conv1D, Dropout, MaxPooling1D, Input, Concatenate, Average,Conv1DTranspose
from keras.models import Model
import tensorflow as tf


def _prepare_mask(sample_weight):
    sample_weight = tf.cast(sample_weight, tf.float32)
    if sample_weight.shape.rank == 3:
        return tf.squeeze(sample_weight, axis=-1) # removes channel dimension if present
    return sample_weight

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

#%%
