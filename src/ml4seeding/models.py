"""U-Net semantic segmentation model definition.

This module is importable without TensorFlow. The model builder functions
are only available when TensorFlow is installed (via the ``ml`` extra).
"""

from __future__ import annotations

# Architecture constants (documented for reference without importing TF)
ARCHITECTURE = {
    "input_shape": (512, 512, 3),
    "base_filters": 32,
    "num_classes": 4,
    "encoder_blocks": 4,
    "decoder_blocks": 4,
    "total_params": 8_651_620,
    "dropout_schedule": [0.1, 0.1, 0.2, 0.2, 0.3],
}


def _check_tf():
    """Raise ImportError with a helpful message if TensorFlow is not installed."""
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        raise ImportError(
            "TensorFlow is required for model building. "
            "Install it with: pip install ml4seeding[ml]"
        ) from None


def conv_block(inputs, n_filters=32, dropout_prob=0.0, max_pooling=True):
    """Create a convolutional block for the contracting (encoder) path.

    Each block: Conv2D -> BatchNorm -> Conv2D -> BatchNorm -> [Dropout] -> [MaxPool].
    Returns (next_layer, skip_connection).
    """
    _check_tf()
    import tensorflow as tf
    from tensorflow.keras.layers import (
        BatchNormalization,
        Conv2D,
        Dropout,
        MaxPooling2D,
    )

    conv = Conv2D(
        n_filters, (3, 3), activation="relu", padding="same",
        kernel_initializer=tf.keras.initializers.HeNormal(),
    )(inputs)
    conv = BatchNormalization()(conv)

    conv = Conv2D(
        n_filters, (3, 3), activation="relu", padding="same",
        kernel_initializer=tf.keras.initializers.HeNormal(),
    )(conv)
    conv = BatchNormalization()(conv)

    if dropout_prob > 0:
        conv = Dropout(dropout_prob)(conv)

    skip_connection = conv

    if max_pooling:
        next_layer = MaxPooling2D(pool_size=(2, 2))(conv)
    else:
        next_layer = conv

    return next_layer, skip_connection


def upsampling_block(expansive_input, contractive_input, n_filters=32):
    """Create an upsampling block for the expanding (decoder) path.

    Each block: Conv2DTranspose -> Concatenate(skip) -> Conv2D -> BatchNorm ->
    Conv2D -> BatchNorm.
    """
    _check_tf()
    import tensorflow as tf
    from tensorflow.keras.layers import (
        BatchNormalization,
        Conv2D,
        Conv2DTranspose,
        concatenate,
    )

    up = Conv2DTranspose(n_filters, (3, 3), strides=(2, 2), padding="same")(expansive_input)
    merge = concatenate([up, contractive_input], axis=-1)

    conv = Conv2D(
        n_filters, (3, 3), activation="relu", padding="same",
        kernel_initializer=tf.keras.initializers.HeNormal(),
    )(merge)
    conv = BatchNormalization()(conv)

    conv = Conv2D(
        n_filters, (3, 3), activation="relu", padding="same",
        kernel_initializer=tf.keras.initializers.HeNormal(),
    )(conv)
    conv = BatchNormalization()(conv)

    return conv


def unet_model(input_size=(512, 512, 3), n_filters=32, n_classes=4):
    """Build the U-Net model for semantic segmentation.

    Parameters
    ----------
    input_size : tuple
        Input image shape (height, width, channels).
    n_filters : int
        Number of base filters (doubled at each encoder level).
    n_classes : int
        Number of output classes.

    Returns
    -------
    tf.keras.Model
        Compiled U-Net model.
    """
    _check_tf()
    import tensorflow as tf
    from tensorflow.keras.layers import Conv2D, Input

    inputs = Input(shape=input_size)

    # Contracting path
    cblock1 = conv_block(inputs, n_filters, dropout_prob=0.1, max_pooling=True)
    cblock2 = conv_block(cblock1[0], 2 * n_filters, dropout_prob=0.1, max_pooling=True)
    cblock3 = conv_block(cblock2[0], 4 * n_filters, dropout_prob=0.2, max_pooling=True)
    cblock4 = conv_block(cblock3[0], 8 * n_filters, dropout_prob=0.2, max_pooling=True)
    cblock5 = conv_block(cblock4[0], 16 * n_filters, dropout_prob=0.3, max_pooling=False)

    # Expanding path
    ublock6 = upsampling_block(cblock5[0], cblock4[1], 8 * n_filters)
    ublock7 = upsampling_block(ublock6, cblock3[1], 4 * n_filters)
    ublock8 = upsampling_block(ublock7, cblock2[1], 2 * n_filters)
    ublock9 = upsampling_block(ublock8, cblock1[1], 1 * n_filters)

    conv9 = Conv2D(
        n_filters, (3, 3), activation="relu", padding="same",
        kernel_initializer="he_normal",
    )(ublock9)

    outputs = Conv2D(n_classes, (1, 1), activation="softmax", padding="same")(conv9)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
