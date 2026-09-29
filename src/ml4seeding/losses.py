"""Loss functions: SCCE, weighted SCCE, Dice, combined.

This module is importable without TensorFlow. The loss functions are only
available when TensorFlow is installed (via the ``ml`` extra).
"""

from __future__ import annotations


def _check_tf():
    """Raise ImportError with a helpful message if TensorFlow is not installed."""
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        raise ImportError(
            "TensorFlow is required for loss functions. "
            "Install it with: pip install ml4seeding[ml]"
        ) from None


def weighted_sparse_cce_loss(class_weights=None):
    """Create a weighted sparse categorical cross-entropy loss function.

    Parameters
    ----------
    class_weights : list[float] or None
        Per-class weights. Default: [0.2, 1.0, 3.0, 0.5] for
        [background, good, fair, poor].

    Returns
    -------
    Callable loss function (y_true, y_pred) -> scalar.
    """
    _check_tf()
    import tensorflow as tf

    if class_weights is None:
        class_weights = [0.2, 1.0, 3.0, 0.5]
    weights = tf.constant(class_weights, dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        if len(y_true.shape) == 4 and y_true.shape[-1] == 1:
            y_true = tf.squeeze(y_true, axis=-1)
        y_true = tf.cast(y_true, tf.int32)

        scce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        pixel_weights = tf.gather(weights, y_true)
        return tf.reduce_mean(scce * pixel_weights)

    return loss_fn


def dice_loss_multiclass(num_classes=4, smooth=1e-6):
    """Create a multiclass Dice loss function (excluding background).

    Parameters
    ----------
    num_classes : int
        Number of classes.
    smooth : float
        Smoothing constant.

    Returns
    -------
    Callable loss function (y_true, y_pred) -> scalar.
    """
    _check_tf()
    import tensorflow as tf

    def loss_fn(y_true, y_pred):
        if len(y_true.shape) == 4 and y_true.shape[-1] == 1:
            y_true = tf.squeeze(y_true, axis=-1)
        y_true = tf.cast(y_true, tf.int32)

        y_true_onehot = tf.one_hot(y_true, depth=num_classes, dtype=tf.float32)

        # Exclude background
        y_true_fg = y_true_onehot[..., 1:]
        y_pred_fg = y_pred[..., 1:]

        intersection = tf.reduce_sum(y_true_fg * y_pred_fg, axis=[0, 1, 2])
        union = tf.reduce_sum(y_true_fg + y_pred_fg, axis=[0, 1, 2])

        dice_per_class = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - tf.reduce_mean(dice_per_class)

    return loss_fn


def combined_loss(class_weights=None, num_classes=4):
    """Create a combined weighted SCCE + Dice loss function.

    Parameters
    ----------
    class_weights : list[float] or None
        Per-class weights for the SCCE component.
    num_classes : int
        Number of classes.

    Returns
    -------
    Callable loss function (y_true, y_pred) -> scalar.
    """
    _check_tf()

    scce_fn = weighted_sparse_cce_loss(class_weights)
    dice_fn = dice_loss_multiclass(num_classes)

    def loss_fn(y_true, y_pred):
        return scce_fn(y_true, y_pred) + dice_fn(y_true, y_pred)

    return loss_fn
