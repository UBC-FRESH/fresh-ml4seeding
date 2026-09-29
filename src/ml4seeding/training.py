"""Training pipeline with configurable augmentation, loss, and early stopping.

This module is importable without TensorFlow. The training functions are only
available when TensorFlow is installed (via the ``ml`` extra).
"""

from __future__ import annotations


def _check_tf():
    """Raise ImportError with a helpful message if TensorFlow is not installed."""
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        raise ImportError(
            "TensorFlow is required for training. "
            "Install it with: pip install ml4seeding[ml]"
        ) from None


def augment_geometric(image, mask):
    """Apply geometric augmentation: random horizontal and vertical flips.

    Parameters
    ----------
    image : tf.Tensor
        Input image (H, W, C).
    mask : tf.Tensor
        Input mask (H, W, 1).

    Returns
    -------
    Augmented (image, mask) tuple.
    """
    _check_tf()
    import tensorflow as tf

    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)

    return image, mask


def augment_photometric(image, mask, brightness_delta=0.10, contrast_range=(0.90, 1.10)):
    """Apply photometric + geometric augmentation.

    Parameters
    ----------
    image : tf.Tensor
        Input image (H, W, C) in [0, 1] range.
    mask : tf.Tensor
        Input mask (H, W, 1).
    brightness_delta : float
        Maximum brightness change.
    contrast_range : tuple
        (lower, upper) contrast multiplier range.

    Returns
    -------
    Augmented (image, mask) tuple.
    """
    _check_tf()
    import tensorflow as tf

    image, mask = augment_geometric(image, mask)

    image = tf.image.random_brightness(image, max_delta=brightness_delta)
    image = tf.image.random_contrast(image, lower=contrast_range[0], upper=contrast_range[1])
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, mask


def process_path(image_path, mask_path):
    """Load and preprocess an image-mask pair from file paths.

    Parameters
    ----------
    image_path : tf.Tensor (string)
        Path to the image file.
    mask_path : tf.Tensor (string)
        Path to the mask file.

    Returns
    -------
    (image, mask) tuple with image as float32 [0,1] and mask as uint8.
    """
    _check_tf()
    import tensorflow as tf

    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.cast(mask, tf.uint8)

    return img, mask


def preprocess_tile(image, mask, target_size=(512, 512)):
    """Resize image and mask to target size.

    Uses bilinear interpolation for images and nearest-neighbor for masks.
    """
    _check_tf()
    import tensorflow as tf

    input_image = tf.image.resize(image, target_size, method="bilinear")
    input_mask = tf.image.resize(mask, target_size, method="nearest")
    return input_image, input_mask
