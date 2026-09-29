"""Test model architecture constants and import safety."""

from ml4seeding.models import ARCHITECTURE


def test_architecture_constants():
    assert ARCHITECTURE["input_shape"] == (512, 512, 3)
    assert ARCHITECTURE["base_filters"] == 32
    assert ARCHITECTURE["num_classes"] == 4
    assert ARCHITECTURE["total_params"] == 8_651_620


def test_unet_model_import_without_tf():
    """Model module should be importable without TensorFlow."""
    from ml4seeding.models import unet_model

    # Should raise ImportError when called without TF
    try:
        unet_model()
        # If TF is installed, this will succeed — that's fine
    except ImportError as e:
        assert "TensorFlow is required" in str(e)


def test_losses_import_without_tf():
    """Losses module should be importable without TensorFlow."""
    from ml4seeding.losses import combined_loss

    try:
        combined_loss()
    except ImportError as e:
        assert "TensorFlow is required" in str(e)


def test_training_import_without_tf():
    """Training module should be importable without TensorFlow."""
    from ml4seeding.training import process_path

    try:
        process_path("x.png", "y.png")
    except ImportError as e:
        assert "TensorFlow is required" in str(e)
