"""Test package import and version."""


def test_import():
    import ml4seeding

    assert ml4seeding.__version__ == "0.1.0a1"


def test_cli_import():
    from ml4seeding.cli import app

    assert app is not None
