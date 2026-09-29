"""Test CLI commands."""

from typer.testing import CliRunner

from ml4seeding.cli import app

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0a1" in result.output


def test_info():
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "ml4seeding" in result.output
