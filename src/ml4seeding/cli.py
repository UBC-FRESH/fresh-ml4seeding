"""ml4seeding command-line interface."""

import typer

from ml4seeding import __version__

app = typer.Typer(
    name="ml4seeding",
    help="Machine learning for precision aerial seeding.",
    add_completion=False,
)


def version_callback(value: bool) -> None:
    if value:
        typer.echo(f"ml4seeding {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-v", callback=version_callback, is_eager=True,
        help="Show version and exit."
    ),
) -> None:
    """Machine learning for precision aerial seeding."""


@app.command()
def info() -> None:
    """Show package information."""
    typer.echo(f"ml4seeding {__version__}")
    typer.echo("Microsite classification from drone imagery for aerial reforestation.")
    typer.echo("Repository: https://github.com/UBC-FRESH/fresh-ml4seeding")
