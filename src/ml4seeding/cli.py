"""ml4seeding command-line interface."""

from pathlib import Path

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


# --- Orthomosaic commands ---

ortho_app = typer.Typer(help="Orthomosaic analysis commands.")
app.add_typer(ortho_app, name="ortho")


@ortho_app.command("analyze")
def ortho_analyze(
    ortho_path: Path = typer.Argument(..., help="Path to orthomosaic GeoTIFF."),
    out_dir: Path = typer.Option(
        Path("./ortho_report"), "--out-dir", "-o", help="Output directory."
    ),
    n_samples: int = typer.Option(30, "--samples", "-n", help="Number of random sample windows."),
    win_size: int = typer.Option(1024, "--window", "-w", help="Sample window size in pixels."),
) -> None:
    """Analyze an orthomosaic: metadata, quality, vegetation indices, sharpness."""
    from ml4seeding.orthomosaic import analyze_rgb_ortho

    typer.echo(f"Analyzing: {ortho_path}")
    report = analyze_rgb_ortho(ortho_path, out_dir, n_samples=n_samples, win_size=win_size)
    meta = report["metadata"]
    typer.echo(f"  Size: {meta['width_px']}x{meta['height_px']} px")
    typer.echo(f"  CRS: {meta['crs']}")
    typer.echo(f"  GSD: {meta['pixel_size_x'] * 100:.2f} cm/px")
    typer.echo(f"  Area: {meta['coverage_area_ha']:.2f} ha")
    typer.echo(f"  Report: {out_dir}")


# --- Pseudo-orthomosaic commands ---

pseudo_app = typer.Typer(help="Pseudo-orthomosaic generation commands.")
app.add_typer(pseudo_app, name="pseudo-ortho")


@pseudo_app.command("build")
def pseudo_build(
    input_dir: Path = typer.Argument(..., help="Directory containing DJI JPEG images."),
    out_dir: Path = typer.Option(
        Path("./pseudo_ortho"), "--out-dir", "-o", help="Output directory."
    ),
    downscale: float = typer.Option(0.20, "--downscale", "-d", help="Image downscale factor."),
    crs: str = typer.Option("EPSG:32610", "--crs", help="Output coordinate reference system."),
) -> None:
    """Build a pseudo-orthomosaic from raw drone images using GPS stitching."""
    from ml4seeding.pseudo_ortho import build_pseudo_ortho_from_dir

    typer.echo(f"Building pseudo-orthomosaic from: {input_dir}")
    build_pseudo_ortho_from_dir(input_dir, out_dir, downscale=downscale, crs=crs)
    typer.echo(f"  Output: {out_dir}")


# --- Tiling commands ---

tile_app = typer.Typer(help="Image tiling commands.")
app.add_typer(tile_app, name="tile")


@tile_app.command("create")
def tile_create(
    image_path: Path = typer.Argument(..., help="Path to GeoTIFF to tile."),
    out_dir: Path = typer.Option(Path("./tiles"), "--out-dir", "-o", help="Output directory."),
    tile_size: int = typer.Option(512, "--size", "-s", help="Tile size in pixels."),
    overlap: int = typer.Option(128, "--overlap", help="Tile overlap in pixels."),
    skip_invalid: float = typer.Option(
        0.70, "--skip", help="Skip tiles with this fraction of invalid pixels."
    ),
) -> None:
    """Tile a GeoTIFF into fixed-size PNG tiles with metadata."""
    from ml4seeding.tiling import tile_geotiff

    typer.echo(f"Tiling: {image_path}")
    tiles = tile_geotiff(
        image_path, out_dir, tile_size=tile_size, overlap=overlap,
        skip_invalid_fraction=skip_invalid,
    )
    typer.echo(f"  Saved {len(tiles)} tiles to: {out_dir}")


# --- Grid commands ---

grid_app = typer.Typer(help="Grid overlay commands.")
app.add_typer(grid_app, name="grid")


@grid_app.command("add")
def grid_add(
    tiles_dir: Path = typer.Argument(..., help="Directory containing PNG tiles."),
    meta_dir: Path = typer.Argument(..., help="Directory containing JSON metadata."),
    out_dir: Path = typer.Option(None, "--out-dir", "-o", help="Output directory."),
    grid_cm: float = typer.Option(10.0, "--grid-cm", help="Grid spacing in cm."),
    pixel_size_m: float = typer.Option(0.0075, "--pixel-size", help="Pixel size in meters."),
) -> None:
    """Add camouflage grid overlay to tiles for annotation QC."""
    from ml4seeding.grid import add_grid_to_tiles

    if out_dir is None:
        out_dir = tiles_dir.parent / f"{tiles_dir.name}_grid{int(grid_cm)}cm"

    typer.echo(f"Adding {grid_cm}cm grid to tiles in: {tiles_dir}")
    saved, skipped = add_grid_to_tiles(
        tiles_dir, meta_dir, out_dir, grid_cm=grid_cm, pixel_size_m=pixel_size_m,
    )
    typer.echo(f"  Saved: {saved}, Skipped: {skipped}")
    typer.echo(f"  Output: {out_dir}")
