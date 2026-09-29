Examples
========

This page shows how to use ``ml4seeding`` from the command line and points
to the example notebook.

Command-Line Interface
----------------------

Installing the package provides the ``ml4seeding`` command. Run
``ml4seeding --help`` for the full command list.

Package information
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   ml4seeding --version
   ml4seeding info

Analyze an orthomosaic
~~~~~~~~~~~~~~~~~~~~~~

Extract metadata, assess image quality, compute vegetation indices, and
measure sharpness for an RGB orthomosaic GeoTIFF:

.. code-block:: bash

   ml4seeding ortho analyze path/to/orthomosaic.tif \
       --out-dir ortho_report --samples 30 --window 1024

Build a pseudo-orthomosaic
~~~~~~~~~~~~~~~~~~~~~~~~~~

Stitch raw DJI drone images into a georeferenced pseudo-orthomosaic using
GPS positions from EXIF metadata:

.. code-block:: bash

   ml4seeding pseudo-ortho build path/to/dji_images/ \
       --out-dir pseudo_ortho --downscale 0.20 --crs EPSG:32610

Tile a GeoTIFF
~~~~~~~~~~~~~~

Cut a GeoTIFF into fixed-size PNG tiles with georeferenced JSON metadata,
skipping tiles with too many invalid pixels:

.. code-block:: bash

   ml4seeding tile create path/to/orthomosaic.tif \
       --out-dir tiles --size 512 --overlap 128 --skip 0.70

Add a grid overlay
~~~~~~~~~~~~~~~~~~

Overlay a camouflage grid on tiles for annotation quality control. Grid
spacing is given in centimetres and converted to pixels using the ground
sample distance:

.. code-block:: bash

   ml4seeding grid add tiles/ tiles/ \
       --grid-cm 10 --pixel-size 0.0075

Python API
----------

All CLI commands are thin wrappers over the Python API. For example,
tiling a GeoTIFF from Python:

.. code-block:: python

   from ml4seeding.tiling import tile_geotiff

   tiles = tile_geotiff(
       "path/to/orthomosaic.tif",
       "tiles",
       tile_size=512,
       overlap=128,
       skip_invalid_fraction=0.70,
   )
   print(f"Saved {len(tiles)} tiles")

See the :doc:`api` for the complete reference.

Example Notebook
----------------

An example notebook demonstrating the package end to end will be added to
the ``examples/`` directory in the repository (roadmap task P4.1). It is
based on the prototype pipeline notebook
``examples/01_1_build_pseudo_ortho_a10_seg1.ipynb`` from
`UBC-FRESH/ML4seeding <https://github.com/UBC-FRESH/ML4seeding>`_, which
the :mod:`ml4seeding.pseudo_ortho` module re-implements as package code.
