Data Management
===============

This guide explains how ``ml4seeding`` manages large data files (drone
imagery, annotation masks, model weights) with DataLad, and how to set up
the data repository as a collaborator.

The full design record lives in ``planning/data-management.md`` in the
repository.

The DataLad Submodule Pattern
-----------------------------

The project uses a **split-repo pattern**:

1. **Code repo** (`UBC-FRESH/fresh-ml4seeding
   <https://github.com/UBC-FRESH/fresh-ml4seeding>`_): the Python package,
   tests, docs, and CI. No large data files.
2. **Data repo** (`UBC-FRESH/fresh-ml4seeding-data
   <https://github.com/UBC-FRESH/fresh-ml4seeding-data>`_): a DataLad
   dataset using git-annex for large binary files (drone imagery,
   annotation masks, model weights).

The data repo is added to the code repo as a git submodule at
``external/fresh-ml4seeding-data``. The submodule pins an exact version of
the data, so every checkout of the code repo knows which data version it
was developed against, while large file content stays out of the code
repo's git history.

DataLad (with git-annex) stores only small symlink placeholders in git;
the actual binary content is fetched on demand with ``datalad get``.

Setting Up as a Collaborator
----------------------------

Clone the code repo and initialize the data submodule:

.. code-block:: bash

   git clone https://github.com/UBC-FRESH/fresh-ml4seeding.git
   cd fresh-ml4seeding
   git submodule update --init --recursive

Fetch the data files you need. With DataLad installed:

.. code-block:: bash

   datalad get -r external/fresh-ml4seeding-data/data

Or fetch only a specific subset (for example, the tiles and masks needed
for training):

.. code-block:: bash

   datalad get external/fresh-ml4seeding-data/data/tiles \
               external/fresh-ml4seeding-data/data/masks

If the dataset uses a git-annex special remote (for example, S3 storage),
enable it first:

.. code-block:: bash

   git -C external/fresh-ml4seeding-data annex enableremote arbutus-s3
   datalad get -r external/fresh-ml4seeding-data/data

.. note::

   The test suite and CI do not require the data submodule. Tracked tests
   use small synthetic fixtures committed to the code repo.

Data Directory Structure
------------------------

Inside the data repo, files are organized under ``data/`` as follows:

.. list-table::
   :header-rows: 1

   * - Directory
     - Contents
   * - ``data/raw_drone_images/``
     - Raw DJI JPEG files with EXIF GPS metadata
   * - ``data/orthomosaics/``
     - GeoTIFF orthomosaics (A10, A58, G15 sites)
   * - ``data/tiles/``
     - 512×512 PNG image tiles
   * - ``data/masks/``
     - 4-class segmentation annotation masks
   * - ``data/models/``
     - Trained U-Net model checkpoints
   * - ``data/predictions/``
     - Model prediction outputs

From the code repo, these paths are reached through the submodule, for
example ``external/fresh-ml4seeding-data/data/orthomosaics/``.

Environment Variable Configuration
----------------------------------

Point the ``ML4SEEDING_EXTERNAL_DATA_ROOT`` environment variable at the
``data/`` directory inside the submodule so scripts and notebooks can
locate the data without hard-coding machine-specific paths:

.. code-block:: bash

   export ML4SEEDING_EXTERNAL_DATA_ROOT=$PWD/external/fresh-ml4seeding-data/data

Add this line to your shell profile (or to a ``.env`` file loaded by your
workflow) to make the setting persistent.

Rules
-----

- Do not commit large data files to the code repo.
- Do not commit model weights or prediction outputs to git directly.
- Use the DataLad data submodule for all binary data.
- Keep ``tmp/``, ``local/``, ``data/private/``, and ``outputs/`` ignored.
- Keep synthetic test fixtures small and self-contained in the code repo.
- Record provenance for every external dataset, model checkpoint, and
  evaluation result.
