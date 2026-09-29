API Reference
=============

This page documents the public Python API of ``ml4seeding``.

The core modules (orthomosaic analysis, pseudo-orthomosaic generation,
tiling, grid overlays, preprocessing, and metrics) depend only on the
base installation. The machine learning modules (``models``, ``losses``,
and ``training``) are importable without TensorFlow, but their functions
require the ``ml`` extra (see :doc:`installation`).

Image Analysis
--------------

orthomosaic
~~~~~~~~~~~

.. automodule:: ml4seeding.orthomosaic
   :members:
   :undoc-members:

pseudo_ortho
~~~~~~~~~~~~

.. automodule:: ml4seeding.pseudo_ortho
   :members:
   :undoc-members:

tiling
~~~~~~

.. automodule:: ml4seeding.tiling
   :members:
   :undoc-members:

grid
~~~~

.. automodule:: ml4seeding.grid
   :members:
   :undoc-members:

Dataset Preparation
-------------------

preprocessing
~~~~~~~~~~~~~

.. automodule:: ml4seeding.preprocessing
   :members:
   :undoc-members:

metrics
~~~~~~~

.. automodule:: ml4seeding.metrics
   :members:
   :undoc-members:

Machine Learning
----------------

These modules require TensorFlow, available via the ``ml`` extra.

models
~~~~~~

.. automodule:: ml4seeding.models
   :members:
   :undoc-members:

losses
~~~~~~

.. automodule:: ml4seeding.losses
   :members:
   :undoc-members:

training
~~~~~~~~

.. automodule:: ml4seeding.training
   :members:
   :undoc-members:

Command-Line Interface
----------------------

cli
~~~

.. automodule:: ml4seeding.cli
   :members:
   :undoc-members:
