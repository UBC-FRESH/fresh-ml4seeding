Installation
============

From PyPI
---------

.. code-block:: bash

   python -m pip install ml4seeding==0.1.0a1

For Development
---------------

.. code-block:: bash

   python -m venv .venv
   . .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -e .[dev]

Machine Learning Dependencies
-----------------------------

The core package has minimal dependencies. For ML training, install the
``ml`` extra:

.. code-block:: bash

   python -m pip install -e .[ml]
