============
Installation
============

A Python >3.6 installation is required. The recommended way to install Python, ``pyfastspm``, 
and all its dependencies is via the `mambaforge <https://github.com/conda-forge/miniforge#mambaforge>`_ 
distribution. In fact, using the ``mamba`` package management system with the `conda-forge <https://conda-forge.org/>`_ package distribution channel (``mambaforge`` is already configured for this) allows to automatically install all dependencies on any platform, particularly ``ffmpeg`` which is required for exporting movies with ``pyfastspm``.

Using mambaforge/conda
======================
First of all, it is good practice to create an enviroment specific for the ``pyfastspm`` 
package, by running the following command:

.. code-block:: bash

    $ mamba create -n pyfastspm

and activate it with (notice the use of the ``conda`` command rather than the ``mamba`` command):[1]_

.. code-block:: bash

    $ conda activate pyfastspm

After that you can install the package and all its dependencies with the following command:

.. code-block:: bash

    mamba install pyfastspm

.. [1] See the `mamba documentation <https://mamba.readthedocs.io/en/latest/user_guide/mamba.html#mamba-vs-conda-clis>`_.

Using setuptools
================
You can also install the package from the PyPI repository by issuing the usual command:

.. code-block:: bash

    pip install pyfastspm

The required dependencies should be all automatically installed except for ``ffmpeg`` that 
needs to me manually installed and made available system-wide via the ``PATH`` enviroment variable.
