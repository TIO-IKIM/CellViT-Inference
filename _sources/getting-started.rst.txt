Getting Started
===============

This package provides the inference code to run CellViT/CellViT++ models, designed for easier setup and usage. It does not include training code or any components unrelated to inference. It is recommended to use this package in a virtual environment (e.g., conda or venv) to avoid package conflicts.

Prerequisites
-------------

Before installing the package, ensure that the following prerequisites are met as listed below. Most of the times, they are already installed on your system. Just give it a try and check out CellViT. If you encounter any issues, please refer to the installation instructions for the respective library.

.. collapse:: Required binaries

    * **libvips** - Image processing library
    * **openslide** - Whole slide image library
    * **gcc/g++** - C/C++ compilers
    * **libopencv-core-dev** - OpenCV core development files
    * **libopencv-imgproc-dev** - OpenCV image processing modules
    * **libsnappy-dev** - Compression library
    * **libgeos-dev** - Geometry engine library
    * **llvm** - Compiler infrastructure
    * **libjpeg-dev** - JPEG image format library
    * **libpng-dev** - PNG image format library
    * **libtiff-dev** - TIFF image format library

    On Linux-based systems, you can install these using:

    .. code-block:: bash

        sudo apt-get install libvips openslide gcc g++ libopencv-core-dev libopencv-imgproc-dev libsnappy-dev libgeos-dev llvm libjpeg-dev libpng-dev libtiff-dev

    For other systems, please consult the relevant package management documentation.


Installation
------------

Required
^^^^^^^^

To install the package, follow these steps:

1. Ensure that all prerequisites are installed as outlined above.
2. Verify that OpenSlide (https://openslide.org/) is installed and accessible. If using conda, you can install it with:

    .. code-block:: bash

        conda install -c conda-forge openslide

3. Install PyTorch for your system by following the instructions at https://pytorch.org/get-started/locally/. Ensure that PyTorch >= 2.0 is installed. To view available versions and their corresponding CUDA versions, visit https://pytorch.org/get-started/previous-versions/.
   CellViT-Inference has been tested with PyTorch 2.2.2 and CUDA 12.1. To install the recommended versions, run:

    .. code-block:: bash

        pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

4. Install the CellViT-Inference package using pip:

    .. code-block:: bash

        pip install cellvit

Optional
^^^^^^^^

To enable hardware-accelerated libraries, you can install the following optional dependencies:

1. CuPy (CUDA accelerated NumPy): https://cupy.dev/
2. cuCIM (RAPIDS cuCIM library): https://github.com/rapidsai/cucim

Check
^^^^^

To verify a successful installation, run the following command:

.. code-block:: bash

    cellvit-check

The output should confirm that all required libraries are installed and accessible. If any libraries are missing, refer to the installation instructions for the respective library. This command will also check for optional dependencies and will print a warning if they are not installed. Installing these optional libraries is **not required**.

If using a virtual environment, ensure it is activated before running the command.

Basic Usage
-----------

This package is designed as a command-line tool. Configuration can be provided either directly via the CellViT CLI or by using a configuration file. The configuration file is a YAML file containing the settings for the inference pipeline.
The main script is located in the `cellvit` module, and can be run using the following command:

.. code-block:: bash

    cellvit-inference

You then have to either specify a configuration file

.. code-block:: bash

    cellvit-inference --config <path_to_config_file>

or provide the required parameters directly in the command line. To list all available parameters, run:

.. code-block:: bash

    cellvit-inference --help

You can select to run inference for one slide only or for a batch of slides. For more information, please refer to the :doc:`Usage <usage>` section.

Configuration
-------------

The `caching-directory` is used to store model weights, requiring at least 3GB of free space. By default, this is set to ``~/.cache/cellvit``, but it can be changed by setting the environment variable ``CELLVIT_CACHE`` to a desired path. Remember to set this variable before running the command.

.. list-table::
   :header-rows: 1

   * - Variable
     - Description
   * - ``CELLVIT_CACHE``
     - Path to the caching directory. Default is ``~/.cache/cellvit``.


Next Steps
----------
- :doc:`Usage <usage>`
- :doc:`Examples <examples>`
- :doc:`API Reference <api-reference>`
