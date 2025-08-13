Usage
=====

Basic Usage
-----------

.. note::
  To load images, we relly on the `OpenSlide <https://openslide.org/>`_ library. Please only use images that are supported by OpenSlide.
  If you encounter any issues with loading images, please check the OpenSlide documentation for supported formats.
  If you are using a virtual environment, ensure it is activated before running the command.
  If you want to use ``.tiff`` or other plain images that are not supported by OpenSlide, install `libvips <https://www.libvips.org/>`_  and convert them to pyramid format.
  For example, you can use the following command to convert a plain image to pyramid format:

  .. code-block:: bash

    vips tiffsave input.tiff output.tiff --tile --tile-width 256 --tile-height 256 --pyramid --compression jpeg --Q 90 --vips-progress

  This will create a pyramid image that can be used with CellViT.

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

You can select to run inference for one slide only or for a batch of slides.

Configuration Options
---------------------

The configuration options are divided into several sections, each with its own purpose. Below is a summary of the main sections and their parameters. They appear in the `.yaml` file as well as in the CLI.

.. danger::
  Always one of the process_wsi or process_dataset options must be selected. They are mutually exclusive.

.. list-table::
   :header-rows: 1

   * - Section
     - Name
     - Description
     - Type
     - Default
     - Required
     - Further Information
   * - General
     -
     -
     -
     -
     -
     -

   * -
     - model
     - | Segmentation model to use
       | Choices: ["SAM", "HIPT"]
     - str
     - ➖
     - ✔️
     -
   * -
     - nuclei_taxonomy
     - | Defines the nuclei classification taxonomy
       | Choices: ["binary", "pannuke", "consep", "lizard", "midog", "nucls_main", "nucls_super", "ocelot", "panoptils"]
     - str
     - "pannuke"
     - ➖
     -

   * - Inference
     -
     -
     -
     -
     -
     -

   * -
     - gpu
     - GPU ID to use for inference
     - int
     - 0
     - ➖
     -
   * -
     - enforce_amp
     - Whether to use Automatic Mixed Precision (AMP) for inference
     - bool
     - false
     - ➖
     -
   * -
     - batch_size
     - Number of images (1024 x 1024 patches) processed per batch
     - int
     - 8
     - ➖
     -

   * - Output Settings
     -
     -
     -
     -
     -
     -

   * -
     - outdir
     - Path to the output directory where results will be stored
     - str
     - ➖
     - ✔️
     -
   * -
     - geojson
     - Whether to export results in GeoJSON format (for QuPath or other tools)
     - bool
     - false
     - ➖
     -
   * -
     - graph
     - Whether to generate a cell graph representation
     - bool
     - false
     - ➖
     -
   * -
     - compression
     - Whether to use Snappy compression for output files
     - bool
     - false
     - ➖
     -

   * - System
     -
     -
     -
     -
     -
     -

   * -
     - cpu_count
     - Number of CPU cores to use for inference
     - int
     - System configuration
     - ➖
     -
   * -
     - ray_worker
     - Number of ray worker to use for inference (limited by cpu-count)
     - int
     - System configuration
     - ➖
     -
   * -
     - ray_remote_cpus
     - Number of CPUs per ray worker
     - int
     - System configuration
     - ➖
     -
   * -
     - memory
     - RAM in MB to use
     - int
     - System configuration
     - ➖
     -

   * - Debug
     -
     -
     -
     -
     -
     -

   * -
     - debug
     - If debug should be used - this changes to logger level and requires ray[default]. Also outputs segmentation mask of the tissue preprocessing
     - bool
     - false
     - ➖
     -

   * - Processing Mode: Process a Single Whole Slide Image (WSI)
     -
     -
     -
     -
     -
     -

   * -
     - wsi_path
     - Path to the Whole Slide Image (WSI) file
     - str
     - ➖
     - ✔️
     -
   * -
     - wsi_mpp
     - Microns per pixel (spatial resolution of the slide)
     - float
     - Extracted automatically from file (if available)
     - ➖
     -
   * -
     - wsi_magnification
     - Magnification level of the slide (e.g., 40)
     - int
     - Extracted automatically from file (if available)
     - ➖
     -

   * - Processing Mode: Process a Dataset (Multiple WSI Files)
     -
     -
     -
     -
     -
     -

   * -
     - wsi_folder
     - Path to a folder containing multiple WSI files
     - str
     - ➖
     - ✔️ (if wsi_filelist is NOT used)
     -
   * -
     - wsi_filelist
     - Path to a CSV file listing WSI files (must have a 'path' column)
     - str
     - ➖
     - ✔️ (if wsi_folder is NOT used)
     -
   * -
     - wsi_extension
     - File extension of WSI files (used for detection within wsi_folder)
     - str
     - svs
     - ➖
     -
   * -
     - wsi_mpp
     - Microns per pixel (spatial resolution of the slide). Overwrites slide settings and also mpp set in the filelist
     - float
     - Extracted automatically from file (if available)
     - ➖
     -
   * -
     - wsi_magnification
     - Magnification level of the slide (e.g., 40). Overwrites slide settings and also magnification set in the filelist
     - int
     - Extracted automatically from file (if available)
     - ➖
     -



YAML-Configuration
------------------

The configuration file for CellViT Inference is structured in ``.yaml`` format. Below is an example configuration with explanations for each setting.

.. code-block:: yaml

    # ==========================
    # CellViT Inference Config
    # ==========================

    # Model selection (REQUIRED)
    model:                # REQUIRED | str: Segmentation model to use.
                          # Choices: ["SAM", "HIPT"]
    # Nuclei classification taxonomy (OPTIONAL)
    nuclei_taxonomy:      # OPTIONAL | str: Defines the nuclei classification taxonomy.
                          # Choices: ["binary", "pannuke", "consep", "lizard", "midog", "nucls_main", "nucls_super", "ocelot", "panoptils"]
                          # Default: "pannuke"

    # ==========================
    # Inference Settings (OPTIONAL)
    # ==========================
    inference:
      gpu:                # OPTIONAL | int: GPU ID to use for inference.
                          # Default: 0 (use first available GPU)
      enforce_amp:        # OPTIONAL | bool: Whether to use Automatic Mixed Precision (AMP) for inference.
                          # Default: false (disabled)
      batch_size:         # OPTIONAL | int: Number of images (1024 x 1024 patches) processed per batch.
                          # Default: 8

    # ==========================
    # Output Settings
    # ==========================
    output_format:
      outdir:             # REQUIRED | str: Path to the output directory where results will be stored.
      geojson:            # OPTIONAL | bool: Whether to export results in GeoJSON format (for QuPath or other tools).
                          # Default: false (disabled)
      graph:              # OPTIONAL | bool: Whether to generate a cell graph representation.
                          # Default: false (disabled)
      compression:        # OPTIONAL | bool: Whether to use Snappy compression for output files.
                          # Default: false (disabled)

    # ==========================
    # Processing Mode (Choose One)
    # ==========================
    # Either `process_wsi` (for a single image) or `process_dataset` (for multiple images) should be used.

    # --- Process a Single Whole Slide Image (WSI) ---
    process_wsi:
      wsi_path:           # REQUIRED | str: Path to the Whole Slide Image (WSI) file.
      wsi_mpp:            # OPTIONAL | float: Microns per pixel (spatial resolution of the slide).
                          # Default: Extracted automatically from file (if available).
      wsi_magnification:  # OPTIONAL | int: Magnification level of the slide (e.g., 20x, 40x).
                          # Default: Extracted automatically from file (if available).

    # --- Process a Dataset (Multiple WSI Files) ---
    process_dataset:
      wsi_folder:         # REQUIRED (if `wsi_filelist` is NOT used) | str: Path to a folder containing multiple WSI files.
                          # Either `wsi_folder` OR `wsi_filelist` must be provided (not both).
      wsi_extension:      # OPTIONAL | str: File extension of WSI files (used for detection within wsi_folder).
                          # Default: "svs"
      wsi_filelist:       # REQUIRED (if `wsi_folder` is NOT used) | str: Path to a CSV file listing WSI files.
                          # CSV must have a 'path' column, with optional 'wsi_mpp' and 'wsi_magnification' columns.
                          # If 'wsi_mpp' and 'wsi_magnification' are provided, they override global settings.
      wsi_mpp:            # OPTIONAL | float: Microns per pixel (spatial resolution).
                          # Default: Extracted automatically from file (if available).
                          # Can be used with both `wsi_folder` and `wsi_filelist`.
      wsi_magnification:  # OPTIONAL | int: Magnification level of the slides.
                          # Default: Extracted automatically from file (if available).
                          # Can be used with both `wsi_folder` and `wsi_filelist`.

    # ==========================
    # System Settings (OPTIONAL)
    # ==========================
    system:
      cpu_count:          # OPTIONAL | int: Number of CPU cores to use for inference.
                          # Default: Uses system configuration.
      ray_worker:         # OPTIONAL | int: Number of ray workers to use for inference. Limited by cpu_count.
                          # Default: Uses system configuration.
      ray_remote_cpus:    # OPTIONAL | int: Number of CPUs per ray worker.
                          # Default: Uses system configuration.
      memory:             # OPTIONAL | int: RAM in MB to use.
                          # Default: Uses system configuration.

    # ==========================
    # Debug Settings (OPTIONAL)
    # ==========================
    debug:                # OPTIONAL | bool: If debug should be used - this changes to logger level and requires ray[default]
                          # Default: False

Examples for ``.yaml`` configuration files can be found in the :doc:`Examples <examples>` section.

.. note::

    - The configuration file must be in YAML format.
    - Either run a single WSI or a dataset of WSIs, but not both at the same time.
    - The `wsi_path` and `wsi_folder` or `wsi_filelist` parameters are mutually exclusive.
    - The `wsi_mpp` and `wsi_magnification` parameters can be set globally or per WSI in the file list.
    - The `output_format` section allows you to customize the output format and compression settings.
    - The `system` section allows you to customize the CPU and memory settings for inference.
    - The `debug` section allows you to enable debug mode for more detailed logging.
    - The configuration file can be passed as a command-line argument using the `--config` flag.

CLI-Configuration
-----------------

The CLI configuration allows you to specify the parameters directly in the command line.

General configuration
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: console

    usage: cellvit-inference [-h] [--config CONFIG] [--model {SAM,HIPT}] [--nuclei_taxonomy {binary,pannuke,consep,lizard,midog,nucls_main,nucls_super,ocelot,panoptils}] [--gpu GPU]
                      [--enforce_amp] [--batch_size BATCH_SIZE] [--outdir OUTDIR] [--geojson] [--graph] [--compression] [--cpu_count CPU_COUNT] [--ray_worker RAY_WORKER]
                      [--ray_remote_cpus RAY_REMOTE_CPUS] [--memory MEMORY] [--debug]
                      {process_wsi,process_dataset} ...

    Perform CellViT++ inference

    positional arguments:
      {process_wsi,process_dataset}
                            Select processing mode
        process_wsi         Process a single Whole Slide Image
        process_dataset     Process multiple WSI files

    options:
      -h, --help            show this help message and exit
      --config CONFIG       Path to a YAML configuration file. If provided, CLI arguments are ignored. (default: None)
      --model {SAM,HIPT}    Segmentation model to use (default: None), REQUIRED
      --nuclei_taxonomy {binary,pannuke,consep,lizard,midog,nucls_main,nucls_super,ocelot,panoptils}
                            Defines the nuclei classification taxonomy (default: pannuke), OPTIONAL
      --debug               Enable debug mode (changes logger level and requires ray[default]) (default: False), OPTIONAL

    Inference Settings:
      --gpu GPU             GPU ID to use for inference (default: 0), OPTIONAL
      --enforce_amp         Whether to use Automatic Mixed Precision (AMP) for inference (default: False), OPTIONAL
      --batch_size BATCH_SIZE
                            Number of images processed per batch (default: 8), OPTIONAL

    Output Settings:
      --outdir OUTDIR       Path to the output directory where results will be stored (default: None), REQUIRED
      --geojson             Whether to export results in GeoJSON format (for QuPath or other tools) (default: False), OPTIONAL
      --graph               Whether to generate a cell graph representation (default: False), OPTIONAL
      --compression         Whether to use Snappy compression for output files (default: False), OPTIONAL

    System Settings:
      --cpu_count CPU_COUNT
                            Number of CPU cores to use for inference (default: None), OPTIONAL
      --ray_worker RAY_WORKER
                            Number of ray worker to use for inference (limited by cpu-count) (default: None), OPTIONAL
      --ray_remote_cpus RAY_REMOTE_CPUS
                            Number of CPUs per ray worker (default: None), OPTIONAL
      --memory MEMORY       RAM in MB to use (default: None), OPTIONAL


Process a single image
^^^^^^^^^^^^^^^^^^^^^^

All previous configuration options need to be set before running the command with ``process_wsi``:

.. code-block:: console

    cellvit-inference [previous options] process_wsi [wsi_options]

The ``process_wsi``options are:

.. code-block:: console

    usage: cellvit-inference process_wsi [-h] (--wsi_folder WSI_FOLDER | --wsi_filelist WSI_FILELIST) [--wsi_extension WSI_EXTENSION] [--wsi_mpp WSI_MPP]
                                        [--wsi_magnification WSI_MAGNIFICATION]

    options:
      -h, --help            show this help message and exit
      --wsi_path WSI_PATH   Path to the Whole Slide Image (WSI) file, REQUIRED
      --wsi_mpp WSI_MPP     Microns per pixel (spatial resolution of the slide), OPTIONAL
                            Default: Extracted automatically from file (if available)
      --wsi_magnification WSI_MAGNIFICATION
                            Magnification level of the slide (e.g., 40), OPTIONAL
                            Default: Extracted automatically from file (if available)


Process a dataset
^^^^^^^^^^^^^^^^^

.. code-block:: console

    cellvit-inference [previous options] process_dataset [wsi_options]

The ``process_dataset``options are:

.. code-block:: console

    usage: cellvit-inference process_dataset [-h] (--wsi_folder WSI_FOLDER | --wsi_filelist WSI_FILELIST) [--wsi_extension WSI_EXTENSION] [--wsi_mpp WSI_MPP]
                                        [--wsi_magnification WSI_MAGNIFICATION]

    options:
      -h, --help            show this help message and exit
      --wsi_folder WSI_FOLDER
                            Path to a folder containing multiple WSI files, REQUIRED if wsi_filelist is NOT used
      --wsi_filelist WSI_FILELIST
                            Path to a CSV file listing WSI files (must have a 'path' column), REQUIRED if wsi_folder is NOT used
      --wsi_extension WSI_EXTENSION
                            File extension of WSI files (used for detection within wsi_folder), OPTIONAL
      --wsi_mpp WSI_MPP     Microns per pixel (spatial resolution), OPTIONAL
                            Default: Extracted automatically from file (if available)
                            Can be used with both wsi_folder and wsi_filelist
      --wsi_magnification WSI_MAGNIFICATION
                            Magnification level of the slides, OPTIONAL
                            Default: Extracted automatically from file (if available)
                            Can be used with both wsi_folder and wsi_filelist

.. note::
    - The `wsi_path` and `wsi_folder` or `wsi_filelist` parameters are mutually exclusive.
    - The `wsi_mpp` and `wsi_magnification` parameters can be set globally or per WSI in the file list.
    - The `output_format` section allows you to customize the output format and compression settings.
    - The `system` section allows you to customize the CPU and memory settings for inference.
    - The `debug` section allows you to enable debug mode for more detailed logging.
    - The configuration file can be passed as a command-line argument using the `--config` flag.
    - The `--wsi_folder` option is used to specify a folder containing multiple WSI files.
    - The `--wsi_filelist` option is used to specify a CSV file listing WSI files, even from different folders. Provide the entire WSI-paths in the `path` column.
    - The `--wsi_extension` option is used to specify the file extension of WSI files (e.g., "svs").
