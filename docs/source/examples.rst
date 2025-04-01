Examples
========

Below we provide examples for the most common use cases of the CellViT package. The examples are provided in both CLI and YAML format. The CLI examples can be run directly in your terminal, while the YAML examples can be used to create a configuration file and then run with the `cellvit-inference --config /path/to/your/config.yaml` command.

Test Database
~~~~~~~~~~~~~

.. admonition:: Download test database into current directory
    :class: example

    .. code-block:: bash

        download-test-database # run in your terminal

This command will download a test database into the current directory. The database is used for testing purposes and contains sample data to demonstrate the functionality of the package.
The database is not required for the package to function, but it can be useful for testing and development purposes.

.. note::
    Contains sample data in these folders:

    - ``x40_svs/``: High-mag WSIs (.svs format)
    - ``x20_svs/``: Low-mag WSIs (.svs format)
    - ``BRACS/``: Breast cancer WSIs (.tiff format)
    - ``Philips/``: Alternative scanner format

Basic Examples
~~~~~~~~~~~~~~

Basic run for the CellViT-HIPT-Backbone (ViT-S)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. grid:: 1 2 2 2
    :gutter: 3

    .. grid-item-card:: CLI
        :class-header: bg-purple

        .. code-block:: bash

            cellvit-inference \
                --model HIPT \
                --outdir ./test_results/x40_svs/minimal/HIPT \
                process_wsi \
                --wsi_path ./test_database/x40_svs/JP2K-33003-2.svs


    .. grid-item-card:: YAML
        :class-header: bg-deep-purple

        .. code-block:: yaml

            model: HIPT
            output_format:
                outdir: ./test_results/x40_svs/minimal/HIPT
            process_wsi:
                wsi_path: ./test_database/x40_svs/JP2K-33003-2.svs


Basic run for the CellViT-SAM-H-Backbone (ViT-H)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. grid:: 1 2 2 2
    :gutter: 3

    .. grid-item-card:: CLI
        :class-header: bg-purple

        .. code-block:: bash

            cellvit-inference \
                --model SAM \
                --outdir ./test_results/x40_svs/minimal/SAM \
                process_wsi \
                --wsi_path ./test_database/x40_svs/JP2K-33003-2.svs



    .. grid-item-card:: YAML
        :class-header: bg-deep-purple

        .. code-block:: yaml

            model: SAM
            output_format:
                outdir: ./test_results/x40_svs/minimal/SAM
            process_wsi:
                wsi_path: ./test_database/x40_svs/JP2K-33003-2.svs

Output Customization
~~~~~~~~~~~~~~~~~~~~

Binary Classification
^^^^^^^^^^^^^^^^^^^^^

.. grid:: 1 2 2 2
    :gutter: 3

    .. grid-item-card:: CLI
        :class-header: bg-purple

        .. code-block:: bash

            cellvit-inference \
                --model SAM \
                --nuclei_taxonomy binary \
                --outdir ./test_results/x40_svs/binary/SAM \
                process_wsi \
                --wsi_path ./test_database/x40_svs/JP2K-33003-2.svs



    .. grid-item-card:: YAML
        :class-header: bg-deep-purple

        .. code-block:: yaml

            model: SAM
            nuclei_taxonomy: binary
            output_format:
                outdir: ./test_results/x40_svs/binary/SAM
            process_wsi:
                wsi_path: ./test_database/x40_svs/JP2K-33003-2.svs


QuPath-Compatible GeoJSON
^^^^^^^^^^^^^^^^^^^^^^^^^

.. grid:: 1 2 2 2
    :gutter: 3

    .. grid-item-card:: CLI
        :class-header: bg-purple

        .. code-block:: bash

            cellvit-inference \
                --model SAM \
                --outdir ./test_results/x40_svs/geojson/SAM \
                --geojson \
                process_wsi \
                --wsi_path ./test_database/x40_svs/JP2K-33003-2.svs



    .. grid-item-card:: YAML
        :class-header: bg-deep-purple

        .. code-block:: yaml

            model: SAM
            output_format:
                outdir: ./test_results/x40_svs/geojson/SAM
                geojson: true
            process_wsi:
                wsi_path: ./test_database/x40_svs/JP2K-33003-2.svs


Advanced Output Formats
^^^^^^^^^^^^^^^^^^^^^^^

All types of output formats can be combined. This example provides the output as a `.geojson` file, a cell graph as `.pt` file and additionally applies snappy compression to the output files:

.. grid:: 1 2 2 2
    :gutter: 3

    .. grid-item-card:: CLI
        :class-header: bg-purple

        .. code-block:: bash

            cellvit-inference \
                --model SAM \
                --outdir ./test_results/x40_svs/compression_graph_geojson/SAM \
                --geojson \
                --graph \
                --compression \
                process_wsi \
                --wsi_path ./test_database/x40_svs/JP2K-33003-2.svs


    .. grid-item-card:: YAML
        :class-header: bg-deep-purple

        .. code-block:: yaml

            model: SAM
            output_format:
                outdir: ./test_results/x40_svs/compression_graph_geojson/SAM
                geojson: true
                graph: true
                compression: true
            process_wsi:
                wsi_path: ./test_database/x40_svs/JP2K-33003-2.svs



Debugging Log-Level
~~~~~~~~~~~~~~~~~~~~

To see debug messages if errors occur and inspect the tissue detection, you can use the debug flag. This will create a folder with the name of the WSI in the output directory and save the tissue detection results there. Also this comes with improved log messages.


.. grid:: 1 2 2 2
    :gutter: 3

    .. grid-item-card:: CLI
        :class-header: bg-purple

        .. code-block:: bash

            cellvit-inference \
                --model SAM \
                --outdir ./test_results/x40_svs/minimal/SAM \
                --debug
                process_wsi \
                --wsi_path ./test_database/x40_svs/JP2K-33003-2.svs



    .. grid-item-card:: YAML
        :class-header: bg-deep-purple

        .. code-block:: yaml

            model: SAM
            output_format:
                outdir: ./test_results/x40_svs/minimal/SAM
            process_wsi:
                wsi_path: ./test_database/x40_svs/JP2K-33003-2.svs
            debug: true

Metadata Handover
~~~~~~~~~~~~~~~~~

With this approach the metadata can either be provided if not available from the WSI (e.g., some tiff-formats do not include metadata) or overwrite existing metadata. This example is for a single file. For a more sophisticated example see the dataset processing examples given below.

.. grid:: 1 2 2 2
    :gutter: 3

    .. grid-item-card:: CLI
        :class-header: bg-purple

        .. code-block:: bash

            cellvit-inference \
                --model HIPT \
                --outdir ./test_results/BRACS/minimal/HIPT \
                process_wsi \
                --wsi_path ./test_database/BRACS/BRACS_1640_N_3_cropped.tiff \
                --wsi_mpp 0.25 \
                --wsi_magnification 40



    .. grid-item-card:: YAML
        :class-header: bg-deep-purple

        .. code-block:: yaml

            model: HIPT
            output_format:
                outdir: ./test_results/BRACS/minimal/HIPT
            process_wsi:
                wsi_path: ./test_database/BRACS/BRACS_1640_N_3_cropped.tiff
                wsi_mpp: 0.25
                wsi_magnification: 40



Processing Multiple WSI (Entire Dataset)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We provide a simple way to process an entire dataset. We provide two options for this. Either, you can hand over a folder with the WSI files and the program will automatically detect all WSI files in the folder and process them. Or you can provide a list of WSI files in a `.csv`` file.

Folder Processing
^^^^^^^^^^^^^^^^^

Simplest Version (just provide a folder with WSI files in .svs format):

.. grid:: 1 2 2 2
    :gutter: 3

    .. grid-item-card:: CLI
        :class-header: bg-purple

        .. code-block:: bash

            cellvit-inference \
                --model SAM \
                --outdir ./test_results/x40_svs/folder/SAM \
                process_dataset \
                --wsi_folder ./test_database/x40_svs


    .. grid-item-card:: YAML
        :class-header: bg-deep-purple

        .. code-block:: yaml

            model: SAM
            output_format:
                outdir: ./test_results/x40_svs/folder/SAM
            process_dataset:
                wsi_folder: ./test_database/x40_svs

Example with a differing WSI-format (here: .tiff):

.. grid:: 1 2 2 2
    :gutter: 3

    .. grid-item-card:: CLI
        :class-header: bg-purple

        .. code-block:: bash

            cellvit-inference \
                --model SAM \
                --outdir ./test_results/Philips/folder/SAM \
                process_dataset \
                --wsi_folder ./test_database/Philips \
                --wsi_extension tiff

    .. grid-item-card:: YAML
        :class-header: bg-deep-purple

        .. code-block:: yaml

            model: SAM
            output_format:
                outdir: ./test_results/Philips/folder/SAM
            process_dataset:
                wsi_folder: ./test_database/Philips
                wsi_extension: tiff

Handover of metadata (e.g., MPP and magnification):

.. grid:: 1 2 2 2
    :gutter: 3

    .. grid-item-card:: CLI
        :class-header: bg-purple

        .. code-block:: bash

            cellvit-inference \
                --model SAM \
                --outdir ./test_results/BRACS/folder/SAM \
                process_dataset \
                --wsi_folder ./test_database/BRACS \
                --wsi_extension tiff \
                --wsi_mpp 0.25 \
                --wsi_magnification 40

    .. grid-item-card:: YAML
        :class-header: bg-deep-purple

        .. code-block:: yaml

            model: SAM
            output_format:
                outdir: ./test_results/BRACS/folder/SAM
            process_dataset:
                wsi_folder: ./test_database/BRACS
                wsi_extension: tiff
                wsi_mpp: 0.25
                wsi_magnification: 40

CSV Filelist Processing
^^^^^^^^^^^^^^^^^^^^^^^

An example of a CSV-Filelist is given in the test database. The structure is at follows:

.. code-block:: yaml

    path,wsi_mpp,wsi_magnification
    ./test_database/x20_svs/CMU-1-Small-Region.svs,,
    ./test_database/BRACS/BRACS_1640_N_3_cropped.tiff,0.25,40

The filelist must at least have the **path** column. The wsi_mpp and wsi_magnification columns are optional. If they are not provided, the program will try to read them from the WSI metadata.

.. code-block:: yaml

    path
    ./test_database/x20_svs/CMU-1-Small-Region.svs

As can be seen in the example above, it is possible to define the MPP and magnification for each WSI separately and input files with differing extensions.


.. grid:: 1 2 2 2
    :gutter: 3

    .. grid-item-card:: CLI
        :class-header: bg-purple

        .. code-block:: bash

            cellvit-inference \
                --model HIPT \
                --outdir ./test_results/all/HIPT \
                process_dataset \
                --wsi_filelist ./test_database/filelist.csv

    .. grid-item-card:: YAML
        :class-header: bg-deep-purple

        .. code-block:: yaml

            model: HIPT
            output_format:
                outdir: /test_results/all/HIPT
            process_dataset:
                wsi_filelist: ./test_database/filelist.csv

You can also set the wsi_mpp and wsi_magnification globally. Be careful, this will overwrite the values in the CSV file:

.. grid:: 1 2 2 2
    :gutter: 3

    .. grid-item-card:: CLI
        :class-header: bg-purple

        .. code-block:: bash

            cellvit-inference \
                --model HIPT \
                --outdir ./test_results/all/HIPT \
                process_dataset \
                --wsi_filelist ./test_database/filelist.csv \
                --wsi_mpp 0.25 \
                --wsi_magnification 40

    .. grid-item-card:: YAML
        :class-header: bg-deep-purple

        .. code-block:: yaml

            model: HIPT
            output_format:
                outdir: /test_results/all/HIPT
            process_dataset:
                wsi_filelist: ./test_database/filelist.csv
                wsi_mpp: 0.25
                wsi_magnification: 40

Slides with x20 / 0.50 MPP
~~~~~~~~~~~~~~~~~~~~~~~~~~

Inference can be run on slides not matching x40 magnification (0.25 MPP). This is useful for example if you have a slide with 0.50 MPP and want to run inference on it. The model will automatically adapt to the given MPP and magnification. The example below shows how to run inference on a slide with 0.50 MPP and 20x magnification.

.. grid:: 1 2 2 2
    :gutter: 3

    .. grid-item-card:: CLI
        :class-header: bg-purple

        .. code-block:: bash

            cellvit-inference \
                --model HIPT \
                --outdir ./test_results/x20_svs/minimal/HIPT \
                process_wsi \
                --wsi_path ./test_database/x20_svs/CMU-1-Small-Region.svs

    .. grid-item-card:: YAML
        :class-header: bg-deep-purple

        .. code-block:: yaml

            model: HIPT
            output_format:
                outdir: ./test_results/x20_svs/minimal/HIPT
            process_wsi:
                wsi_path: ./test_database/x20_svs/CMU-1-Small-Region.svs



Classification Taxonomies
~~~~~~~~~~~~~~~~~~~~~~~~~

CoNSeP Classification
^^^^^^^^^^^^^^^^^^^^^

This classifier was trained on the CoNSeP dataset, which is specifically designed for colorectal nuclear segmentation and phenotyping. The dataset contains 24,319 annotated nuclei from 41 H&E-stained images, and we utilized the following label map for classification:

Label Map:

- **0**: Other
- **1**: Inflammatory
- **2**: Epithelial
- **3**: Spindle-Shaped

.. grid:: 1 2 2 2
    :gutter: 3

    .. grid-item-card:: CLI - SAM
        :class-header: bg-purple

        .. code-block:: bash

            cellvit-inference \
                --model SAM \
                --nuclei_taxonomy consep \
                --outdir ./test_results/BRACS/consep/SAM \
                process_wsi \
                --wsi_path ./test_database/BRACS/BRACS_1640_N_3_cropped.tiff \
                --wsi_mpp 0.25 \
                --wsi_magnification 40

        Note: wsi_mpp and wsi_magnification are just necessary because the example file has no metadata.


    .. grid-item-card:: YAML - SAM
        :class-header: bg-deep-purple

        .. code-block:: yaml

            model: SAM
            nuclei_taxonomy: consep

            output_format:
                outdir: ./test_results/BRACS/consep/SAM

            process_wsi:
                wsi_path: ./test_database/BRACS/BRACS_1640_N_3_cropped.tiff
                wsi_mpp: 0.25
                wsi_magnification: 40

        Note: wsi_mpp and wsi_magnification are just necessary because the example file has no metadata.


    .. grid-item-card:: CLI - HIPT
        :class-header: bg-purple

        .. code-block:: bash

            cellvit-inference \
                --model HIPT \
                --nuclei_taxonomy consep \
                --outdir ./test_results/BRACS/consep/HIPT \
                process_wsi \
                --wsi_path ./test_database/BRACS/BRACS_1640_N_3_cropped.tiff \
                --wsi_mpp 0.25 \
                --wsi_magnification 40

        Note: wsi_mpp and wsi_magnification are just necessary because the example file has no metadata.


    .. grid-item-card:: YAML - HIPT
        :class-header: bg-deep-purple

        .. code-block:: yaml

            model: HIPT
            nuclei_taxonomy: consep

            output_format:
                outdir: ./test_results/BRACS/consep/HIPT

            process_wsi:
                wsi_path: ./test_database/BRACS/BRACS_1640_N_3_cropped.tiff
                wsi_mpp: 0.25
                wsi_magnification: 40

        Note: wsi_mpp and wsi_magnification are just necessary because the example file has no metadata.


Lizard Classification
^^^^^^^^^^^^^^^^^^^^^

Lizard is the largest known available dataset for nuclear segmentation and phenotyping, containing nearly half a million nuclei for colon tissue.

Label Map:

- **0**: Neutrophil
- **1**: Epithelial
- **2**: Lymphocyte
- **3**: Plasma
- **4**: Eosinophil
- **5**: Connective Tissue

.. grid:: 1 2 2 2
    :gutter: 3

    .. grid-item-card:: CLI - SAM
        :class-header: bg-purple

        .. code-block:: bash

            cellvit-inference \
                --model SAM \
                --nuclei_taxonomy lizard \
                --outdir ./test_results/BRACS/lizard/SAM \
                process_wsi \
                --wsi_path ./test_database/BRACS/BRACS_1640_N_3_cropped.tiff \
                --wsi_mpp 0.25 \
                --wsi_magnification 40

        Note: wsi_mpp and wsi_magnification are just necessary because the example file has no metadata.


    .. grid-item-card:: YAML - SAM
        :class-header: bg-deep-purple

        .. code-block:: yaml

            model: SAM
            nuclei_taxonomy: lizard

            output_format:
                outdir: ./test_results/BRACS/lizard/SAM

            process_wsi:
                wsi_path: ./test_database/BRACS/BRACS_1640_N_3_cropped.tiff
                wsi_mpp: 0.25
                wsi_magnification: 40

        Note: wsi_mpp and wsi_magnification are just necessary because the example file has no metadata.


    .. grid-item-card:: CLI - HIPT
        :class-header: bg-purple

        .. code-block:: bash

            cellvit-inference \
                --model HIPT \
                --nuclei_taxonomy lizard \
                --outdir ./test_results/BRACS/lizard/HIPT \
                process_wsi \
                --wsi_path ./test_database/BRACS/BRACS_1640_N_3_cropped.tiff \
                --wsi_mpp 0.25 \
                --wsi_magnification 40

        Note: wsi_mpp and wsi_magnification are just necessary because the example file has no metadata.


    .. grid-item-card:: YAML - HIPT
        :class-header: bg-deep-purple

        .. code-block:: yaml

            model: HIPT
            nuclei_taxonomy: lizard

            output_format:
                outdir: ./test_results/BRACS/lizard/HIPT

            process_wsi:
                wsi_path: ./test_database/BRACS/BRACS_1640_N_3_cropped.tiff
                wsi_mpp: 0.25
                wsi_magnification: 40

        Note: wsi_mpp and wsi_magnification are just necessary because the example file has no metadata.


NuCLS Classification
^^^^^^^^^^^^^^^^^^^^

The NuCLS dataset contains over 220,000 labeled nuclei from breast cancer images from TCGA. We provide classification for both **Main Groups** and **Super Groups** in nuclear phenotyping:

nucls_super:
0: Tumor
1: nonTIL Stromal
2: sTIL
3: Other Nucleus

Label Map for main NuCLS classes:

- **0**: Tumor nonMitotic
- **1**: Tumor Mitotic
- **2**: nonTILnonMQ Stromal
- **3**: Macrophage
- **4**: Lymphocyte
- **5**: Plasma Cell
- **6**: Other Nucleus

Label Map for super NuCLS classes:

- **0**: Tumor
- **1**: nonTIL Stromal
- **2**: sTIL
- **3**: Other Nucleus



.. grid:: 1 2 2 2
    :gutter: 3

    .. grid-item-card:: CLI - SAM - Main NuCLS
        :class-header: bg-purple

        .. code-block:: bash

            cellvit-inference \
                --model SAM \
                --nuclei_taxonomy nucls_main \
                --outdir ./test_results/BRACS/nucls_main/SAM \
                process_wsi \
                --wsi_path ./test_database/BRACS/BRACS_1640_N_3_cropped.tiff \
                --wsi_mpp 0.25 \
                --wsi_magnification 40

        Note: wsi_mpp and wsi_magnification are just necessary because the example file has no metadata.


    .. grid-item-card:: YAML - SAM - Main NuCLS
        :class-header: bg-deep-purple

        .. code-block:: yaml

            model: SAM
            nuclei_taxonomy: nucls_main

            output_format:
                outdir: ./test_results/BRACS/nucls_main/SAM

            process_wsi:
                wsi_path: ./test_database/BRACS/BRACS_1640_N_3_cropped.tiff
                wsi_mpp: 0.25
                wsi_magnification: 40

        Note: wsi_mpp and wsi_magnification are just necessary because the example file has no metadata.


    .. grid-item-card:: CLI - HIPT - Main NuCLS
        :class-header: bg-purple

        .. code-block:: bash

            cellvit-inference \
                --model HIPT \
                --nuclei_taxonomy nucls_main \
                --outdir ./test_results/BRACS/nucls_main/HIPT \
                process_wsi \
                --wsi_path ./test_database/BRACS/BRACS_1640_N_3_cropped.tiff \
                --wsi_mpp 0.25 \
                --wsi_magnification 40

        Note: wsi_mpp and wsi_magnification are just necessary because the example file has no metadata.


    .. grid-item-card:: YAML - HIPT - Main NuCLS
        :class-header: bg-deep-purple

        .. code-block:: yaml

            model: HIPT
            nuclei_taxonomy: nucls_main

            output_format:
                outdir: ./test_results/BRACS/nucls_main/HIPT

            process_wsi:
                wsi_path: ./test_database/BRACS/BRACS_1640_N_3_cropped.tiff
                wsi_mpp: 0.25
                wsi_magnification: 40

        Note: wsi_mpp and wsi_magnification are just necessary because the example file has no metadata.

    .. grid-item-card:: CLI - SAM - Super NuCLS
        :class-header: bg-purple

        .. code-block:: bash

            cellvit-inference \
                --model SAM \
                --nuclei_taxonomy nucls_super \
                --outdir ./test_results/BRACS/nucls_super/SAM \
                process_wsi \
                --wsi_path ./test_database/BRACS/BRACS_1640_N_3_cropped.tiff \
                --wsi_mpp 0.25 \
                --wsi_magnification 40

        Note: wsi_mpp and wsi_magnification are just necessary because the example file has no metadata.


    .. grid-item-card:: YAML - SAM - Super NuCLS
        :class-header: bg-deep-purple

        .. code-block:: yaml

            model: SAM
            nuclei_taxonomy: nucls_super

            output_format:
                outdir: ./test_results/BRACS/nucls_super/SAM

            process_wsi:
                wsi_path: ./test_database/BRACS/BRACS_1640_N_3_cropped.tiff
                wsi_mpp: 0.25
                wsi_magnification: 40

        Note: wsi_mpp and wsi_magnification are just necessary because the example file has no metadata.


    .. grid-item-card:: CLI - HIPT - Super NuCLS
        :class-header: bg-purple

        .. code-block:: bash

            cellvit-inference \
                --model HIPT \
                --nuclei_taxonomy nucls_super \
                --outdir ./test_results/BRACS/nucls_super/HIPT \
                process_wsi \
                --wsi_path ./test_database/BRACS/BRACS_1640_N_3_cropped.tiff \
                --wsi_mpp 0.25 \
                --wsi_magnification 40

        Note: wsi_mpp and wsi_magnification are just necessary because the example file has no metadata.


    .. grid-item-card:: YAML - HIPT - Super NuCLS
        :class-header: bg-deep-purple

        .. code-block:: yaml

            model: HIPT
            nuclei_taxonomy: nucls_super

            output_format:
                outdir: ./test_results/BRACS/nucls_super/HIPT

            process_wsi:
                wsi_path: ./test_database/BRACS/BRACS_1640_N_3_cropped.tiff
                wsi_mpp: 0.25
                wsi_magnification: 40

        Note: wsi_mpp and wsi_magnification are just necessary because the example file has no metadata.



PanOpTILS Classification
^^^^^^^^^^^^^^^^^^^^^^^^

PanopTILs was created by reconciling and expanding two public datasets, BCSS and NuCLS, to enable in-depth analysis of the tumor microenvironment (TME) in whole-slide images (WSI) of H&E stained slides of breast cancer.

Label Map:

- **0**: Other Cells
- **1**: Epithelial Cells
- **2**: Stromal Cells
- **3**: TILs

.. grid:: 1 2 2 2
    :gutter: 3

    .. grid-item-card:: CLI - SAM
        :class-header: bg-purple

        .. code-block:: bash

            cellvit-inference \
                --model SAM \
                --nuclei_taxonomy panoptils \
                --outdir ./test_results/BRACS/panoptils/SAM \
                process_wsi \
                --wsi_path ./test_database/BRACS/BRACS_1640_N_3_cropped.tiff \
                --wsi_mpp 0.25 \
                --wsi_magnification 40

        Note: wsi_mpp and wsi_magnification are just necessary because the example file has no metadata.


    .. grid-item-card:: YAML - SAM
        :class-header: bg-deep-purple

        .. code-block:: yaml

            model: SAM
            nuclei_taxonomy: panoptils

            output_format:
                outdir: ./test_results/BRACS/panoptils/SAM

            process_wsi:
                wsi_path: ./test_database/BRACS/BRACS_1640_N_3_cropped.tiff
                wsi_mpp: 0.25
                wsi_magnification: 40

        Note: wsi_mpp and wsi_magnification are just necessary because the example file has no metadata.


    .. grid-item-card:: CLI - HIPT
        :class-header: bg-purple

        .. code-block:: bash

            cellvit-inference \
                --model HIPT \
                --nuclei_taxonomy panoptils \
                --outdir ./test_results/BRACS/panoptils/HIPT \
                process_wsi \
                --wsi_path ./test_database/BRACS/BRACS_1640_N_3_cropped.tiff \
                --wsi_mpp 0.25 \
                --wsi_magnification 40

        Note: wsi_mpp and wsi_magnification are just necessary because the example file has no metadata.


    .. grid-item-card:: YAML - HIPT
        :class-header: bg-deep-purple

        .. code-block:: yaml

            model: HIPT
            nuclei_taxonomy: panoptils

            output_format:
                outdir: ./test_results/BRACS/panoptils/HIPT

            process_wsi:
                wsi_path: ./test_database/BRACS/BRACS_1640_N_3_cropped.tiff
                wsi_mpp: 0.25
                wsi_magnification: 40

        Note: wsi_mpp and wsi_magnification are just necessary because the example file has no metadata.



Ocelot Classification
^^^^^^^^^^^^^^^^^^^^^

​The OCELOT dataset is a histopathology resource designed to enhance tumor cell detection methods by providing small field-of-view image patches annotated with precise cell locations and classifications.

Label Map:

- **0**: Other Cells
- **1**: Tumor

.. grid:: 1 2 2 2
    :gutter: 3

    .. grid-item-card:: CLI - SAM
        :class-header: bg-purple

        .. code-block:: bash

            cellvit-inference \
                --model SAM \
                --nuclei_taxonomy ocelot \
                --outdir ./test_results/BRACS/ocelot/SAM \
                process_wsi \
                --wsi_path ./test_database/BRACS/BRACS_1640_N_3_cropped.tiff \
                --wsi_mpp 0.25 \
                --wsi_magnification 40

        Note: wsi_mpp and wsi_magnification are just necessary because the example file has no metadata.


    .. grid-item-card:: YAML - SAM
        :class-header: bg-deep-purple

        .. code-block:: yaml

            model: SAM
            nuclei_taxonomy: ocelot

            output_format:
                outdir: ./test_results/BRACS/ocelot/SAM

            process_wsi:
                wsi_path: ./test_database/BRACS/BRACS_1640_N_3_cropped.tiff
                wsi_mpp: 0.25
                wsi_magnification: 40

        Note: wsi_mpp and wsi_magnification are just necessary because the example file has no metadata.


    .. grid-item-card:: CLI - HIPT
        :class-header: bg-purple

        .. code-block:: bash

            cellvit-inference \
                --model HIPT \
                --nuclei_taxonomy ocelot \
                --outdir ./test_results/BRACS/ocelot/HIPT \
                process_wsi \
                --wsi_path ./test_database/BRACS/BRACS_1640_N_3_cropped.tiff \
                --wsi_mpp 0.25 \
                --wsi_magnification 40

        Note: wsi_mpp and wsi_magnification are just necessary because the example file has no metadata.


    .. grid-item-card:: YAML - HIPT
        :class-header: bg-deep-purple

        .. code-block:: yaml

            model: HIPT
            nuclei_taxonomy: ocelot

            output_format:
                outdir: ./test_results/BRACS/ocelot/HIPT

            process_wsi:
                wsi_path: ./test_database/BRACS/BRACS_1640_N_3_cropped.tiff
                wsi_mpp: 0.25
                wsi_magnification: 40

        Note: wsi_mpp and wsi_magnification are just necessary because the example file has no metadata.



MIDOG Classification
^^^^^^^^^^^^^^^^^^^^^

The MIDOG dataset is tailored for Mitotic cell detection. However, we **do not recommend** using it within CellViT, as it might have a inferior performance on rare types such as mitotic figures.

Label Map:

- **0**: Mitotic
- **1**: Non-Mitotic

.. grid:: 1 2 2 2
    :gutter: 3

    .. grid-item-card:: CLI - SAM
        :class-header: bg-purple

        .. code-block:: bash

            cellvit-inference \
                --model SAM \
                --nuclei_taxonomy midog \
                --outdir ./test_results/BRACS/midog/SAM \
                process_wsi \
                --wsi_path ./test_database/BRACS/BRACS_1640_N_3_cropped.tiff \
                --wsi_mpp 0.25 \
                --wsi_magnification 40

        Note: wsi_mpp and wsi_magnification are just necessary because the example file has no metadata.


    .. grid-item-card:: YAML - SAM
        :class-header: bg-deep-purple

        .. code-block:: yaml

            model: SAM
            nuclei_taxonomy: midog

            output_format:
                outdir: ./test_results/BRACS/midog/SAM

            process_wsi:
                wsi_path: ./test_database/BRACS/BRACS_1640_N_3_cropped.tiff
                wsi_mpp: 0.25
                wsi_magnification: 40

        Note: wsi_mpp and wsi_magnification are just necessary because the example file has no metadata.

.. warning:: We do not provide a MIDOG classifier for the HIPT backbone


Hardware Setup
~~~~~~~~~~~~~~

The CellViT inference code is determined to automatically detect the available hardware and use it for inference. For this, we detect the available GPUs, CPUs and Memory and use them for inference. This *should* work regardless if you are using a bare metal server, a cloud instance, slurm-clusters, docker container or kubernetes clusters. However, if problems occur and you want to apply hardware constraints, we provide an interface to set the hardware constraints manually.
If you want to see the hardware setup that CellViT is detecting on your setup, you can run the `cellvit-check` command in your terminal.

Batch-size
^^^^^^^^^^

Set the batch-size (amount of 1024 x 1024 patches per batch):

.. grid:: 1 2 2 2
    :gutter: 3

    .. grid-item-card:: CLI
        :class-header: bg-purple

        .. code-block:: bash

            cellvit-inference \
                --model SAM \
                --batch_size 32
                --outdir ./test_results/x40_svs/batch_size/SAM \
                process_dataset \
                --wsi_folder ./test_database/x40_svs


    .. grid-item-card:: YAML
        :class-header: bg-deep-purple

        .. code-block:: yaml

            model: SAM
            output_format:
                outdir: ./test_results/x40_svs/batch_size/SAM
            inference:
                batch_size: 32
            process_dataset:
                wsi_folder: ./test_database/x40_svs

CPU-cores
^^^^^^^^^

Set available CPU-cores:

.. grid:: 1 2 2 2
    :gutter: 3

    .. grid-item-card:: CLI
        :class-header: bg-purple

        .. code-block:: bash

            cellvit-inference \
                --model SAM \
                --outdir ./test_results/x40_svs/cpu_count/SAM \
                --cpu_count 16 \
                process_dataset \
                --wsi_folder ./test_database/x40_svs


    .. grid-item-card:: YAML
        :class-header: bg-deep-purple

        .. code-block:: yaml

            model: SAM
            output_format:
                outdir: ./test_results/x40_svs/cpu_count/SAM
            process_dataset:
                wsi_folder: ./test_database/x40_svs
            system:
                cpu_count: 16

Ray-Setup (Multiprocessing)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

We are using ray for post-processing and scheduling the HV-Net postprocessing strategy. Set available CPU-cores and number of ray-worker (ray is used for parallelization). Each ray worker is post-processing one batch in parallel:

.. grid:: 1 2 2 2
    :gutter: 3

    .. grid-item-card:: CLI
        :class-header: bg-purple

        .. code-block:: bash

            cellvit-inference \
                --model SAM \
                --outdir ./test_results/x40_svs/cpu_count_ray_worker/SAM \
                --cpu_count 16 \
                --ray_worker 4 \
                process_dataset \
                --wsi_folder ./test_database/x40_svs


    .. grid-item-card:: YAML
        :class-header: bg-deep-purple

        .. code-block:: yaml

            model: SAM
            output_format:
                outdir: ./test_results/x40_svs/cpu_count_ray_count/SAM
            process_dataset:
                wsi_folder: ./test_database/x40_svs
            system:
                cpu_count: 16
                ray_worker: 4

Usually, the each ray worker is getting a shared portion of the available CPUs. To set the CPUs per worker, we provide the following option:

.. grid:: 1 2 2 2
    :gutter: 3

    .. grid-item-card:: CLI
        :class-header: bg-purple

        .. code-block:: bash

            cellvit-inference \
                --model SAM \
                --outdir ./test_results/x40_svs/ray_worker_cpu_remote/SAM \
                --cpu_count 16 \
                --ray_worker 6 \
                --ray_remote_cpus 1 \
                process_dataset \
                --wsi_folder ./test_database/x40_svs


    .. grid-item-card:: YAML
        :class-header: bg-deep-purple

        .. code-block:: yaml

            model: SAM
            output_format:
                outdir: ./test_results/x40_svs/ray_worker_cpu_remote/SAM
            process_dataset:
                wsi_folder: ./test_database/x40_svs
            system:
                cpu_count: 16
                ray_worker: 4
                ray_remote_cpus: 1
