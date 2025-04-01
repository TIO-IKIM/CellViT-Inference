Examples
========

Test Database
~~~~~~~~~~~~~

.. admonition:: Download test database into current directory
    :class: example

    .. code-block:: bash

        download-test-database # run in your terminal

This command will download a test database into the current directory. The database is used for testing purposes and contains sample data to demonstrate the functionality of the package.
The database is not required for the package to function, but it can be useful for testing and development purposes.

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

        .. code-block:: YAML

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

        .. code-block:: YAML

            model: SAM
            output_format:
                outdir: ./test_results/x40_svs/minimal/SAM
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

        .. code-block:: YAML

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

        .. code-block:: YAML

            model: HIPT
            output_format:
                outdir: ./test_results/BRACS/minimal/HIPT
            process_wsi:
                wsi_path: ./test_database/BRACS/BRACS_1640_N_3_cropped.tiff
                wsi_mpp: 0.25
                wsi_magnification: 40



Classification Examples
~~~~~~~~~~~~~~~~~~~~~~~

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

        .. code-block:: YAML

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

        .. code-block:: YAML

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

        .. code-block:: YAML

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

        .. code-block:: YAML

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

        .. code-block:: YAML

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

        .. code-block:: YAML

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

        .. code-block:: YAML

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

        .. code-block:: YAML

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

        .. code-block:: YAML

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

        .. code-block:: YAML

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

        .. code-block:: YAML

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

        .. code-block:: YAML

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

        .. code-block:: YAML

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
