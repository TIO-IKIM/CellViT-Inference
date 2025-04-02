[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![CodeFactor](https://www.codefactor.io/repository/github/tio-ikim/cellvit-inference/badge)](https://www.codefactor.io/repository/github/tio-ikim/cellvit-inference)
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=Pytorch&logoColor=white"/>
[![arXiv](https://img.shields.io/badge/arXiv-2501.05269-b31b1b.svg)](https://arxiv.org/abs/2501.05269)
[![Documentation](https://img.shields.io/badge/docs-CellViT--Inference-blue.svg)](https://tio-ikim.github.io/CellViT-Inference/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TIO-IKIM/CellViT-Inference/blob/main/CellViT_Inference.ipynb)
___
<p align="center">
  <img src="./docs/source/_static/banner.png"/>
</p>

___

# CellViT/CellViT++ Inference
<div align="center">

[Key Features](#key-features) • [Documentation](#documentation) • [Installation](#installation) • [Usage](#basic-usage) • [Examples](#examples-and-usage) • [Citation](#citation)

</div>

> [!IMPORTANT]  
> The package is now available on PyPI: `pip install cellvit`

> [!TIP]
> This repository is solely for performing inference on WSIs using CellViT++ and the basic CellViT model. It includes CellViT-HIPT-256 and CellViT-SAM-H as well as lightweight classifier modules. This repo does not contain training code.
>
> To access the previous version (CellViT), follow this [link](https://github.com/TIO-IKIM/CellViT)  
> To access the CellViT++ repo, follow this [link](https://github.com/TIO-IKIM/CellViT-plus-plus)

## Key Features

- 🚀 Optimized inference pipeline for high-performance processing
- 🔄 Support for multiple WSI formats and magnifications
- 📊 Comprehensive analysis and visualization tools
- 💻 Easy to install via PyPI (pip)

---

## Documentation

<div align="center">
  <a href="https://tio-ikim.github.io/CellViT-Inference/" target="_blank">
    <img src="https://img.shields.io/badge/Read%20The%20Docs-4285F4?style=for-the-badge&logo=readthedocs&logoColor=white" alt="Documentation"/>
  </a>
</div>


The full documentation is available at: [https://tio-ikim.github.io/CellViT-Inference/](https://tio-ikim.github.io/CellViT-Inference/)


### Documentation Sections

| Section | Description |
|---------|-------------|
| [📚 Installation Guide](https://tio-ikim.github.io/CellViT-Inference/getting-started.html) | Detailed installation instructions for various environments |
| [📝 Usage Documentation](https://tio-ikim.github.io/CellViT-Inference/usage.html) | Comprehensive guide on how to use CellViT-Inference |
| [💡 Examples](https://tio-ikim.github.io/CellViT-Inference/examples.html) | Sample configurations and use cases |

## Colab Example

If you want to have a quick test, check out the colab notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TIO-IKIM/CellViT-Inference/blob/main/CellViT_Inference.ipynb)

## Installation

### Hardware Requirements

- 🚀 **CUDA-capable GPU**: A GPU with at least 24 GB VRAM (48 GB recommended for faster inference, e.g., RTX-A6000). We performed experiments using one NVIDIA A100 with 80GB VRAM.
- 🧠 **Memory**: Minimum 32 GB RAM.
- 💾 **Storage**: At least 30 GB disk space.
- 🖥️ **CPU**: Minimum of 16 CPU cores.

### Prerequisites

Before installing the package, ensure that the following prerequisites are met:

<details>
  <summary> Binaries </summary>

- libvips - Image processing library
- openslide - Whole slide image library
- gcc/g++ - C/C++ compilers
- libopencv-core-dev - OpenCV core development files
- libopencv-imgproc-dev - OpenCV image processing modules
- libsnappy-dev - Compression library
- libgeos-dev - Geometry engine library
- llvm - Compiler infrastructure
- libjpeg-dev - JPEG image format library
- libpng-dev - PNG image format library
- libtiff-dev - TIFF image format library

Most of the times, they are already installed on your system. Just give it a try and check out CellViT.
On Linux-based systems, you can install these using:

```sh
sudo apt-get install libvips openslide gcc g++ libopencv-core-dev libopencv-imgproc-dev libsnappy-dev libgeos-dev llvm libjpeg-dev libpng-dev libtiff-dev
```

</details>

### Package installation

#### To install the package, follow these steps:

1. Ensure that all prerequisites are installed as outlined above.
2. Verify that OpenSlide (https://openslide.org/) is installed and accessible. If using conda, you can install it with:

   ```bash
   conda install -c conda-forge openslide
   ```

3. Install PyTorch for your system by following the instructions at https://pytorch.org/get-started/locally/. Ensure that PyTorch >= 2.0 is installed. To view available versions and their corresponding CUDA versions, visit https://pytorch.org/get-started/previous-versions/.
   CellViT-Inference has been tested with PyTorch 2.2.2 and CUDA 12.1, such that we installed it via:

   ```bash
   pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
   ```

    > [!IMPORTANT]  
    > This might differ for your system

4. Install the CellViT-Inference package using pip:

   ```bash
   pip install cellvit
   ```

#### Optional

To enable hardware-accelerated libraries, you can install the following optional dependencies:

1. CuPy (CUDA accelerated NumPy): https://cupy.dev/
2. cuCIM (RAPIDS cuCIM library): https://github.com/rapidsai/cucim


### Install from Git Repository as integrative framework

If you prefer to install CellViT-Inference directly from the GitHub repository, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/TIO-IKIM/CellViT-Inference.git
   cd CellViT-Inference
   ```

2. Install the required prerequisites as mentioned in the [Prerequisites](#prerequisites) section.

3. Install PyTorch according to your system requirements (as described in the main installation steps).

4. Install the package in development mode:
   ```bash
   pip install -e .
   ```

5. Verify the installation:
   ```bash
   cellvit-check
   ```

Installing in development mode (`-e` flag) allows you to modify the source code and have the changes reflected immediately without needing to reinstall the package. This is useful for developers who want to contribute to the project or make custom modifications.

> [!NOTE]
> When installing from Git, you will always have the latest development version, which may include experimental features not yet available in the PyPI release.

### Check your installation and the system

To verify a successful installation, run the following command:

```bash
cellvit-check
```

The output should confirm that all required libraries are installed and accessible. If any libraries are missing, refer to the installation instructions for the respective library. This command will also check for optional dependencies and will print a warning if they are not installed. Installing these optional libraries is **not required**.

If using a virtual environment, ensure it is activated before running the command.

## Basic Usage

This package is designed as a command-line tool. Configuration can be provided either directly via the CellViT CLI or by using a configuration file. The configuration file is a YAML file containing the settings for the inference pipeline.
The main script is located in the `cellvit` module, and can be run using the following command:

```bash
cellvit-inference
```

You then have to either specify a configuration file:

```bash
cellvit-inference --config <path_to_config_file>
```

or provide the required parameters directly in the command line. To list all available parameters, run:

```bash
cellvit-inference --help
```

You can select to run inference for one slide only or for a batch of slides. For more information, please refer to the Usage section in the documentation.

### Configuration

The `caching-directory` is used to store model weights, requiring at least 3GB of free space. By default, this is set to `~/.cache/cellvit`, but it can be changed by setting the environment variable `CELLVIT_CACHE` to a desired path. Remember to set this variable before running the command.

| Variable | Description |
|----------|-------------|
| `CELLVIT_CACHE` | Path to the caching directory. Default is `~/.cache/cellvit`. |

### Downloading Test Database

To download a test database into your current directory:

```bash
cellvit-download-examples # run in your terminal
```

This command will download a test database into the current directory. The database is used for testing purposes and contains sample data to demonstrate the functionality of the package.
The database is not required for the package to function, but it can be useful for testing and development purposes.

**Note:**
Contains sample data in these folders:

- `x40_svs/`: High-mag WSIs (.svs format)
- `x20_svs/`: Low-mag WSIs (.svs format)
- `BRACS/`: Breast cancer WSIs (.tiff format)
- `Philips/`: Alternative scanner format

## Examples and Usage

A thorough guideline on how to configure and run inference is provided in the [Usage Documentation](https://tio-ikim.github.io/CellViT-Inference/usage.html).

Additional examples can be found in the [Examples Documentation](https://tio-ikim.github.io/CellViT-Inference/examples.html).

## Citation

**CellViT++**
```latex
@misc{horst2025cellvitenergyefficientadaptivecell,
      title   = {CellViT++: Energy-Efficient and Adaptive Cell Segmentation and  
                Classification Using Foundation Models},
      author  = {Fabian Hörst and Moritz Rempe and Helmut Becker and Lukas Heine and
                Julius Keyl and Jens Kleesiek},
      year    = {2025},
      eprint  = {2501.05269},
      archivePrefix = {arXiv},
      primaryClass  = {cs.CV},
      url     = {https://arxiv.org/abs/2501.05269},
}
```

**CellViT**
```latex
@ARTICLE{Horst2024,
  title    =  {{CellViT}: Vision Transformers for precise cell segmentation and
              classification},
  author   =  {Hörst, Fabian and Rempe, Moritz and Heine, Lukas and Seibold,
              Constantin and Keyl, Julius and Baldini, Giulia and Ugurel, Selma
              and Siveke, Jens and Grünwald, Barbara and Egger, Jan and
              Kleesiek, Jens},
  journal  =  {Med. Image Anal.},
  volume   =  {94},
  pages    =  {103143},
  month    =  {may},
  year     =  {2024},
  keywords =  {Cell segmentation; Deep learning; Digital pathology; Vision
              transformer},
  language = {en}
}
```
