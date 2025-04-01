[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![CodeFactor](https://www.codefactor.io/repository/github/tio-ikim/cellvit-inference/badge)](https://www.codefactor.io/repository/github/tio-ikim/cellvit-inference)
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=Pytorch&logoColor=white"/>
[![arXiv](https://img.shields.io/badge/arXiv-2501.05269-b31b1b.svg)](https://arxiv.org/abs/2501.05269)
[![Documentation](https://img.shields.io/badge/docs-CellViT--Inference-blue.svg)](https://tio-ikim.github.io/CellViT-Inference/)


# CellViT/CellViT++ Inference


---

## Documentation

<div align="center">
  <a href="https://tio-ikim.github.io/CellViT-Inference/" target="_blank">
    <img src="https://img.shields.io/badge/Read%20The%20Docs-4285F4?style=for-the-badge&logo=readthedocs&logoColor=white" alt="Documentation"/>
  </a>
</div>


The full documentation is available at: [https://tio-ikim.github.io/CellViT-Inference/](https://tio-ikim.github.io/CellViT-Inference/)

The code can be assessed here: [https://github.com/TIO-IKIM/CellViT-Inference](https://github.com/TIO-IKIM/CellViT-Inference)

### Documentation Sections

| Section | Description |
|---------|-------------|
| [📚 Installation Guide](https://tio-ikim.github.io/CellViT-Inference/getting-started.html) | Detailed installation instructions for various environments |
| [📝 Usage Documentation](https://tio-ikim.github.io/CellViT-Inference/usage.html) | Comprehensive guide on how to use CellViT-Inference |
| [💡 Examples](https://tio-ikim.github.io/CellViT-Inference/examples.html) | Sample configurations and use cases |


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


### Check your installation and the system

To verify a successful installation, run the following command:

```bash
cellvit-check
```

The output should confirm that all required libraries are installed and accessible. If any libraries are missing, refer to the installation instructions for the respective library. This command will also check for optional dependencies and will print a warning if they are not installed. Installing these optional libraries is **not required**.

If using a virtual environment, ensure it is activated before running the command.

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
