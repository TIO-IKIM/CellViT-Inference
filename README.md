[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![CodeFactor](https://www.codefactor.io/repository/github/tio-ikim/cellvit-inference/badge)](https://www.codefactor.io/repository/github/tio-ikim/cellvit-inference)
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=Pytorch&logoColor=white"/></a>
[![arXiv](https://img.shields.io/badge/arXiv-2501.05269-b31b1b.svg)](https://arxiv.org/abs/2501.05269)
___
<p align="center">
  <img src="./docs/source/_static/banner.png"/>
</p>

___

# CellViT/CellViT++ Inference
<div align="center">

[Key Features](#key-features) • [Installation](#installation) • [Examples](#examples) • [Re-training](#re-training-your-own-classifier) • [Citation](#Citation)

</div>

> [!IMPORTANT]  
> The package is now available on PyPI: `pip install cellvit`

> [!TIP]
> This repository is solely based to perform inference on WIS using CellViT++ and the basic CellViT model. For this, we include CellViT-HIPT-256 and CellViT-SAM-H as well as the lightweight classifier modules. This repo does not contain training codes.
> To access the previous version (CellViT), follow this [link](https://github.com/TIO-IKIM/CellViT)
> To access the CellViT++ repo, follow this [link](https://github.com/TIO-IKIM/CellViT-plus-plus)


## Documentation
The documentation ... https://tio-ikim.github.io/CellViT-Inference/

## Installation

### Hardware Requirements

- 🚀 **CUDA-capable GPU**: A GPU with at least 24 GB VRAM (48 GB recommended for faster inference, e.g., RTX-A6000). We performed experiments using one NVIDIA A100 with 80GB VRAM.
- 🧠 **Memory**: Minimum 32 GB RAM.
- 💾 **Storage**: At least 30 GB disk space.
- 🖥️ **CPU**: Minimum of 16 CPU cores.

### Local installation

#### Prerequisites

Before installing the package, ensure that the following prerequisites are met as listed below:

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























## Citation

**CellViT++**
```latex
@misc{hörst2025cellvitenergyefficientadaptivecell,
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
