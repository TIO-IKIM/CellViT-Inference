# -*- coding: utf-8 -*-
# Configuration
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

from colorama import Fore
from pathlib import Path
import os

COLOR_CODES_LOGGING = {
    "DEBUG": Fore.LIGHTBLUE_EX,
    "INFO": Fore.WHITE,
    "WARNING": Fore.LIGHTYELLOW_EX,
    "ERROR": Fore.LIGHTRED_EX,
    "CRITICAL": Fore.RED,
}

COLOR_DICT_CELLS = {
    0: [92, 20, 186],
    1: [255, 0, 0],
    2: [34, 221, 77],
    3: [35, 92, 236],
    4: [254, 255, 0],
    5: [255, 159, 68],
    6: [80, 56, 112],
    7: [87, 112, 56],
    8: [110, 0, 0],
    9: [255, 196, 196],
    10: [214, 255, 196],
}

TYPE_NUCLEI_DICT_PANNUKE = {
    1: "Neoplastic",
    2: "Inflammatory",
    3: "Connective",
    4: "Dead",
    5: "Epithelial",
}

BACKBONE_EMBED_DIM = {
    "ViT256": 384,
    "SAM-H": 1280,
}

CELL_IMAGE_SIZES = [
    256,
    288,
    320,
    352,
    384,
    416,
    448,
    480,
    512,
    544,
    576,
    608,
    640,
    672,
    704,
    736,
    768,
    800,
    832,
    864,
    896,
    928,
    960,
    992,
    1024,
]

# Define the cache directory
CACHE_DIR = Path(os.getenv("CELLVIT_CACHE", str(Path.home() / ".cache" / "cellvit")))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
