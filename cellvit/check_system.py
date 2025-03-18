# -*- coding: utf-8 -*-
# Check the CellViT environment
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essenimport subprocess

import logging
import sys
from importlib.resources import files

import subprocess
from cellvit.utils.ressource_manager import get_cpu_resources, get_gpu_resources
from cellvit.utils.check_module import perform_module_check, check_module
from cellvit.utils.check_cupy import check_cupy
from cellvit.utils.logger import Logger

PYTHON_PATH = sys.executable
CHECK_RAY_PATH = str(files("cellvit.utils").joinpath("check_ray.py"))


def check_optional_modules(logger: logging.Logger) -> None:
    perform_module_check(module_name="cupy", logger=logger)
    perform_module_check(module_name="cupyx", logger=logger)
    perform_module_check(module_name="cucim", logger=logger)
    perform_module_check(module_name="numba", logger=logger)


def check_necessary_modules(logger: logging.Logger) -> None:
    perform_module_check(module_name="torch", logger=logger)
    perform_module_check(module_name="torchaudio", logger=logger)
    perform_module_check(module_name="torchvision", logger=logger)
    perform_module_check(module_name="pandas", logger=logger)
    perform_module_check(module_name="numpy", logger=logger)
    perform_module_check(module_name="PIL", logger=logger)
    perform_module_check(module_name="uuid", logger=logger)
    perform_module_check(module_name="shapely", logger=logger)
    perform_module_check(module_name="ujson", logger=logger)
    perform_module_check(module_name="tqdm", logger=logger)
    perform_module_check(module_name="snappy", logger=logger)
    perform_module_check(module_name="collections", logger=logger)
    perform_module_check(module_name="einops", logger=logger)
    perform_module_check(module_name="scipy", logger=logger)
    perform_module_check(module_name="skimage", logger=logger)
    perform_module_check(module_name="cv2", logger=logger)
    perform_module_check(module_name="functools", logger=logger)
    perform_module_check(module_name="dataclasses", logger=logger)
    perform_module_check(module_name="pathopatch", logger=logger)
    perform_module_check(module_name="cellvit.inference.cli", logger=logger)
    perform_module_check(module_name="cellvit.inference.inference", logger=logger)


def main():
    logger = Logger(
        level="DEBUG",
        formatter="%(asctime)s - Check-System - %(levelname)s - %(message)s",
    )
    logger = logger.create_logger()
    logger.warning("To check the environment, the following python path is used:")
    logger.warning(PYTHON_PATH)

    # import necessary volumes
    print(f"\n{20*'*'} Necessary Modules {20*'*'}\n")
    check_necessary_modules(logger)

    # log system resources
    print(f"\n{20*'*'} System {20*'*'}\n")
    get_cpu_resources(logger)
    get_gpu_resources(logger)

    # import optional modules
    print(f"\n{20*'*'} Optional Modules {20*'*'}\n")
    print(f"\n{100*'*'}")
    print(
        f"* Error or warning messages here indicate that optional packages might not be installed correctly  *"
    )
    print(
        f"* If an error occur, check if you installed to library accordingly                                 *"
    )
    print(
        f"* CellViT++ will still work, but with limited features and might be severly slower                 *"
    )
    print(f"{100*'*'}")
    check_optional_modules(logger)

    # check if cupy functionality is really working
    if check_module("cupy"):
        print(f"\n{20*'*'} CuPY {20*'*'}\n")
        check_cupy(True, logger)
    else:
        logger.warning(
            "CuPY cannot be loaded - falling back to numpy for postprocessing"
        )

    # check ray
    if check_module("ray"):
        try:
            print(f"\n{20*'*'} Ray {20*'*'}\n")
            result = subprocess.run(
                [PYTHON_PATH, CHECK_RAY_PATH],
                check=True,
                text=True,
            )
            logger.info("Executed Ray")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error while running 'ray_test.py': {e.stderr}")
            raise e
        except Exception as e:
            logger.error(f"Unexpected error while running 'ray_test.py': {e}")
            raise e


if __name__ == "__main__":
    main()
