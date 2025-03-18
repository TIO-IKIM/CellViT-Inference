# -*- coding: utf-8 -*-
# Check if a module is installed in the environment.
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import importlib.util
from cellvit.utils.logger import NullLogger
from typing import Optional
import logging


def check_module(module_name: str) -> bool:
    """Check if a module is installed in the environment."""
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return False
    return True


def perform_module_check(
    module_name: str, logger: Optional[logging.Logger] = None
) -> None:
    """Check if a module is installed in the environment and log the result.

    Args:
        module_name (str): Name of the module to check.
        logger (Optional[logging.Logger], optional): Logger. Defaults to None.

    Raises:
        e: ImportError if the module is not installed.
    """
    logger = logger or NullLogger()
    try:
        logger.info(f"Checking library {module_name}")
        module_loaded = check_module(module_name=module_name)
        if module_loaded:
            logger.info(f"Module installed and loaded")
        else:
            logger.error(f"Failed loading {module_name}")
    except ImportError as e:
        logger.error(f"Error: {e}")
        raise e
