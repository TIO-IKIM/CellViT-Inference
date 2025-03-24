# -*- coding: utf-8 -*-
# Check if CuPy is installed and working correctly.
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

from typing import Optional
import logging
from cellvit.utils.logger import NullLogger


def check_cupy(
    check_gpu: True, logger: Optional[logging.Logger] = NullLogger()
) -> bool:
    """Check if CuPy is installed and working correctly.

    Args:
        check_gpu (True): Check if CuPy can access the GPU.
        logger (Optional[logging.Logger], optional): Logger. Defaults to None.

    Raises:
        SystemError: No CUDA-capable GPU detected by CuPy.

    Returns:
        bool: True if CuPy is installed and working correctly.
    """
    try:
        import cupy as cp

        # Perform a simple CuPy operation to ensure it's functioning
        a = cp.array([1, 2, 3])
        b = cp.array([4, 5, 6])
        c = a + b
        if not cp.allclose(c, [5, 7, 9]):
            logger.error("CuPy operation validation failed.")
            return False
        elif not check_gpu:
            logger.info("CuPy is functioning correctly, but without GPU check")
            return True
        device_count = cp.cuda.runtime.getDeviceCount()
        if device_count == 0:
            raise SystemError("No CUDA-capable GPU detected by CuPy.")

        logger.info(f"CuPy detected {device_count} CUDA device(s).")
        for device_id in range(device_count):
            # Set the current device
            cp.cuda.Device(device_id).use()

            # Get device properties
            props = cp.cuda.runtime.getDeviceProperties(device_id)
            logger.info(f"Device ID: {device_id}")
            logger.info(f"  Name: {props['name']}")
            logger.info(f"  Total Global Memory: {props['totalGlobalMem']} bytes")
            logger.info(f"  Multiprocessor Count: {props['multiProcessorCount']}")
            logger.info(f"  Compute Capability: {props['major']}.{props['minor']}")
            logger.info("")

        logger.info("CuPy is able to access the GPU and perform operations.")
        return True

    except ImportError as e:
        logger.error(f"CuPy Import Error: {e}")
        return False
    except RuntimeError as e:
        logger.error(f"CuPy Error: {e}")
        return False
    except SystemError as e:
        logger.error(f"CuPy GPU Error: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during CuPy check: {e}")
        return False
