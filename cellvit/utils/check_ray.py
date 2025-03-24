# -*- coding: utf-8 -*-
# Test if the CellViT components can be imported in a Ray worker.
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen
# ruff: noqa: F401

import os
import sys
import ray
import time

PYTHON_PATH = sys.executable


# Logger function
def log_message(message, level="INFO"):
    """Log messages with timestamps and severity levels."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


# Helper function to check paths and imports in Ray workers
@ray.remote
def test_import():
    """Ray worker test for importing cellvit components."""
    import sys

    log_message("\nWorker sys.path:")
    for path in sys.path:
        log_message(f"  - {path}")

    try:
        import cupy

        log_message("CuPy is available.")

        try:
            from cellvit.inference.postprocessing_cupy import DetectionCellPostProcessor

            log_message("\nSuccessfully imported cellvit (cupy) in worker!")
            result = "Success"
        except Exception as e:
            result = f"Failed to import cellvit in worker: {str(e)}"
            log_message(result)

    except ImportError:
        result = "CuPy is not available."
        log_message(result)
        try:
            log_message("\nSuccessfully imported cellvit (numpy) in worker!")
            result = "Success"
        except Exception as e:
            result = f"Failed to import cellvit in worker: {str(e)}"
            log_message(result)

    return result


def main():
    """Main function to test cellvit imports."""
    status_code = 1  # Assume success

    # Print main process paths
    log_message("\nMain process sys.path:")
    for path in sys.path:
        log_message(f"  - {path}")

    # Initialize Ray with runtime environment
    runtime_env = {
        "env_vars": {"PYTHONPATH": PYTHON_PATH}  # Set PYTHONPATH for workers
    }

    try:
        ray.init(runtime_env=runtime_env, ignore_reinit_error=True)
        log_message("\nRay initialized successfully.")
    except Exception as e:
        log_message(f"Error initializing Ray: {str(e)}", level="ERROR")
        status_code = 0  # Error initializing Ray
        return status_code

    # Test import in main process
    log_message("\nTesting import in main process:")
    try:
        import cupy

        log_message("CuPy is available.")

        try:
            from cellvit.inference.postprocessing_cupy import DetectionCellPostProcessor

            log_message("\nSuccessfully imported cellvit (cupy) in main!")
            result = "Success"
        except Exception as e:
            result = f"Failed to import cellvit in main: {str(e)}"
            log_message(result)

    except ImportError:
        result = "CuPy is not available."
        log_message(result)
        try:
            log_message("\nSuccessfully imported cellvit (numpy) in main!")
            result = "Success"
        except Exception as e:
            result = f"Failed to import cellvit in main: {str(e)}"
            log_message(result)

    # Test import in Ray worker
    log_message("\nTesting import in Ray worker:")
    try:
        result = ray.get(test_import.remote())
        log_message(f"Worker result: {result}")
        if "Failed" in result:
            status_code = 0  # Import failed in worker
    except Exception as e:
        log_message(f"Ray worker test failed: {str(e)}", level="ERROR")
        status_code = 0  # Worker test failed

    # Shutdown Ray
    ray.shutdown()
    log_message("\nRay has been shut down.")

    return status_code


if __name__ == "__main__":
    status_code = main()
    print(status_code)
