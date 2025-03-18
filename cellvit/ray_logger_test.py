# -*- coding: utf-8 -*-
# CellViT Inference Pipeline for Whole Slide Images (WSI) in Memory
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen


from cellvit.utils.logger import Logger
import ray
import logging


@ray.remote
def test_ray_logging():
    """Test function to verify Ray logging configuration"""
    # Get logger instance inside the remote function
    ray_logger = logging.getLogger("ray")

    # Log test messages at different levels
    print("Check print")
    ray_logger.debug("Remote function debug message")
    ray_logger.info("Remote function info message")
    ray_logger.warning("Remote function warning message")
    ray_logger.error("Remote function error message")

    return True


def main():
    logger = Logger(
        level="DEBUG",
    )
    logger = logger.create_logger()
    logger.debug("Debug")
    logger.info("Info")

    # ray logger settings
    formatter = None
    if logger.handlers:
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and hasattr(
                handler, "formatter"
            ):
                formatter = handler.formatter
                break

    logger.info("Init Ray")
    ray.init(
        include_dashboard=True,
        logging_level=logger.level,
        log_to_driver=True,
    )
    if formatter is not None:
        ray_loggers = [
            logging.getLogger("ray"),
            logging.getLogger("ray.worker"),
            logging.getLogger("ray.remote_function"),
            logging.getLogger("ray._private"),  # Covers internal modules
        ]

        for ray_logger in ray_loggers:
            # Modify existing handlers (preserves Ray's logging destinations)
            for handler in ray_logger.handlers:
                if isinstance(handler, logging.StreamHandler):
                    handler.setFormatter(formatter)

    logger.info("Ray Ready")
    logger.debug("Debug")
    logger.info("Info")

    # Test Ray logging with remote function
    logger.info("Testing Ray logging with remote function...")
    ray.get(test_ray_logging.remote())  # <-- Add this line
    logger.info("Test completed")

    # Cleanup
    ray.shutdown()


if __name__ == "__main__":
    main()
