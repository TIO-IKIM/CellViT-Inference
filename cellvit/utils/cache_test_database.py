# -*- coding: utf-8 -*-
# Cache CellViT models and classifier
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen


import os

from cellvit.utils.download import check_and_download
from cellvit.utils.logger import PrintLogger

import zipfile
from pathlib import Path
from typing import Optional
import logging


def cache_test_database(
    run_dir: str = os.path.abspath(os.getcwd()), logger: Optional[logging.Logger] = None
) -> Path:
    """Download and cache the classifier.zip file, if not already cached.

    Args:
        logger (Optional[logging.Logger], optional): Logger. Defaults to None.

    Returns:
        Path: Path to the cached classifier directory.
    """
    zip_dir = Path(run_dir) / "test_database.zip"
    database_dir = Path(run_dir) / "test_database"

    logger = logger or PrintLogger()

    if not database_dir.exists():
        check_and_download(
            directory_path=run_dir,
            file_name="test_database.zip",
            download_link="https://zenodo.org/records/15094831/files/test_database.zip",
            logger=logger,
        )
        with zipfile.ZipFile(zip_dir, "r") as zip_ref:
            zip_ref.extractall(run_dir)
        os.remove(zip_dir)
    else:
        if zip_dir.exists():
            os.remove(zip_dir)


if __name__ == "__main__":
    # Example usage
    cache_test_database()
