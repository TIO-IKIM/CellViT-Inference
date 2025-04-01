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
import csv


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
        filelist_path = database_dir / "filelist.csv"
        with open(filelist_path, mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["path", "wsi_mpp", "wsi_magnification"])
            writer.writerow(
                ["./test_database/BRACS/BRACS_1640_N_3_cropped.tiff", "0.25", "40"]
            )
            writer.writerow(["./test_database/x20_svs/CMU-1-Small-Region.svs", "", ""])

        os.remove(zip_dir)
    else:
        if zip_dir.exists():
            os.remove(zip_dir)


if __name__ == "__main__":
    # Example usage
    cache_test_database()
