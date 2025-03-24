# -*- coding: utf-8 -*-
# Set up test environment by checking and downloading the test database.
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import sys
import os

os.environ["CELLVIT_CACHE"] = "tests/test_data/temp_cache"

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from pathlib import Path
from cellvit.utils.download import check_and_download


def check_test_database() -> None:
    """Check if the test database exists, and download it if it does not exist."""
    print("Checking Test Database")
    base_path = Path("./tests/data/test_wsi_database")
    if not base_path.exists():
        base_path.mkdir(parents=True, exist_ok=True)
    check_and_download(
        base_path,
        "CMU-1-Small-Region.svs",
        "https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1-Small-Region.svs",
    )
    check_and_download(
        base_path,
        "JP2K-33003-2.svs",
        "https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/JP2K-33003-2.svs",
    )
    check_and_download(
        base_path,
        "Philips-1.tiff",
        "https://openslide.cs.cmu.edu/download/openslide-testdata/Philips-TIFF/Philips-1.tiff",
    )
    print("Test Database is now cached on local machine.")


check_test_database()


def pytest_configure(config):
    print("Setting up test environment in conftest.py")
