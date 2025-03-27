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
from cellvit.utils.cache_test_database import cache_test_database

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent

cache_test_database(run_dir=str(parent_dir))

os.environ["DATABASE_DIR"] = f"{str(parent_dir)}/test_database"


def pytest_configure(config):
    print("Setting up test environment in conftest.py")
