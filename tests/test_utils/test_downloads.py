# -*- coding: utf-8 -*-
# Test Cache Models for CellViT
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import os

os.environ["CELLVIT_CACHE"] = "tests/test_data/temp_cache"

import shutil
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from cellvit.config.config import CACHE_DIR
from cellvit.utils.cache_models import (
    cache_cellvit_256,
    cache_cellvit_sam_h,
    cache_classifier,
)


@pytest.mark.slow
class TestCacheModels(unittest.TestCase):
    def tearDown(self):
        # Clean up the temporary cache directory
        if os.path.exists(CACHE_DIR):
            shutil.rmtree(CACHE_DIR)

    def test_cache_cellvit_sam_h(self):
        """Test the caching of the CellViT-SAM-H-x40-AMP model file."""
        mock_logger = MagicMock()
        expected_path = Path(CACHE_DIR) / "CellViT-SAM-H-x40-AMP.pth"

        result = cache_cellvit_sam_h(logger=mock_logger)

        self.assertTrue(expected_path.exists(), "Cache file should exist")
        self.assertEqual(result, expected_path, "Cache path should be the same")

    def test_cache_cellvit_256(self):
        """Test the caching of the CellViT-256-x40-AMP model file."""
        mock_logger = MagicMock()
        expected_path = Path(CACHE_DIR) / "CellViT-256-x40-AMP.pth"

        result = cache_cellvit_256(logger=mock_logger)

        self.assertTrue(expected_path.exists())
        self.assertEqual(result, expected_path)

    def test_cache_exists(self):
        """Test if the cache file exists and is a file."""
        mock_logger = MagicMock()
        result = cache_cellvit_256(logger=mock_logger)
        self.assertTrue(result.exists(), "Cache file should exist")
        self.assertTrue(result.is_file(), "Cache path should be a file")

        start_time = time.time()
        result_2 = cache_cellvit_256(logger=mock_logger)
        end_time = time.time()
        execution_time = end_time - start_time
        self.assertLess(execution_time, 0.05, "Cache retrieval took too long")
        self.assertEqual(
            result, result_2, "Cache retrieval should return the same path"
        )

    def test_cache_classifier(self):
        """Test the caching of the classifier directory."""
        mock_logger = MagicMock()
        classifier_dir = Path(CACHE_DIR) / "classifier"
        zip_dir = Path(CACHE_DIR) / "classifier.zip"

        # Simulate the case where the classifier directory does not exist
        if classifier_dir.exists():
            for item in classifier_dir.iterdir():
                item.unlink()
            classifier_dir.rmdir()

        result = cache_classifier(logger=mock_logger)

        self.assertTrue(classifier_dir.exists())
        self.assertEqual(result, classifier_dir)

        # Simulate the case where the classifier directory already exists
        result = cache_classifier(logger=mock_logger)

        self.assertTrue(classifier_dir.exists(), "Classifier directory should exist")
        self.assertEqual(result, classifier_dir, "Cache directory should be the same")

        # check if the zip file is removed
        self.assertFalse(
            zip_dir.exists(), "Zip file should be removed after extraction"
        )


if __name__ == "__main__":
    unittest.main()
