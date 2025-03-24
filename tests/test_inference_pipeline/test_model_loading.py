# -*- coding: utf-8 -*-
# Test model loading for CellViT
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import os

os.environ["CELLVIT_CACHE"] = "tests/test_data/temp_cache"

import unittest
from unittest.mock import patch, MagicMock

from cellvit.inference.inference import CellViTInference
from cellvit.utils.ressource_manager import SystemConfiguration
from cellvit.config.config import TYPE_NUCLEI_DICT_PANNUKE
import shutil
from cellvit.models.cell_segmentation.cellvit_256 import CellViT256
from cellvit.models.cell_segmentation.cellvit_sam import CellViTSAM


class TestCellViTInferenceModel(unittest.TestCase):
    """Test for the _load_classifier method of the CellViTInference class."""

    def setUp(self):
        """Setup for each test: Create a mock object for the CellViTInference class."""
        self.mock_system_config = MagicMock(spec=SystemConfiguration)
        self.mock_system_config.__getitem__.return_value = 0

        with patch.object(CellViTInference, "__init__", return_value=None):
            self.inference = CellViTInference(
                model_name="SAM",
                outdir="dummy_path",
                system_configuration=self.mock_system_config,
            )

        self.inference.logger = MagicMock()
        self.inference.model_name = "SAM"
        self.inference.label_map = TYPE_NUCLEI_DICT_PANNUKE
        self.inference.binary = False
        self.inference.device = "cpu"

    def tearDown(self):
        from cellvit.config.config import CACHE_DIR

        # Clean up the temporary cache directory
        if os.path.exists(CACHE_DIR):
            shutil.rmtree(CACHE_DIR)

    def test_load_sam(self):
        """Test for loading SAM classifier."""
        self.inference.model_name = "SAM"
        self.inference._load_model()

        self.assertEqual(self.inference.model_arch, "CellViTSAM")
        self.assertIsInstance(self.inference.model, CellViTSAM)

    def test_load_hipt(self):
        """Test for loading the binary classifier."""
        self.inference.model_name = "HIPT"
        self.inference._load_model()

        self.assertEqual(self.inference.model_arch, "CellViT256")
        self.assertIsInstance(self.inference.model, CellViT256)


if __name__ == "__main__":
    unittest.main()
