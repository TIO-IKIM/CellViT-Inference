# -*- coding: utf-8 -*-
# Test WSI Metadata Loading for CellViT
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import unittest
from pathlib import Path
from cellvit.data.dataclass.wsi_meta import load_wsi_meta
import logging


class TestWSIMeta(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("TestWSIMeta")
        self.test_data_path = Path("./tests/data/test_wsi_database")
        self.svs_file_1 = self.test_data_path / "CMU-1-Small-Region.svs"
        self.svs_file_2 = self.test_data_path / "JP2K-33003-2.svs"
        self.tiff_file = self.test_data_path / "Philips-1.tiff"

    def test_load_wsi_meta_with_svs_file_1(self):
        """Test loading WSI metadata from SVS file 1."""
        slide_properties, target_mpp = load_wsi_meta(self.svs_file_1, self.logger)
        self.assertEqual(slide_properties["mpp"], 0.499)  # Fill in expected value
        self.assertEqual(
            slide_properties["magnification"], 20.0
        )  # Fill in expected value
        self.assertEqual(target_mpp, 0.25)  # Fill in expected value

    def test_load_wsi_meta_with_svs_file_2(self):
        """Test loading WSI metadata from SVS file 2."""
        slide_properties, target_mpp = load_wsi_meta(self.svs_file_2, self.logger)
        self.assertEqual(slide_properties["mpp"], 0.2498)  # Fill in expected value
        self.assertEqual(
            slide_properties["magnification"], 40.0
        )  # Fill in expected value
        self.assertEqual(target_mpp, 0.2498)  # Fill in expected value

    def test_load_wsi_meta_with_tiff_file(self):
        slide_properties, target_mpp = load_wsi_meta(self.tiff_file, self.logger)
        """Test loading WSI metadata from TIFF file."""
        self.assertEqual(slide_properties["mpp"], 0.226907)  # Fill in expected value
        self.assertEqual(
            slide_properties["magnification"], 40.0
        )  # Fill in expected value
        self.assertEqual(target_mpp, 0.226907)  # Fill in expected value

    def test_load_wsi_meta_with_custom_mpp(self):
        """Test loading WSI metadata with custom MPP."""
        custom_mpp = 0.25
        slide_properties, target_mpp = load_wsi_meta(
            self.svs_file_2, self.logger, wsi_mpp=custom_mpp
        )
        self.assertEqual(slide_properties["mpp"], custom_mpp)
        self.assertEqual(target_mpp, 0.25)

    def test_load_wsi_meta_with_custom_magnification(self):
        """Test loading WSI metadata with custom magnification."""
        custom_magnification = 20.0
        slide_properties, target_mpp = load_wsi_meta(
            self.svs_file_2, self.logger, wsi_magnification=custom_magnification
        )
        self.assertEqual(slide_properties["magnification"], custom_magnification)


if __name__ == "__main__":
    unittest.main()
