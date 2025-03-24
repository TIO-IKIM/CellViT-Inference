# -*- coding: utf-8 -*-
# Test classifier loading for CellViT
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import os

os.environ["CELLVIT_CACHE"] = "tests/test_data/temp_cache"

import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

from cellvit.inference.inference import CellViTInference, LinearClassifier
from cellvit.utils.ressource_manager import SystemConfiguration
from cellvit.config.config import TYPE_NUCLEI_DICT_PANNUKE
import shutil


class TestCellViTInferenceLoadClassifier(unittest.TestCase):
    """Test for the _load_classifier method of the CellViTInference class."""

    def setUp(self):
        """Setup for each test: Create a mock object for the CellViTInference class."""
        # Erstelle ein Mock für SystemConfiguration
        self.mock_system_config = MagicMock(spec=SystemConfiguration)
        self.mock_system_config.__getitem__.return_value = 0  # gpu_index = 0

        # Patch die __init__ Methode, um sie zu umgehen
        with patch.object(CellViTInference, "__init__", return_value=None):
            self.inference = CellViTInference(
                model_name="SAM",
                outdir="dummy_path",
                system_configuration=self.mock_system_config,
            )

        # Setze die benötigten Attribute manuell
        self.inference.logger = MagicMock()
        self.inference.model_name = "SAM"  # oder "HIPT"
        self.inference.label_map = TYPE_NUCLEI_DICT_PANNUKE
        self.inference.binary = False

    def tearDown(self):
        from cellvit.config.config import CACHE_DIR

        # Clean up the temporary cache directory
        if os.path.exists(CACHE_DIR):
            shutil.rmtree(CACHE_DIR)

    @patch("cellvit.inference.inference.cache_classifier")
    def test_load_classifier_pannuke(self, mock_cache_classifier):
        """Test for loading the default PanNuke classifier."""
        self.inference.nuclei_taxonomy = "pannuke"
        self.inference._load_classifier()

        # Prüfe, ob info-Meldung geloggt wurde
        self.inference.logger.info.assert_called_with("Using default PanNuke classes")

        # Prüfe, ob cache_classifier nicht aufgerufen wurde
        mock_cache_classifier.assert_not_called()

        # Prüfe, ob binary nicht gesetzt wurde
        self.assertFalse(self.inference.binary)

    @patch("cellvit.inference.inference.cache_classifier")
    def test_load_classifier_binary(self, mock_cache_classifier):
        """Test for loading the binary classifier."""
        self.inference.nuclei_taxonomy = "binary"
        self.inference._load_classifier()

        # Prüfe, ob binary gesetzt wurde
        self.assertTrue(self.inference.binary)

        # Prüfe, ob label_map korrekt gesetzt wurde
        self.assertEqual(self.inference.label_map, {0: "background", 1: "nuclei"})

        # Prüfe, ob info-Meldung geloggt wurde
        self.inference.logger.info.assert_called_with("Using binary detection")

        # Prüfe, ob cache_classifier nicht aufgerufen wurde
        mock_cache_classifier.assert_not_called()

    def test_load_classifier_lizard_sam(self):
        """Test for loading the lizard classifier."""
        # Setup
        self.inference.nuclei_taxonomy = "lizard"
        self.inference.model_name = "SAM"

        self.inference._load_classifier()

        # Prüfungen
        self.inference.logger.info.assert_any_call("Using Lizard classifier")
        self.assertEqual(
            self.inference.label_map,
            {
                0: "Neutrophil",
                1: "Epithelial",
                2: "Lymphocyte",
                3: "Plasma",
                4: "Eosinophil",
                5: "Connective tissue",
            },
            "Label map should be set correctly",
        )
        self.assertIsInstance(
            self.inference.classifier,
            LinearClassifier,
            "Classifier should be a PyTorch module",
        )

    def test_load_classifier_midog_sam(self):
        """Test for loading the midog classifier."""
        # Setup
        self.inference.nuclei_taxonomy = "midog"
        self.inference.model_name = "SAM"

        self.inference._load_classifier()

        # Prüfungen
        self.inference.logger.info.assert_any_call("Using MIDOG classifier")
        self.assertEqual(
            self.inference.label_map,
            {0: "Mitotic", 1: "Hard-Negative"},
            "Label map should be set correctly",
        )
        self.assertIsInstance(
            self.inference.classifier,
            LinearClassifier,
            "Classifier should be a PyTorch module",
        )

    def test_load_classifier_consep_sam(self):
        """Test for loading the consep classifier."""
        # Setup
        self.inference.nuclei_taxonomy = "consep"
        self.inference.model_name = "SAM"

        self.inference._load_classifier()

        # Prüfungen
        self.inference.logger.info.assert_any_call("Using CoNSeP classifier")
        self.assertEqual(
            self.inference.label_map,
            {0: "Other", 1: "Inflammatory", 2: "Epithelial", 3: "Spindle-Shaped"},
            "Label map should be set correctly",
        )
        self.assertIsInstance(
            self.inference.classifier,
            LinearClassifier,
            "Classifier should be a PyTorch module",
        )

    def test_load_classifier_nucls_main_sam(self):
        """Test for loading the nucls_main classifier."""
        # Setup
        self.inference.nuclei_taxonomy = "nucls_main"
        self.inference.model_name = "SAM"

        self.inference._load_classifier()

        # Prüfungen
        self.inference.logger.info.assert_any_call("Using NUCLS Main classifier")
        self.assertEqual(
            self.inference.label_map,
            {
                0: "Tumor nonMitotic",
                1: "Tumor Mitotic",
                2: "nonTILnonMQ Stromal",
                3: "Macrophage",
                4: "Lymphocyte",
                5: "Plasma Cell",
                6: "Other Nucleus",
            },
            "Label map should be set correctly",
        )
        self.assertIsInstance(
            self.inference.classifier,
            LinearClassifier,
            "Classifier should be a PyTorch module",
        )

    def test_load_classifier_nucls_super_sam(self):
        """Test for loading the nucls_super classifier."""
        # Setup
        self.inference.nuclei_taxonomy = "nucls_super"
        self.inference.model_name = "SAM"

        self.inference._load_classifier()

        # Prüfungen
        self.inference.logger.info.assert_any_call("Using NUCLS Super classifier")
        self.assertEqual(
            self.inference.label_map,
            {0: "Tumor", 1: "nonTIL Stromal", 2: "sTIL", 3: "Other"},
            "Label map should be set correctly",
        )
        self.assertIsInstance(
            self.inference.classifier,
            LinearClassifier,
            "Classifier should be a PyTorch module",
        )

    def test_load_classifier_ocelot_sam(self):
        """Test for loading the ocelot classifier."""
        # Setup
        self.inference.nuclei_taxonomy = "ocelot"
        self.inference.model_name = "SAM"

        self.inference._load_classifier()

        # Prüfungen
        self.inference.logger.info.assert_any_call("Using Ocelot classifier")
        self.assertEqual(
            self.inference.label_map,
            {0: "Other Cell", 1: "Tumor Cell"},
            "Label map should be set correctly",
        )
        self.assertIsInstance(
            self.inference.classifier,
            LinearClassifier,
            "Classifier should be a PyTorch module",
        )

    def test_load_classifier_panoptils_sam(self):
        """Test for loading the panoptils classifier."""
        # Setup
        self.inference.nuclei_taxonomy = "panoptils"
        self.inference.model_name = "SAM"

        self.inference._load_classifier()

        # Prüfungen
        self.inference.logger.info.assert_any_call("Using Panoptils classifier")
        self.assertEqual(
            self.inference.label_map,
            {0: "Other Cells", 1: "Epithelial Cells", 2: "Stromal Cells", 3: "TILs"},
            "Label map should be set correctly",
        )
        self.assertIsInstance(
            self.inference.classifier,
            LinearClassifier,
            "Classifier should be a PyTorch module",
        )

    def test_load_classifier_lizard_hipt(self):
        """Test for loading the lizard classifier."""
        # Setup
        self.inference.nuclei_taxonomy = "lizard"
        self.inference.model_name = "HIPT"

        self.inference._load_classifier()

        # Prüfungen
        self.inference.logger.info.assert_any_call("Using Lizard classifier")
        self.assertEqual(
            self.inference.label_map,
            {
                0: "Neutrophil",
                1: "Epithelial",
                2: "Lymphocyte",
                3: "Plasma",
                4: "Eosinophil",
                5: "Connective tissue",
            },
            "Label map should be set correctly",
        )
        self.assertIsInstance(
            self.inference.classifier,
            LinearClassifier,
            "Classifier should be a PyTorch module",
        )

    def test_load_classifier_midog_hipt(self):
        """Test for loading the midog classifier."""
        # Setup
        self.inference.nuclei_taxonomy = "midog"
        self.inference.model_name = "HIPT"

        expected_message = r"Classifier cannot be loaded - Expecting .*"
        with self.assertRaisesRegex(AssertionError, expected_message):
            self.inference._load_classifier()

    def test_load_classifier_consep_hipt(self):
        """Test for loading the consep classifier."""
        # Setup
        self.inference.nuclei_taxonomy = "consep"
        self.inference.model_name = "HIPT"

        self.inference._load_classifier()

        # Prüfungen
        self.inference.logger.info.assert_any_call("Using CoNSeP classifier")
        self.assertEqual(
            self.inference.label_map,
            {0: "Other", 1: "Inflammatory", 2: "Epithelial", 3: "Spindle-Shaped"},
            "Label map should be set correctly",
        )
        self.assertIsInstance(
            self.inference.classifier,
            LinearClassifier,
            "Classifier should be a PyTorch module",
        )

    def test_load_classifier_nucls_main_hipt(self):
        """Test for loading the nucls_main classifier."""
        # Setup
        self.inference.nuclei_taxonomy = "nucls_main"
        self.inference.model_name = "HIPT"

        self.inference._load_classifier()

        # Prüfungen
        self.inference.logger.info.assert_any_call("Using NUCLS Main classifier")
        self.assertEqual(
            self.inference.label_map,
            {
                0: "Tumor nonMitotic",
                1: "Tumor Mitotic",
                2: "nonTILnonMQ Stromal",
                3: "Macrophage",
                4: "Lymphocyte",
                5: "Plasma Cell",
                6: "Other Nucleus",
            },
            "Label map should be set correctly",
        )
        self.assertIsInstance(
            self.inference.classifier,
            LinearClassifier,
            "Classifier should be a PyTorch module",
        )

    def test_load_classifier_nucls_super_hipt(self):
        """Test for loading the nucls_super classifier."""
        # Setup
        self.inference.nuclei_taxonomy = "nucls_super"
        self.inference.model_name = "HIPT"

        self.inference._load_classifier()

        # Prüfungen
        self.inference.logger.info.assert_any_call("Using NUCLS Super classifier")
        self.assertEqual(
            self.inference.label_map,
            {0: "Tumor", 1: "nonTIL Stromal", 2: "sTIL", 3: "Other"},
            "Label map should be set correctly",
        )
        self.assertIsInstance(
            self.inference.classifier,
            LinearClassifier,
            "Classifier should be a PyTorch module",
        )

    def test_load_classifier_ocelot_hipt(self):
        """Test for loading the ocelot classifier."""
        # Setup
        self.inference.nuclei_taxonomy = "ocelot"
        self.inference.model_name = "HIPT"

        self.inference._load_classifier()

        # Prüfungen
        self.inference.logger.info.assert_any_call("Using Ocelot classifier")
        self.assertEqual(
            self.inference.label_map,
            {0: "Other Cell", 1: "Tumor Cell"},
            "Label map should be set correctly",
        )
        self.assertIsInstance(
            self.inference.classifier,
            LinearClassifier,
            "Classifier should be a PyTorch module",
        )

    def test_load_classifier_panoptils_hipt(self):
        """Test for loading the panoptils classifier."""
        # Setup
        self.inference.nuclei_taxonomy = "panoptils"
        self.inference.model_name = "HIPT"

        self.inference._load_classifier()

        # Prüfungen
        self.inference.logger.info.assert_any_call("Using Panoptils classifier")
        self.assertEqual(
            self.inference.label_map,
            {0: "Other Cells", 1: "Epithelial Cells", 2: "Stromal Cells", 3: "TILs"},
            "Label map should be set correctly",
        )
        self.assertIsInstance(
            self.inference.classifier,
            LinearClassifier,
            "Classifier should be a PyTorch module",
        )

    @patch("cellvit.inference.inference.cache_classifier")
    def test_load_classifier_no_path(self, mock_cache_classifier):
        """Test for loading the classifier when no path is provided."""
        # Setup
        self.inference.nuclei_taxonomy = "unknown_taxonomy"

        # Mock für cache_classifier
        mock_cache_path = MagicMock(spec=Path)
        mock_cache_path.__truediv__ = MagicMock(return_value=MagicMock(spec=Path))
        mock_cache_classifier.return_value = mock_cache_path

        with self.assertRaisesRegex(NotImplementedError, "Unknown classifier"):
            self.inference._load_classifier()

        # Prüfe, ob error-Meldung geloggt wurde
        # self.as


if __name__ == "__main__":
    unittest.main()
