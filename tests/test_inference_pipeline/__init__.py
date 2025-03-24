import unittest
from unittest.mock import patch, MagicMock, PropertyMock
import torch
import pathlib
from pathlib import Path

from cellvit.inference.inference import CellViTInference
from cellvit.utils.ressource_manager import SystemConfiguration
from cellvit.config.config import TYPE_NUCLEI_DICT_PANNUKE


class TestCellViTInference_LoadClassifier(unittest.TestCase):
    """Tests für die _load_classifier Methode der CellViTInference Klasse."""

    def setUp(self):
        """Setup für jeden Test: Erstelle ein Mock-Objekt für die CellViTInference-Klasse."""
        # Erstelle ein Mock für SystemConfiguration
        self.mock_system_config = MagicMock(spec=SystemConfiguration)
        self.mock_system_config.__getitem__.return_value = 0  # gpu_index = 0
        
        # Patch die __init__ Methode, um sie zu umgehen
        with patch.object(CellViTInference, '__init__', return_value=None):
            self.inference = CellViTInference(
                model_name="SAM",
                outdir="dummy_path",
                system_configuration=self.mock_system_config
            )
        
        # Setze die benötigten Attribute manuell
        self.inference.logger = MagicMock()
        self.inference.model_name = "SAM"  # oder "HIPT"
        self.inference.label_map = TYPE_NUCLEI_DICT_PANNUKE
        self.inference.binary = False
    
    @patch('cellvit.inference.inference.cache_classifier')
    def test_load_classifier_pannuke(self, mock_cache_classifier):
        """Test für das Laden des Standard-PanNuke-Klassifikators."""
        self.inference.nuclei_taxonomy = "pannuke"
        self.inference._load_classifier()
        
        # Prüfe, ob info-Meldung geloggt wurde
        self.inference.logger.info.assert_called_with("Using default PanNuke classes")
        
        # Prüfe, ob cache_classifier nicht aufgerufen wurde
        mock_cache_classifier.assert_not_called()
        
        # Prüfe, ob binary nicht gesetzt wurde
        self.assertFalse(self.inference.binary)
    
    @patch('cellvit.inference.inference.cache_classifier')
    def test_load_classifier_binary(self, mock_cache_classifier):
        """Test für das Laden des binären Klassifikators."""
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
    
    @patch('cellvit.inference.inference.cache_classifier')
    @patch('torch.load')
    @patch('cellvit.inference.inference.LinearClassifier')
    @patch('pathlib.Path.exists', return_value=True)
    def test_load_classifier_lizard(self, mock_exists, mock_linear_classifier, 
                                   mock_torch_load, mock_cache_classifier):
        """Test für das Laden des lizard-Klassifikators."""
        # Setup
        self.inference.nuclei_taxonomy = "lizard"
        self.inference.model_name = "SAM"
        
        # Mocks für die Pfad-Konstruktion
        mock_cache_path = MagicMock(spec=Path)
        mock_cache_path.__truediv__ = MagicMock(return_value=MagicMock(spec=Path))
        mock_cache_classifier.return_value = mock_cache_path
        
        # Mock für torch.load
        mock_checkpoint = {
            'config': {'data.label_map': {'0': 'label1', '1': 'label2'}},
            'model_state_dict': {'fc1.weight': torch.zeros((10, 512))}
        }
        mock_torch_load.return_value = mock_checkpoint
        
        # Mock für den LinearClassifier
        mock_model = MagicMock()
        mock_linear_classifier.return_value = mock_model
        
        # Führe die Methode aus
        self.inference._load_classifier()
        
        # Prüfungen
        self.inference.logger.info.assert_any_call("Using Lizard classifier")
        mock_cache_classifier.assert_called_once_with(self.inference.logger)
        mock_torch_load.assert_called_once()
        mock_linear_classifier.assert_called_once()
        mock_model.load_state_dict.assert_called_once()
        mock_model.eval.assert_called_once()
        self.assertEqual(mock_model, self.inference.classifier)
    
    @patch('cellvit.inference.inference.cache_classifier')
    def test_load_classifier_unknown(self, mock_cache_classifier):
        """Test für das Laden eines unbekannten Klassifikators."""
        # Setup
        self.inference.nuclei_taxonomy = "unknown_taxonomy"
        
        # Mock für cache_classifier
        mock_cache_path = MagicMock(spec=Path)
        mock_cache_path.__truediv__ = MagicMock(return_value=MagicMock(spec=Path))
        mock_cache_classifier.return_value = mock_cache_path
        
        # Führe die Methode aus, sollte eine Error-Meldung loggen
        with self.assertRaises(AssertionError):  # Da der Pfad nicht existiert
            self.inference._load_classifier()
        
        # Prüfe, ob error-Meldung geloggt wurde
        self.inference.logger.error.assert_called_with(
            f"Unknown classifier: {self.inference.nuclei_taxonomy}, using default settings"
        )


if __name__ == "__main__":
    unittest.main()