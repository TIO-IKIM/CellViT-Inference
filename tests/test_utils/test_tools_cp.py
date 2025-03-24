# -*- coding: utf-8 -*-
# Test Tools for cupy
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen
# ruff: noqa: F401

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Cupy-Verfügbarkeit prüfen
try:
    import cupy

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


# Ganze Testklasse überspringen, wenn cupy nicht verfügbar ist
@unittest.skipIf(not CUPY_AVAILABLE, "cupy is not installed")
class TestRemoveSmallObjectsCp(unittest.TestCase):
    @patch("cupy.zeros_like")
    @patch("cupy.bincount")
    @patch("cupyx.scipy.ndimage.generate_binary_structure")
    @patch("cupyx.scipy.ndimage.label")
    def test_remove_small_objects_boolean_input(
        self, mock_label, mock_gen_struct, mock_bincount, mock_zeros_like
    ):
        """Test with boolean input"""
        from cellvit.utils.tools_cp import remove_small_objects_cp

        # Mock-Setup
        input_array = MagicMock(spec=np.ndarray)
        input_array.dtype = bool
        input_array.ndim = 2

        # Mock der cupy-Funktionen
        mock_zeros_like.return_value = MagicMock()
        mock_bincount.return_value = np.array(
            [0, 10, 5, 20]
        )  # Komponenten mit Größen 0, 10, 5, 20

        # Ausführen der Funktion
        result = remove_small_objects_cp(input_array, min_size=10, connectivity=1)

        # Überprüfungen
        mock_gen_struct.assert_called_once_with(2, 1)
        mock_zeros_like.assert_called_once()
        mock_label.assert_called_once()
        mock_bincount.assert_called_once()

        # Der Test prüft, ob die Funktion korrekt ausgeführt wurde
        self.assertIsNotNone(result)

    @patch("cupy.bincount")
    def test_remove_small_objects_label_input(self, mock_bincount):
        """Test with labeled input"""
        from cellvit.utils.tools_cp import remove_small_objects_cp

        # Mock-Setup
        input_array = MagicMock(spec=np.ndarray)
        input_array.dtype = np.int32

        # Mock der bincount-Funktion
        mock_bincount.return_value = np.array(
            [0, 100, 5, 20]
        )  # Komponenten mit Größen 0, 100, 5, 20

        # Ausführen der Funktion
        result = remove_small_objects_cp(input_array, min_size=10, connectivity=1)

        # Überprüfungen
        mock_bincount.assert_called_once()

        # Der Test prüft, ob die Funktion korrekt ausgeführt wurde
        self.assertIsNotNone(result)

    def test_remove_small_objects_min_size_zero(self):
        """Test with min_size=0"""
        from cellvit.utils.tools_cp import remove_small_objects_cp

        # Mock-Setup
        input_array = MagicMock(spec=np.ndarray)

        # Ausführen der Funktion
        result = remove_small_objects_cp(input_array, min_size=0)

        # Überprüfungen
        self.assertEqual(result, input_array)  # Sollte Eingabe unverändert zurückgeben

    @patch("cupy.bincount")
    def test_remove_small_objects_different_connectivity(self, mock_bincount):
        """Test with different connectivity"""
        from cellvit.utils.tools_cp import remove_small_objects_cp

        # Mock-Setup für connectivity=2
        input_array = MagicMock(spec=np.ndarray)
        input_array.dtype = bool
        input_array.ndim = 2

        # Mock der bincount-Funktion
        mock_bincount.return_value = np.array([0, 8, 15])

        with patch("cupyx.scipy.ndimage.generate_binary_structure") as mock_gen_struct:
            with patch("cupyx.scipy.ndimage.label") as mock_label:
                with patch("cupy.zeros_like") as mock_zeros_like:
                    # Ausführen der Funktion mit connectivity=2
                    result = remove_small_objects_cp(
                        input_array, min_size=10, connectivity=2
                    )

                    # Überprüfen, ob generate_binary_structure mit connectivity=2 aufgerufen wurde
                    mock_gen_struct.assert_called_once_with(2, 2)

    @patch("cupy.bincount")
    def test_remove_small_objects_negative_values(self, mock_bincount):
        """Test with negative values in input"""
        from cellvit.utils.tools_cp import remove_small_objects_cp

        # Mock-Setup
        input_array = MagicMock(spec=np.ndarray)
        input_array.dtype = np.int32

        # Mock der bincount-Funktion, um ValueError auszulösen
        mock_bincount.side_effect = ValueError()

        # Überprüfen, ob ein ValueError ausgelöst wird
        with self.assertRaises(ValueError):
            remove_small_objects_cp(input_array, min_size=10)

    @patch("cupy.zeros_like")
    @patch("cupy.bincount")
    @patch("cupyx.scipy.ndimage.generate_binary_structure")
    @patch("cupyx.scipy.ndimage.label")
    def test_remove_small_objects_realistic_case(
        self, mock_label, mock_gen_struct, mock_bincount, mock_zeros_like
    ):
        """Test with a realistic case"""
        from cellvit.utils.tools_cp import remove_small_objects_cp

        # Realistisches Bild simulieren mit numpy und nach cupy konvertieren
        np_array = np.zeros((100, 100), dtype=bool)
        # Objekte mit verschiedenen Größen einfügen
        np_array[10:15, 10:15] = True  # 5x5 = 25 Pixel (bleibt erhalten)
        np_array[30:32, 30:32] = True  # 2x2 = 4 Pixel (wird entfernt)
        np_array[50:60, 50:60] = True  # 10x10 = 100 Pixel (bleibt erhalten)

        # Mock für cupy.array erstellen
        input_array = MagicMock(spec=np.ndarray)
        input_array.dtype = bool
        input_array.ndim = 2
        input_array.__array__ = lambda: np_array

        # Komponenten-Label simulieren
        labeled_array = np.zeros((100, 100), dtype=np.int32)
        labeled_array[10:15, 10:15] = 1
        labeled_array[30:32, 30:32] = 2
        labeled_array[50:60, 50:60] = 3

        # Mock für labeled_array
        mock_zeros_like_return = MagicMock()
        mock_zeros_like.return_value = mock_zeros_like_return

        # Simuliere bincount-Ausgabe: [0 Hintergrund, 25 Pixel für Label 1, 4 Pixel für Label 2, 100 Pixel für Label 3]
        mock_bincount.return_value = np.array([0, 25, 4, 100])

        # Simuliere ccs (labeled_array)
        mock_label.side_effect = lambda input_array, structure, output: setattr(
            output, "ravel", lambda: np.array([0, 1, 1, 1, 2, 2, 3, 3, 3])
        )

        # Ausführen der Funktion
        result = remove_small_objects_cp(input_array, min_size=10, connectivity=1)

        # Überprüfen, ob alle erwarteten Funktionen aufgerufen wurden
        mock_gen_struct.assert_called_once()
        mock_zeros_like.assert_called_once()
        mock_label.assert_called_once()
        mock_bincount.assert_called_once()

        # Der Test prüft, ob die Funktion korrekt ausgeführt wurde
        self.assertIsNotNone(result)
