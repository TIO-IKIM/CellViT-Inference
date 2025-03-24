# -*- coding: utf-8 -*-
# Test if the CuPy checker is working correctly.
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import sys
import unittest
from unittest.mock import MagicMock, patch


class TestCheckCuPy(unittest.TestCase):
    @patch.dict("sys.modules", {"cupy": MagicMock()})
    def test_cupy_installed_and_working(self):
        """Test if CuPy is installed and working correctly."""
        from cellvit.utils.check_cupy import check_cupy

        # Konfigurieren Sie das Mock-CuPy-Modul
        mock_cupy = sys.modules["cupy"]
        mock_cupy.cuda.runtime.getDeviceCount.return_value = 1

        # Führen Sie die Funktion aus und überprüfen Sie das Ergebnis
        result = check_cupy(True)
        self.assertTrue(result, "CuPy should be detected as installed and working")

        # Verifizieren Sie, dass die entsprechenden Methoden aufgerufen wurden
        mock_cupy.cuda.runtime.getDeviceCount.assert_called_once()

    def test_cupy_not_installed(self):
        """Test that CuPy is not installed."""

        # Entfernen Sie cupy komplett aus sys.modules, falls es vorhanden ist
        orig_cupy = None
        if "cupy" in sys.modules:
            orig_cupy = sys.modules["cupy"]
            del sys.modules["cupy"]

        # Patchen Sie builtins.__import__, um eine ImportError für cupy auszulösen
        original_import = __import__

        def mock_import(name, *args, **kwargs):
            if name == "cupy":
                raise ImportError("No module named 'cupy'")
            return original_import(name, *args, **kwargs)

        try:
            with patch("builtins.__import__", side_effect=mock_import):
                from cellvit.utils.check_cupy import check_cupy

                # Führen Sie die Funktion aus und überprüfen Sie das Ergebnis
                result = check_cupy(True)
                self.assertFalse(result)
        finally:
            # Stellen Sie den ursprünglichen Zustand wieder her
            if orig_cupy is not None:
                sys.modules["cupy"] = orig_cupy

    @patch.dict("sys.modules", {"cupy": MagicMock()})
    def test_cupy_installed_but_not_working_gpu(self):
        """Test if CuPy is installed but not working correctly (GPU False)."""
        from cellvit.utils.check_cupy import check_cupy

        # Konfigurieren Sie das Mock-CuPy-Modul, um eine Exception auszulösen
        mock_cupy = sys.modules["cupy"]
        mock_cupy.cuda.runtime.getDeviceCount.side_effect = Exception("GPU error")

        # Führen Sie die Funktion aus und überprüfen Sie das Ergebnis
        result = check_cupy(True)
        self.assertFalse(result)

        # Verifizieren Sie, dass die entsprechenden Methoden aufgerufen wurden
        mock_cupy.cuda.runtime.getDeviceCount.assert_called_once()


if __name__ == "__main__":
    unittest.main()
