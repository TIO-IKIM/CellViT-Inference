# -*- coding: utf-8 -*-
# Test Module Check for CellViT
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import unittest
from unittest.mock import MagicMock, patch

from cellvit.utils.check_module import check_module, perform_module_check


class TestCheckModule(unittest.TestCase):
    @patch("cellvit.utils.check_module.importlib.util.find_spec")
    def test_check_module_installed(self, mock_find_spec):
        """Test if a module is installed."""
        mock_find_spec.return_value = MagicMock()
        self.assertTrue(
            check_module("some_installed_module"), "Module should be installed"
        )

    @patch("cellvit.utils.check_module.importlib.util.find_spec")
    def test_check_module_not_installed(self, mock_find_spec):
        """Test if a module is not installed."""
        mock_find_spec.return_value = None
        self.assertFalse(
            check_module("some_nonexistent_module"), "Module should not be installed"
        )

    @patch("cellvit.utils.check_module.check_module")
    @patch("cellvit.utils.check_module.NullLogger")
    def test_perform_module_check_installed(self, mock_null_logger, mock_check_module):
        """Test perform_module_check with installed module."""
        mock_logger = MagicMock()
        mock_null_logger.return_value = mock_logger
        mock_check_module.return_value = True

        perform_module_check("some_installed_module", logger=None)

        # Correct assertions
        mock_logger.info.assert_any_call("Checking library some_installed_module")
        mock_logger.info.assert_any_call("Module installed and loaded")

    @patch("cellvit.utils.check_module.check_module")
    @patch("cellvit.utils.check_module.NullLogger")
    def test_perform_module_check_not_installed(
        self, mock_null_logger, mock_check_module
    ):
        """Test perform_module_check with not installed module."""
        mock_logger = MagicMock()
        mock_null_logger.return_value = mock_logger
        mock_check_module.return_value = False

        perform_module_check("some_nonexistent_module", logger=None)
        mock_logger.info.assert_any_call("Checking library some_nonexistent_module")
        mock_logger.error.assert_any_call("Failed loading some_nonexistent_module")

    @patch("cellvit.utils.check_module.check_module")
    @patch("cellvit.utils.check_module.NullLogger")
    def test_perform_module_check_import_error(
        self, mock_null_logger, mock_check_module
    ):
        """Test perform_module_check with ImportError."""
        mock_logger = MagicMock()
        mock_null_logger.return_value = mock_logger
        mock_check_module.side_effect = ImportError("Test ImportError")

        with self.assertRaises(ImportError):
            perform_module_check("some_faulty_module", logger=None)
        mock_logger.info.assert_any_call("Checking library some_faulty_module")
        mock_logger.error.assert_any_call("Error: Test ImportError")


if __name__ == "__main__":
    unittest.main()
