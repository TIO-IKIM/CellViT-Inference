import unittest
import numpy as np
from cellvit.utils.tools import get_bounding_box, remove_small_objects, flatten_dict, unflatten_dict, get_size_of_dict, load_wsi_files_from_csv, close_logger, remove_parameter_tag, remap_label
import sys
import os
import pandas as pd
import logging
from unittest.mock import patch, MagicMock

class TestGetBoundingBox(unittest.TestCase):
    """Tests for the get_bounding_box function."""
    
    def test_get_bounding_box_all_zeros(self):
        """Test for an image with all zeros."""
        img = np.zeros((10, 10), dtype=int)
        result = get_bounding_box(img)
        self.assertEqual(result, [], 
                         "Bounding box for all zeros should be empty")
    
    def test_get_bounding_box_single_pixel(self):
        """Test for a single pixel."""
        img = np.zeros((10, 10), dtype=int)
        img[4, 5] = 1
        result = get_bounding_box(img)
        self.assertEqual(result, [4, 5, 5, 6], 
                         "Bounding box for single pixel should be [4, 5, 5, 6]")
    
    def test_get_bounding_box_rectangle(self):
        """Test for a rectangle in the image."""
        img = np.zeros((10, 10), dtype=int)
        img[2:6, 3:8] = 1
        result = get_bounding_box(img)
        self.assertEqual(result, [2, 6, 3, 8], 
                         "Bounding box for rectangle should be [2, 6, 3, 8]")
    
    def test_get_bounding_box_full_image(self):
        """Test für ein vollständig mit Einsen gefülltes Bild."""
        img = np.ones((10, 10), dtype=int)
        result = get_bounding_box(img)
        self.assertEqual(result, [0, 10, 0, 10], 
                         "Bounding box for full image should be [0, 10, 0, 10]")
    
    def test_get_bounding_box_non_contiguous(self):
        """Test for non-contiguous regions."""
        img = np.zeros((10, 10), dtype=int)
        img[1, 1] = 1
        img[8, 8] = 1
        result = get_bounding_box(img)
        self.assertEqual(result, [1, 9, 1, 9], 
                         "Bounding box for non-contiguous regions should be [1, 9, 1, 9]")

class TestRemoveSmallObjects(unittest.TestCase):
    """Tests for the remove_small_objects function."""
    
    def test_remove_small_objects_no_removal(self):
        """Test when no objects are removed."""
        pred = np.zeros((10, 10), dtype=int)
        pred[2:5, 2:5] = 1
        result = remove_small_objects(pred, min_size=4)
        np.testing.assert_array_equal(result, pred, 
                                        "No objects should be removed when all are above min_size")
    
    def test_remove_small_objects_removal(self):
        """Test when small objects are removed."""
        pred = np.zeros((10, 10), dtype=int)
        pred[2:4, 2:4] = 1  # Small object
        pred[5:9, 5:9] = 2  # Large object
        result = remove_small_objects(pred, min_size=5)
        expected = np.zeros((10, 10), dtype=int)
        expected[5:9, 5:9] = 2
        np.testing.assert_array_equal(result, expected, 
                                        "Small objects should be removed")
    
    def test_remove_small_objects_all_removed(self):
        """Test when all objects are removed."""
        pred = np.zeros((10, 10), dtype=int)
        pred[2:4, 2:4] = 1  # Small object
        result = remove_small_objects(pred, min_size=10)
        expected = np.zeros((10, 10), dtype=int)
        np.testing.assert_array_equal(result, expected, 
                                        "All objects should be removed when below min_size")
    
    def test_remove_small_objects_binary_input(self):
        """Test with binary input."""
        pred = np.zeros((10, 10), dtype=bool)
        pred[2:4, 2:4] = True  # Small object
        pred[5:9, 5:9] = True  # Large object
        result = remove_small_objects(pred, min_size=6)
        expected = np.zeros((10, 10), dtype=bool)
        expected[5:9, 5:9] = True
        np.testing.assert_array_equal(result, expected, 
                                        "Small objects should be removed for binary input")
    
    def test_remove_small_objects_no_min_size(self):
        """Test when min_size is 0."""
        pred = np.zeros((10, 10), dtype=int)
        pred[2:4, 2:4] = 1
        result = remove_small_objects(pred, min_size=0)
        np.testing.assert_array_equal(result, pred, 
                                        "No objects should be removed when min_size is 0")

class TestFlattenDict(unittest.TestCase):
    """Tests for the flatten_dict function."""

    def test_flatten_dict_empty(self):
        """Test flattening an empty dictionary."""
        d = {}
        result = flatten_dict(d)
        self.assertEqual(result, {}, "Flattening an empty dictionary should return an empty dictionary")

    def test_flatten_dict_single_level(self):
        """Test flattening a single-level dictionary."""
        d = {"a": 1, "b": 2}
        result = flatten_dict(d)
        self.assertEqual(result, {"a": 1, "b": 2}, "Flattening a single-level dictionary should return the same dictionary")

    def test_flatten_dict_nested(self):
        """Test flattening a nested dictionary."""
        d = {"a": {"b": {"c": 1}}, "d": 2}
        result = flatten_dict(d)
        self.assertEqual(result, {"a.b.c": 1, "d": 2}, "Flattening a nested dictionary should return a flattened dictionary")

    def test_flatten_dict_custom_separator(self):
        """Test flattening a dictionary with a custom separator."""
        d = {"a": {"b": {"c": 1}}, "d": 2}
        result = flatten_dict(d, sep="/")
        self.assertEqual(result, {"a/b/c": 1, "d": 2}, "Flattening with a custom separator should use the specified separator")

    def test_flatten_dict_non_string_keys(self):
        """Test flattening a dictionary with non-string keys."""
        d = {1: {"b": 2}, "c": 3}
        result = flatten_dict(d)
        self.assertEqual(result, {"1.b": 2, "c": 3}, "Flattening should handle non-string keys by converting them to strings")

    def test_flatten_dict_no_parent_key(self):
        """Test flattening with no parent key."""
        d = {"a": {"b": 1}}
        result = flatten_dict(d)
        self.assertEqual(result, {"a.b": 1}, "Flattening should correctly handle cases with no parent key")

class TestUnflattenDict(unittest.TestCase):
    """Tests for the unflatten_dict function."""

    def test_unflatten_dict_empty(self):
        """Test unflattening an empty dictionary."""
        d = {}
        result = unflatten_dict(d)
        self.assertEqual(result, {}, "Unflattening an empty dictionary should return an empty dictionary")

    def test_unflatten_dict_single_level(self):
        """Test unflattening a single-level dictionary."""
        d = {"a": 1, "b": 2}
        result = unflatten_dict(d)
        self.assertEqual(result, {"a": 1, "b": 2}, "Unflattening a single-level dictionary should return the same dictionary")

    def test_unflatten_dict_nested(self):
        """Test unflattening a nested dictionary."""
        d = {"a.b.c": 1, "d": 2}
        result = unflatten_dict(d)
        self.assertEqual(result, {"a": {"b": {"c": 1}}, "d": 2}, "Unflattening a nested dictionary should return a nested dictionary")

    def test_unflatten_dict_custom_separator(self):
        """Test unflattening a dictionary with a custom separator."""
        d = {"a/b/c": 1, "d": 2}
        result = unflatten_dict(d, sep="/")
        self.assertEqual(result, {"a": {"b": {"c": 1}}, "d": 2}, "Unflattening with a custom separator should use the specified separator")

    def test_unflatten_dict_overlapping_keys(self):
        """Test unflattening with overlapping keys."""
        d = {"a.b": 1, "a.b.c": 2}
        result = unflatten_dict(d)
        self.assertEqual(result, {"a": {"b": {"c": 2}}}, "Unflattening should handle overlapping keys correctly")

    def test_unflatten_dict_non_string_keys(self):
        """Test unflattening a dictionary with non-string keys."""
        d = {"1.b": 2, "c": 3}
        result = unflatten_dict(d)
        self.assertEqual(result, {"1": {"b": 2}, "c": 3}, "Unflattening should handle non-string keys correctly")

class TestGetSizeOfDict(unittest.TestCase):
    """Tests for the get_size_of_dict function."""

    def test_get_size_of_dict_empty(self):
        """Test size of an empty dictionary."""
        d = {}
        result = get_size_of_dict(d)
        self.assertEqual(result, sys.getsizeof(d), "Size of an empty dictionary should match sys.getsizeof")

    def test_get_size_of_dict_single_key_value(self):
        """Test size of a dictionary with a single key-value pair."""
        d = {"a": 1}
        expected_size = sys.getsizeof(d) + sys.getsizeof("a") + sys.getsizeof(1)
        result = get_size_of_dict(d)
        self.assertEqual(result, expected_size, "Size of a single key-value dictionary should match expected size")

    def test_get_size_of_dict_multiple_key_values(self):
        """Test size of a dictionary with multiple key-value pairs."""
        d = {"a": 1, "b": 2, "c": 3}
        expected_size = sys.getsizeof(d) + sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in d.items())
        result = get_size_of_dict(d)
        self.assertEqual(result, expected_size, "Size of a multi key-value dictionary should match expected size")

    def test_get_size_of_dict_nested(self):
        """Test size of a nested dictionary."""
        d = {"a": {"b": 2}, "c": 3}
        expected_size = sys.getsizeof(d) + sys.getsizeof("a") + sys.getsizeof({"b": 2}) + sys.getsizeof("c") + sys.getsizeof(3)
        result = get_size_of_dict(d)
        self.assertEqual(result, expected_size, "Size of a nested dictionary should match expected size")

    def test_get_size_of_dict_non_string_keys(self):
        """Test size of a dictionary with non-string keys."""
        d = {1: "a", 2: "b"}
        expected_size = sys.getsizeof(d) + sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in d.items())
        result = get_size_of_dict(d)
        self.assertEqual(result, expected_size, "Size of a dictionary with non-string keys should match expected size")

class TestLoadWsiFilesFromCsv(unittest.TestCase):
    """Tests for the load_wsi_files_from_csv function."""

    def setUp(self):
        """Set up a temporary CSV file for testing."""
        self.csv_path = "test_wsi_files.csv"
        data = {"Filename": ["file1.svs", "file2.tiff", "file3.svs", "file4.png"]}
        self.df = pd.DataFrame(data)
        self.df.to_csv(self.csv_path, index=False)

    def tearDown(self):
        """Remove the temporary CSV file after testing."""
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)

    def test_load_wsi_files_from_csv_valid_extension(self):
        """Test loading files with a valid WSI extension."""
        result = load_wsi_files_from_csv(self.csv_path, "svs")
        expected = ["file1.svs", "file3.svs"]
        self.assertEqual(result, expected, "Should return files with the specified extension")

    def test_load_wsi_files_from_csv_invalid_extension(self):
        """Test loading files with an invalid WSI extension."""
        result = load_wsi_files_from_csv(self.csv_path, "jpg")
        expected = []
        self.assertEqual(result, expected, "Should return an empty list for an invalid extension")

    def test_load_wsi_files_from_csv_mixed_case_extension(self):
        """Test loading files with a mixed-case WSI extension."""
        result = load_wsi_files_from_csv(self.csv_path, "TIFF")
        expected = []
        self.assertEqual(result, expected, "Should handle mixed-case extensions correctly - Case sensitive")

    def test_load_wsi_files_from_csv_empty_csv(self):
        """Test loading files from an empty CSV."""
        empty_csv_path = "empty_test_wsi_files.csv"
        pd.DataFrame({"Filename": []}).to_csv(empty_csv_path, index=False)
        result = load_wsi_files_from_csv(empty_csv_path, "svs")
        expected = []
        self.assertEqual(result, expected, "Should return an empty list for an empty CSV")
        os.remove(empty_csv_path)

    def test_load_wsi_files_from_csv_missing_column(self):
        """Test loading files when the 'Filename' column is missing."""
        invalid_csv_path = "invalid_test_wsi_files.csv"
        pd.DataFrame({"NotFilename": ["file1.svs"]}).to_csv(invalid_csv_path, index=False)
        with self.assertRaises(KeyError, msg="Should raise KeyError if 'Filename' column is missing"):
            load_wsi_files_from_csv(invalid_csv_path, "svs")
        os.remove(invalid_csv_path)

class TestCloseLogger(unittest.TestCase):
    """Tests for the close_logger function."""

    def test_close_logger_no_handlers(self):
        """Test closing a logger with no handlers."""
        logger = logging.getLogger("test_logger_no_handlers")
        logger.handlers = []  # Ensure no handlers are attached
        close_logger(logger)
        self.assertEqual(len(logger.handlers), 0, "Logger should have no handlers after closing")

    def test_close_logger_with_handlers(self):
        """Test closing a logger with handlers."""
        logger = logging.getLogger("test_logger_with_handlers")
        handler = logging.StreamHandler()
        logger.addHandler(handler)
        self.assertEqual(len(logger.handlers), 1, "Logger should have one handler before closing")
        close_logger(logger)
        self.assertEqual(len(logger.handlers), 0, "Logger should have no handlers after closing")

    def test_close_logger_multiple_handlers(self):
        """Test closing a logger with multiple handlers."""
        logger = logging.getLogger("test_logger_multiple_handlers")
        handler1 = logging.StreamHandler()
        handler2 = logging.FileHandler("test.log")
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        self.assertEqual(len(logger.handlers), 2, "Logger should have two handlers before closing")
        close_logger(logger)
        self.assertEqual(len(logger.handlers), 0, "Logger should have no handlers after closing")
        if os.path.exists("test.log"):
            os.remove("test.log")

    def test_close_logger_shutdown(self):
        """Test if logging.shutdown is called."""
        logger = logging.getLogger("test_logger_shutdown")
        with patch("logging.shutdown") as mock_shutdown:  # Verwende das bereits importierte patch
            close_logger(logger)
            mock_shutdown.assert_called_once_with()

class TestRemoveParameterTag(unittest.TestCase):
    """Tests for the remove_parameter_tag function."""

    def test_remove_parameter_tag_empty_dict(self):
        """Test with an empty dictionary."""
        d = {}
        result = remove_parameter_tag(d)
        self.assertEqual(result, {}, "Removing parameter tags from an empty dictionary should return an empty dictionary")

    def test_remove_parameter_tag_no_parameters_key(self):
        """Test with a dictionary that has no 'parameters' key."""
        d = {"a.b.c": 1, "x.y.z": 2}
        result = remove_parameter_tag(d)
        expected = {"a.b": {"c": 1}, "x.y": {"z": 2}}
        self.assertEqual(result, expected, "Dictionary without 'parameters' key should remain unchanged")

    def test_remove_parameter_tag_with_parameters_key(self):
        """Test with a dictionary that contains 'parameters' key."""
        d = {"a.parameters.b": 1, "x.y.parameters.z": 2}
        result = remove_parameter_tag(d)
        expected = {"a": {"b": 1}, "x.y": {"z": 2}}
        self.assertEqual(result, expected, "Keys containing 'parameters' should be removed correctly")

    def test_remove_parameter_tag_mixed_keys(self):
        """Test with a mix of keys with and without 'parameters'."""
        d = {"a.parameters.b": 1, "x.y.z": 2, "p.q.parameters.r": 3}
        result = remove_parameter_tag(d)
        expected = {"a": {"b": 1}, "x.y": {"z": 2}, "p.q": {"r": 3}}
        self.assertEqual(result, expected, "Mixed keys should be processed correctly")

    def test_remove_parameter_tag_custom_separator(self):
        """Test with a custom separator."""
        d = {"a/parameters/b": 1, "x/y/z": 2}
        result = remove_parameter_tag(d, sep="/")
        expected = {"a": {"b": 1}, "x/y": {"z": 2}}
        self.assertEqual(result, expected, "Custom separator should be handled correctly")

    def test_remove_parameter_tag_nested_keys(self):
        """Test with deeply nested keys."""
        d = {"a.parameters.b.c": 1, "x.y.parameters.z.w": 2}
        result = remove_parameter_tag(d)
        expected = {"a.b": {"c": 1}, "x.y.z": {"w": 2}}
        self.assertEqual(result, expected, "Deeply nested keys with 'parameters' should be processed correctly")

    def test_remove_parameter_tag_no_removal_needed(self):
        """Test when no keys contain 'parameters'."""
        d = {"a.b.c": 1, "x.y.z": 2}
        result = remove_parameter_tag(d)
        expected = {"a.b": {"c": 1}, "x.y": {"z": 2}}
        self.assertEqual(result, expected, "Dictionary without 'parameters' should remain unchanged")


class TestRemapLabel(unittest.TestCase):
    """Tests for the remap_label function."""

    def test_remap_label_no_instances(self):
        """Test when there are no instances in the input."""
        pred = np.zeros((5, 5), dtype=int)
        result = remap_label(pred)
        np.testing.assert_array_equal(result, pred, "Output should match input when there are no instances")

    def test_remap_label_single_instance(self):
        """Test when there is a single instance."""
        pred = np.zeros((5, 5), dtype=int)
        pred[1:4, 1:4] = 1
        result = remap_label(pred)
        expected = np.zeros((5, 5), dtype=int)
        expected[1:4, 1:4] = 1
        np.testing.assert_array_equal(result, expected, "Single instance should remain unchanged")

    def test_remap_label_multiple_instances(self):
        """Test when there are multiple instances."""
        pred = np.zeros((5, 5), dtype=int)
        pred[1:3, 1:3] = 2
        pred[3:5, 3:5] = 4
        result = remap_label(pred)
        expected = np.zeros((5, 5), dtype=int)
        expected[1:3, 1:3] = 1
        expected[3:5, 3:5] = 2
        np.testing.assert_array_equal(result, expected, "Instances should be renumbered contiguously")

    def test_remap_label_by_size(self):
        """Test when instances are reordered by size."""
        pred = np.zeros((5, 6), dtype=int)
        pred[1:3, 1:3] = 2  # Smaller instance
        pred[3:5, 3:6] = 4  # Larger instance
        result = remap_label(pred, by_size=True)
        expected = np.zeros((5, 6), dtype=int)
        expected[1:3, 1:3] = 2
        expected[3:5, 3:6] = 1
        np.testing.assert_array_equal(result, expected, "Instances should be renumbered by size")

    def test_remap_label_with_zero_background(self):
        """Test when the input contains a zero background."""
        pred = np.zeros((5, 5), dtype=int)
        pred[1:3, 1:3] = 5
        pred[3:5, 3:5] = 10
        result = remap_label(pred)
        expected = np.zeros((5, 5), dtype=int)
        expected[1:3, 1:3] = 1
        expected[3:5, 3:5] = 2
        np.testing.assert_array_equal(result, expected, "Zero background should remain unchanged")

    def test_remap_label_all_removed(self):
        """Test when all instances are removed."""
        pred = np.zeros((5, 5), dtype=int)
        result = remap_label(pred)
        expected = np.zeros((5, 5), dtype=int)
        np.testing.assert_array_equal(result, expected, "Output should be all zeros when input has no instances")




if __name__ == "__main__":
    unittest.main()