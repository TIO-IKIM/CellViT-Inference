# -*- coding: utf-8 -*-
# Test YAML Configuration for CellViT Inference
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import unittest

from cellvit.inference.cli import InferenceConfiguration
from pathlib import Path
import pandas as pd
from unittest.mock import patch
import shutil


class TestInferenceConfiguration(unittest.TestCase):
    def setUp(self):
        # Sample valid configuration
        self.valid_config = {
            "model": "SAM",
            "nuclei_taxonomy": "pannuke",
            "inference": {
                "gpu": 0,
                "enforce_amp": True,
                "batch_size": 8,
            },
            "output_format": {
                "outdir": "output",
                "geojson": True,
                "graph": False,
                "compression": True,
            },
            "process_wsi": {
                "wsi_path": "tests/test_data/sample_wsi.svs",
                "wsi_mpp": 0.25,
                "wsi_magnification": 20,
            },
            "system": {
                "cpu_count": 4,
                "ray_worker": 2,
                "ray_remote_cpus": 2,
                "memory": 8192,
            },
            "debug": False,
        }

        # Create a temporary directory and file for testing
        self.temp_dir = Path("tests/test_data")
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.temp_wsi_file = self.temp_dir / "sample_wsi.svs"
        self.temp_wsi_file.touch()

    def tearDown(self):
        # Clean up temporary files and directories
        if self.temp_wsi_file.exists():
            self.temp_wsi_file.unlink()
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @patch("torch.cuda.device_count")
    def test_valid_configuration(self, mock_device_count):
        """Test initialization with a valid configuration."""
        # mock the CUDA device count to simulate GPU availability
        mock_device_count.return_value = 1

        config = InferenceConfiguration(self.valid_config)
        self.assertEqual(config.model, "SAM", "Model should be SAM")
        self.assertEqual(
            config.nuclei_taxonomy, "pannuke", "Nuclei taxonomy should be pannuke"
        )
        self.assertEqual(config.gpu, 0, "GPU should be 0")
        self.assertTrue(config.enforce_amp, "Enforce AMP should be True")
        self.assertEqual(config.batch_size, 8, "Batch size should be 8")
        self.assertEqual(
            config.outdir, Path("output"), "Output directory should be 'output'"
        )
        self.assertTrue(config.geojson, "GeoJSON should be True")
        self.assertFalse(config.graph, "Graph should be False")
        self.assertTrue(config.compression, "Compression should be True")
        self.assertEqual(
            config.wsi_path,
            Path("tests/test_data/sample_wsi.svs"),
            "WSI path should be 'tests/test_data/sample_wsi.svs'",
        )
        self.assertEqual(config.wsi_mpp, 0.25, "WSI MPP should be 0.25")
        self.assertEqual(config.wsi_magnification, 20, "WSI magnification should be 20")
        self.assertEqual(config.cpu_count, 4, "CPU count should be 4")
        self.assertEqual(config.ray_worker, 2, "Ray worker should be 2")
        self.assertEqual(config.ray_remote_cpus, 2, "Ray remote CPUs should be 2")
        self.assertEqual(config.memory, 8192, "Memory should be 8192MB")
        self.assertFalse(config.debug, "Debug should be False")

    def test_missing_model(self):
        """Test configuration with missing model."""
        invalid_config = self.valid_config.copy()
        del invalid_config["model"]
        with self.assertRaises(AssertionError) as context:
            InferenceConfiguration(invalid_config)
        self.assertEqual(
            str(context.exception),
            "Model must be provided in config",
            "Model must be provided in config",
        )

    def test_invalid_gpu(self):
        """Test configuration with invalid GPU ID."""
        invalid_config = self.valid_config.copy()
        invalid_config["inference"]["gpu"] = -1
        with self.assertRaises(AssertionError) as context:
            InferenceConfiguration(invalid_config)
        self.assertIn(
            "GPU must be between 0 and",
            str(context.exception),
            "GPU must be between 0 and -1",
        )

    @patch("torch.cuda.device_count")
    def test_invalid_batch_size(self, mock_device_count):
        """Test configuration with invalid batch size."""
        mock_device_count.return_value = 1
        invalid_config = self.valid_config.copy()
        invalid_config["inference"]["batch_size"] = 50
        with self.assertRaises(AssertionError) as context:
            InferenceConfiguration(invalid_config)
        self.assertEqual(
            str(context.exception),
            "Batch size must be between 2 and 32",
            "Batch size must be between 2 and 32",
        )

    @patch("torch.cuda.device_count")
    def test_invalid_cpu_count(self, mock_device_count):
        """Test configuration with missing output directory."""
        mock_device_count.return_value = 1
        invalid_config = self.valid_config.copy()
        del invalid_config["output_format"]["outdir"]
        with self.assertRaises(AssertionError) as context:
            InferenceConfiguration(invalid_config)
        self.assertEqual(
            str(context.exception),
            "Output directory must be provided",
            "Output directory must be provided",
        )

    @patch("torch.cuda.device_count")
    def test_invalid_wsi_path(self, mock_device_count):
        """Test configuration with invalid WSI path."""
        mock_device_count.return_value = 1

        invalid_config = self.valid_config.copy()
        invalid_config["process_wsi"]["wsi_path"] = "invalid_path.svs"
        with self.assertRaises(AssertionError) as context:
            InferenceConfiguration(invalid_config)
        self.assertEqual(
            str(context.exception), "WSI path does not exist", "WSI path does not exist"
        )

    def test_invalid_nuclei_taxonomy(self):
        """Test configuration with invalid nuclei taxonomy."""
        invalid_config = self.valid_config.copy()
        invalid_config["nuclei_taxonomy"] = "invalid_taxonomy"
        with self.assertRaises(AssertionError) as context:
            InferenceConfiguration(invalid_config)
        self.assertIn(
            "Nuclei taxonomy must be one of",
            str(context.exception),
            "Nuclei taxonomy must be one of the valid options",
        )

    @patch("torch.cuda.device_count")
    def test_invalid_memory(self, mock_device_count):
        """Test configuration with insufficient memory."""
        mock_device_count.return_value = 1

        invalid_config = self.valid_config.copy()
        invalid_config["system"]["memory"] = 4096
        with self.assertRaises(AssertionError) as context:
            InferenceConfiguration(invalid_config)
        self.assertEqual(
            str(context.exception),
            "Memory must be larger than 8GB (8192MB)",
            "Memory must be larger than 8GB (8192MB)",
        )

    @patch("torch.cuda.device_count")
    def test_invalid_command(self, mock_device_count):
        """Test configuration with missing command."""
        mock_device_count.return_value = 1

        invalid_config = self.valid_config.copy()
        del invalid_config["process_wsi"]
        with self.assertRaises(AssertionError) as context:
            InferenceConfiguration(invalid_config)
        self.assertEqual(
            str(context.exception),
            "Command must be provided",
            "Command must be provided",
        )

    @patch("torch.cuda.device_count")
    def test_invalid_cpu_count(self, mock_device_count):
        """Test GPU setting with mocked CUDA device count."""
        mock_device_count.return_value = 4  # Simulate 4 GPUs available
        # Valid GPU ID
        valid_config = self.valid_config.copy()
        valid_config["inference"]["gpu"] = 2
        config = InferenceConfiguration(valid_config)
        self.assertEqual(config.gpu, 2, "GPU should be 2")

        # Edge case: GPU ID 0
        valid_config["inference"]["gpu"] = 0
        config = InferenceConfiguration(valid_config)
        self.assertEqual(config.gpu, 0, "GPU should be 0")

        # Edge case: Maximum valid GPU ID
        valid_config["inference"]["gpu"] = 3
        config = InferenceConfiguration(valid_config)
        self.assertEqual(config.gpu, 3, "GPU should be 3")

        # Invalid GPU ID: too high
        invalid_config = self.valid_config.copy()
        invalid_config["inference"]["gpu"] = 4  # Out of range (0-3)
        with self.assertRaises(AssertionError) as context:
            InferenceConfiguration(invalid_config)
        self.assertIn(
            "GPU must be between 0 and 3",
            str(context.exception),
            "GPU must be between 0 and 3",
        )

        # Invalid GPU ID: negative
        invalid_config["inference"]["gpu"] = -1
        with self.assertRaises(AssertionError) as context:
            InferenceConfiguration(invalid_config)
        self.assertIn(
            "GPU must be between 0 and 3",
            str(context.exception),
            "GPU must be between 0 and 3",
        )

    @patch("torch.cuda.device_count")
    def test_gpu_with_no_cuda(self, mock_device_count):
        """Test GPU setting when CUDA is not available."""
        mock_device_count.return_value = 0  # Simulate no GPUs available
        # Any GPU ID should fail since no GPUs are available
        invalid_config = self.valid_config.copy()
        invalid_config["inference"]["gpu"] = 0
        with self.assertRaises(AssertionError) as context:
            InferenceConfiguration(invalid_config)
        self.assertEqual(
            str(context.exception),
            "GPU must be between 0 and -1",
            "GPU must be between 0 and -1",
        )

    def test_gpu_missing_config(self):
        """Test behavior when GPU config is missing."""
        config_no_inference = self.valid_config.copy()
        del config_no_inference["inference"]
        config = InferenceConfiguration(config_no_inference)
        # Should use default value
        self.assertEqual(config.gpu, 0, "GPU should be 0")

        config_no_gpu = self.valid_config.copy()
        del config_no_gpu["inference"]["gpu"]
        config = InferenceConfiguration(config_no_gpu)
        # Should use default value
        self.assertEqual(config.gpu, 0, "GPU should be 0")

    def test_invalid_model(self):
        """Test configuration with invalid model."""
        invalid_config = self.valid_config.copy()
        invalid_config["model"] = "INVALID_MODEL"
        with self.assertRaises(AssertionError) as context:
            InferenceConfiguration(invalid_config)
        self.assertIn(
            "Model must be either 'SAM' or 'HIPT'",
            str(context.exception),
            "Model must be either 'SAM' or 'HIPT'",
        )

    @patch("torch.cuda.device_count")
    def test_default_nuclei_taxonomy(self, mock_device_count):
        """Test default nuclei taxonomy when not provided."""
        mock_device_count.return_value = 1
        config_without_taxonomy = self.valid_config.copy()
        del config_without_taxonomy["nuclei_taxonomy"]
        config = InferenceConfiguration(config_without_taxonomy)
        self.assertEqual(
            config.nuclei_taxonomy, "pannuke", "Must be pannuke"
        )  # Default value

    @patch("torch.cuda.device_count")
    def test_nuclei_taxonomy_writing(self, mock_device_count):
        """Test default nuclei taxonomy when not provided."""
        mock_device_count.return_value = 1
        config_taxonomy = self.valid_config.copy()
        config_taxonomy["nuclei_taxonomy"] = "PaNOpTilS"
        config = InferenceConfiguration(config_taxonomy)
        self.assertEqual(
            config.nuclei_taxonomy, "panoptils", "Must be panoptils"
        )  # Default value

    @patch("torch.cuda.device_count")
    def test_enforce_amp_settings(self, mock_device_count):
        """Test different enforce_amp settings."""
        mock_device_count.return_value = 1

        # Test with True
        config_true = self.valid_config.copy()
        config_true["inference"]["enforce_amp"] = True
        config = InferenceConfiguration(config_true)
        self.assertTrue(config.enforce_amp)

        # Test with False
        config_false = self.valid_config.copy()
        config_false["inference"]["enforce_amp"] = False
        config = InferenceConfiguration(config_false)
        self.assertFalse(config.enforce_amp)

    @patch("torch.cuda.device_count")
    def test_default_enforce_amp(self, mock_device_count):
        """Test default enforce_amp value when not provided."""
        mock_device_count.return_value = 1
        config_without_amp = self.valid_config.copy()
        del config_without_amp["inference"]["enforce_amp"]
        config = InferenceConfiguration(config_without_amp)
        self.assertFalse(config.enforce_amp)  # Default value should be False

    @patch("torch.cuda.device_count")
    def test_default_batch_size(self, mock_device_count):
        """Test default batch size when not provided."""
        mock_device_count.return_value = 1
        config_without_batch_size = self.valid_config.copy()
        del config_without_batch_size["inference"]["batch_size"]
        config = InferenceConfiguration(config_without_batch_size)
        self.assertEqual(config.batch_size, 8)  # Default value

    @patch("torch.cuda.device_count")
    def test_output_format_options(self, mock_device_count):
        """Test different output format options."""
        mock_device_count.return_value = 1

        # Test geojson settings
        config_geojson_true = self.valid_config.copy()
        config_geojson_true["output_format"]["geojson"] = True
        config = InferenceConfiguration(config_geojson_true)
        self.assertTrue(config.geojson)

        config_geojson_false = self.valid_config.copy()
        config_geojson_false["output_format"]["geojson"] = False
        config = InferenceConfiguration(config_geojson_false)
        self.assertFalse(config.geojson)

        # Test graph settings
        config_graph_true = self.valid_config.copy()
        config_graph_true["output_format"]["graph"] = True
        config = InferenceConfiguration(config_graph_true)
        self.assertTrue(config.graph)

        # Test compression settings
        config_compression_false = self.valid_config.copy()
        config_compression_false["output_format"]["compression"] = False
        config = InferenceConfiguration(config_compression_false)
        self.assertFalse(config.compression)

    @patch("torch.cuda.device_count")
    def test_default_output_format_options(self, mock_device_count):
        """Test default output format options when not provided."""
        mock_device_count.return_value = 1

        config_without_options = self.valid_config.copy()
        config_without_options["output_format"] = {
            "outdir": "output"
        }  # Nur required outdir

        config = InferenceConfiguration(config_without_options)
        self.assertFalse(config.geojson)  # Default sollte False sein
        self.assertFalse(config.graph)  # Default sollte False sein
        self.assertFalse(config.compression)  # Default sollte False sein

    @patch("torch.cuda.device_count")
    def test_wsi_without_mpp_magnification(self, mock_device_count):
        """Test WSI processing without MPP and magnification."""
        mock_device_count.return_value = 1

        config_without_mpp_mag = self.valid_config.copy()
        config_without_mpp_mag["process_wsi"] = {
            "wsi_path": "tests/test_data/sample_wsi.svs"
        }

        config = InferenceConfiguration(config_without_mpp_mag)
        self.assertIsNone(config.wsi_mpp)  # Sollte None sein, wenn nicht angegeben
        self.assertIsNone(
            config.wsi_magnification
        )  # Sollte None sein, wenn nicht angegeben

    @patch("torch.cuda.device_count")
    def test_process_dataset_folder(self, mock_device_count):
        """Test dataset processing with folder path."""
        mock_device_count.return_value = 1

        # Temporäres Verzeichnis für den Test erstellen
        dataset_dir = Path("tests/test_data/dataset")
        dataset_dir.mkdir(exist_ok=True, parents=True)

        try:
            config_with_dataset = self.valid_config.copy()
            del config_with_dataset["process_wsi"]
            config_with_dataset["process_dataset"] = {
                "wsi_folder": str(dataset_dir),
                "wsi_extension": "svs",
            }

            config = InferenceConfiguration(config_with_dataset)
            self.assertEqual(config.wsi_folder, dataset_dir)
            self.assertEqual(config.wsi_extension, "svs")
            self.assertIsNone(config.wsi_path)
        finally:
            # Aufräumen
            if dataset_dir.exists():
                dataset_dir.rmdir()

    @patch("torch.cuda.device_count")
    def test_process_dataset_filelist(self, mock_device_count):
        """Test dataset processing with file list."""
        mock_device_count.return_value = 1

        # Temporäre CSV-Datei für den Test erstellen
        csv_path = Path("tests/test_data/filelist.csv")
        df = pd.DataFrame({"path": ["file1.svs", "file2.svs"]})
        df.to_csv(csv_path, index=False)

        try:
            config_with_filelist = self.valid_config.copy()
            del config_with_filelist["process_wsi"]
            config_with_filelist["process_dataset"] = {"wsi_filelist": str(csv_path)}

            config = InferenceConfiguration(config_with_filelist)
            self.assertIsNone(config.wsi_path)  # Sollte kein wsi_path haben
        finally:
            # Aufräumen
            if csv_path.exists():
                csv_path.unlink()

    @patch("torch.cuda.device_count")
    def test_invalid_dataset_config(self, mock_device_count):
        """Test dataset processing with both folder and filelist (should fail)."""
        mock_device_count.return_value = 1

        invalid_dataset_config = self.valid_config.copy()
        del invalid_dataset_config["process_wsi"]
        invalid_dataset_config["process_dataset"] = {
            "wsi_folder": "folder_path",
            "wsi_filelist": "filelist.csv",
        }

        with self.assertRaises(AssertionError) as context:
            InferenceConfiguration(invalid_dataset_config)
        self.assertIn(
            "Either 'wsi_folder' or 'wsi_filelist' must be provided, but not both.",
            str(context.exception),
        )

    @patch("torch.cuda.device_count")
    def test_debug_settings(self, mock_device_count):
        """Test debug setting configuration."""
        mock_device_count.return_value = 1

        # Test mit debug=True
        config_debug_true = self.valid_config.copy()
        config_debug_true["debug"] = True
        config = InferenceConfiguration(config_debug_true)
        self.assertTrue(config.debug)

        # Test mit debug=False
        config_debug_false = self.valid_config.copy()
        config_debug_false["debug"] = False
        config = InferenceConfiguration(config_debug_false)
        self.assertFalse(config.debug)

    @patch("torch.cuda.device_count")
    def test_default_debug_setting(self, mock_device_count):
        """Test default debug setting when not provided."""
        mock_device_count.return_value = 1

        config_without_debug = self.valid_config.copy()
        del config_without_debug["debug"]

        config = InferenceConfiguration(config_without_debug)
        self.assertFalse(config.debug)  # Default sollte False sein
