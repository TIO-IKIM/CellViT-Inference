# -*- coding: utf-8 -*-
# Test entire inference pipeline for CellViT
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import unittest
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from cellvit.inference.inference import CellViTInference
from cellvit.utils.ressource_manager import SystemConfiguration
import os


class TestCellViTInference(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for outputs
        self.temp_dir = tempfile.mkdtemp()
        self.outdir = Path(self.temp_dir)

        # Create a mock SystemConfiguration
        self.system_config = MagicMock(spec=SystemConfiguration)
        self.system_config.get_current_memory_percentage.return_value = 50.1
        self.system_config.__getitem__.side_effect = lambda key: {
            "gpu_index": 0,
            "cpu_count": 4,
            "ray_worker": 1,
            "ray_remote_cpus": 2,
            "memory": 8192,
            "gpu_memory": 24,
            "cupy": False,
        }[key]
        self.system_config.memory = 8192

        self.default_params = {
            "model_name": "HIPT",
            "outdir": self.outdir,
            "system_configuration": self.system_config,
        }

        self.patches = []

        original_init = CellViTInference.__init__

        def test_init(self_obj, *args, **kwargs):
            original_init(self_obj, *args, **kwargs)
            self_obj.device = "cpu"

        init_patch = patch.object(CellViTInference, "__init__", test_init)
        init_patch.start()
        self.patches.append(init_patch)

        self.original_methods = {}
        for method_name in [
            "_instantiate_logger",
            "_load_model",
            "_check_devices",
            "_load_classifier",
            "_load_inference_transforms",
            "_setup_amp",
            "_setup_worker",
        ]:
            self.original_methods[method_name] = getattr(CellViTInference, method_name)

            method_patch = patch.object(
                CellViTInference, method_name, return_value=None
            )
            method_patch.start()
            self.patches.append(method_patch)

    def tearDown(self):
        # Stop all patches
        for p in self.patches:
            p.stop()

        # Remove temporary directory
        shutil.rmtree(self.temp_dir)

    def restore_method(self, method_name):
        for i, p in enumerate(self.patches):
            if getattr(p, "target", None) == getattr(
                CellViTInference, method_name, None
            ):
                p.stop()
                self.patches.pop(i)
                setattr(
                    CellViTInference, method_name, self.original_methods[method_name]
                )
                return True
        return False

    def mock_method_again(self, method_name):
        current_method = getattr(CellViTInference, method_name)
        if current_method == self.original_methods[method_name]:
            method_patch = patch.object(
                CellViTInference, method_name, return_value=None
            )
            method_patch.start()
            self.patches.append(method_patch)
            return True
        return False

    def test_init_with_default_params(self):
        """Test initialization with default parameters"""
        inference = CellViTInference(**self.default_params)

        # Check that attributes are set correctly
        self.assertEqual(inference.model_name, "HIPT")
        self.assertEqual(inference.outdir, self.outdir)
        self.assertEqual(inference.system_configuration, self.system_config)
        self.assertEqual(inference.nuclei_taxonomy, "pannuke")
        self.assertEqual(inference.batch_size, 8)
        self.assertEqual(inference.patch_size, 1024)
        self.assertEqual(inference.overlap, 64)
        self.assertFalse(inference.geojson)
        self.assertFalse(inference.graph)
        self.assertFalse(inference.compression)
        self.assertFalse(inference.debug)
        self.assertEqual(inference.device, "cpu")

        # Check that setup methods were called
        for method in [
            "_instantiate_logger",
            "_load_model",
            "_check_devices",
            "_load_classifier",
            "_load_inference_transforms",
            "_setup_amp",
            "_setup_worker",
        ]:
            getattr(CellViTInference, method).assert_called_once()

    def test_load_model_real_implementation(self):
        """Test loading the model with real implementation"""
        inference = CellViTInference(**self.default_params)

        for method in [
            "_instantiate_logger",
            "_load_model",
            "_check_devices",
            "_load_classifier",
            "_load_inference_transforms",
            "_setup_amp",
            "_setup_worker",
        ]:
            original_method = self.original_methods[method]
            bound_method = original_method.__get__(inference, CellViTInference)
            # Dynamisch das richtige Attribut setzen
            setattr(inference, method, bound_method)
        os.environ["RAY_ENABLE_MAC_LARGE_OBJECT_STORE"] = "1"

        inference._instantiate_logger()
        inference._load_model()
        inference._load_classifier()
        inference._load_inference_transforms()
        inference._setup_amp()
        inference._setup_worker()

        inference.process_wsi(
            wsi_path="./tests/data/test_wsi_database/BRACS_1640_N_3_cropped.tiff",
            wsi_mpp=0.25,
            wsi_magnification=40,
        )

        check_wsi_path = self.outdir / "BRACS_1640_N_3_cropped"
        check_cell_detection_path = check_wsi_path / "cell_detection.json"

        # load detections
        with open(check_cell_detection_path, "r") as f:
            import json

            detections = json.load(f)
        # calculate the number of detections per class
        class_counts = {}
        for detection in detections["cells"]:
            class_name = detection["type"]
            if class_name not in class_counts:
                class_counts[class_name] = 0
            class_counts[class_name] += 1

        self.assertEqual(len(detections["cells"]), 185)
        self.assertEqual(class_counts[5], 97)
        self.assertEqual(class_counts[3], 81)
        self.assertEqual(class_counts[2], 7)
