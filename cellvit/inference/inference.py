# -*- coding: utf-8 -*-
# CellViT Inference Method for Patch-Wise Inference on a patches test set/Whole WSI
#
# Detect Cells with our Networks
# Patches dataset needs to have the follwoing requirements:
# Patch-Size must be 1024, with overlap of 64
#
# We provide preprocessing code here: ./preprocessing/patch_extraction/main_extraction.py
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen


import sys

import itertools
import logging
import uuid
from pathlib import Path
from typing import Callable, List, Literal, Tuple, Union
from importlib.resources import files

import numpy as np
import pandas as pd
import ray
import snappy
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import ujson
from pathopatch.patch_extraction.dataset import (
    LivePatchWSIConfig,
    LivePatchWSIDataloader,
    LivePatchWSIDataset,
)
from torchvision import transforms as T

from cellvit.config.config import COLOR_DICT_CELLS, TYPE_NUCLEI_DICT_PANNUKE
from cellvit.config.templates import get_template_point, get_template_segmentation
from cellvit.data.dataclass.cell_graph import CellGraphDataWSI
from cellvit.data.dataclass.wsi import WSIMetadata
from cellvit.data.dataclass.wsi_meta import load_wsi_meta
from cellvit.inference.overlap_cell_cleaner import OverlapCellCleaner
from cellvit.models.cell_segmentation.cellvit import CellViT
from cellvit.models.cell_segmentation.cellvit_256 import CellViT256
from cellvit.models.cell_segmentation.cellvit_sam import CellViTSAM
from cellvit.models.classifier.linear_classifier import LinearClassifier
from cellvit.utils.cache_models import (
    cache_cellvit_256,
    cache_cellvit_sam_h,
    cache_classifier,
)
from cellvit.utils.logger import Logger
from cellvit.utils.ressource_manager import SystemConfiguration, retrieve_actor_usage
from cellvit.utils.tools import unflatten_dict

PYTHON_PATH = sys.executable
CHECK_RAY_PATH = str(files("cellvit.utils").joinpath("check_ray.py"))


class CellViTInference:
    def __init__(
        self,
        model_name: Literal["SAM", "HIPT"],
        outdir: Union[Path, str],
        system_configuration: SystemConfiguration,
        nuclei_taxonomy: Literal[
            "pannuke",
            "binary",
            "consep",
            "lizard",
            "midog",
            "nucls_main",
            "nucls_super",
            "ocelot",
            "panoptils",
        ] = "pannuke",
        batch_size: int = 8,
        patch_size: int = 1024,
        overlap: int = 64,
        geojson: bool = False,
        graph: bool = False,
        compression: bool = False,
        enforce_amp: bool = False,
        debug: bool = False,
    ) -> None:
        """CellViT Inference Class

        Setup the CellViT Inference Pipeline

        Args:
            model_name (Literal["SAM", "HIPT"]): Name of the model to use. Must be one of: SAM, HIPT
            outdir (Union[Path, str]): Output directory
            system_configuration (SystemConfiguration): System configuration to define the hardware setup, libraries to use and other settings
            nuclei_taxonomy (Literal["pannuke", "binary", "consep", "lizard", "midog", "nucls_main", "nucls_super", "ocelot", "panoptils"], optional): Classification scheme. Defaults to "pannuke".
            batch_size (int, optional): Batch-size for inference. Defaults to 8.
            patch_size (int, optional): Patch-size for inference. Defaults to 1024.
            overlap (int, optional): Overlap between patches. Defaults to 64.
            geojson (bool, optional): If a geojson export should be performed. Defaults to False.
            graph (bool, optional): If a graph export should be performed. Defaults to False.
            compression (bool, optional): If a snappy compression should be performed. Defaults to False.
            enforce_amp (bool, optional): Using PyTorch autocasting with dtype float16 to speed up inference. Also good for trained amp networks.
                Can be used to enforce amp inference even for networks trained without amp. Otherwise, the network setting is used. Defaults to False.
            debug (bool, optional): If debug level. Defaults to False.

        Attributes:
            model_name (Literal["SAM", "HIPT"]): Name of the model to use. Must be one of: SAM, HIPT
            outdir (Path): Output directory
            system_configuration (SystemConfiguration): System configuration to define the hardware setup, libraries to use and other settings
            nuclei_taxonomy (Literal["pannuke", "binary", "consep", "lizard", "midog", "nucls_main", "nucls_super", "ocelot", "panoptils"]): Classification scheme
            batch_size (int): Batch-size for inference
            patch_size (int): Patch-size for inference
            overlap (int): Overlap between patches
            geojson (bool): If a geojson export should be performed
            graph (bool): If a graph export should be performed
            compression (bool): If a snappy compression should be performed
            debug (bool): If debug level
            logger (Logger): Logger
            model (CellViT): Model
            model_arch (str): Model architecture
            run_conf (dict): Run configuration
            inference_transforms (Callable): Inference transformations
            mixed_precision (bool): Using PyTorch autocasting with dtype float16 to speed up inference. Also good for trained amp networks.
            num_workers_torch (int): Number of workers for PyTorch
            label_map (dict): Label map
            classifier (nn.Module): Classifier
            binary (bool): If binary detection
            device (torch.device): Device

        Methods:
            _instantiate_logger() -> None:
                Instantiate logger
            _load_model() -> None:
                Load model and checkpoint and load the state_dict
            _get_model(model_type: Literal["CellViT256", "CellViTSAM"]) -> CellViT:
                Return the trained model for inference (CellViT-Backbone)
            _check_devices() -> None:
                Check batch size based on GPU memory
            _load_classifier() -> None:
                Load the classifier
            _load_inference_transforms() -> None:
                Load the inference transformations from the run_configuration
            _setup_amp(enforce_amp: bool=False) -> None:
                Setup automated mixed precision (amp) for inference
            _setup_worker() -> None:
                Setup the worker for inference
            _import_postprocessing() -> None:
                Import the postprocessing module
            process_wsi(wsi_path: Union[Path, str], wsi_mpp: float = None, wsi_magnification: float = None, apply_prefilter: bool = True, filter_patches: bool = False, **kwargs) -> None:
                Process a whole slide image with CellViT
            apply_softmax_reorder(predictions: dict) -> dict:
                Reorder and apply softmax on predictions
            _post_process_edge_cells(cell_list: List[dict]) -> List[int]:
                Use the CellPostProcessor to remove multiple cells and merge due to overlap
            def _reallign_grid(cell_dict_wsi: List[dict], cell_dict_detection: List[dict], graph_data: dict, rescaling_factor: float) -> Tuple[List[dict], List[dict], dict]:
                Reallign grid if interpolation was used (including target_mpp_tolerance)
            def _convert_json_geojson(cell_dict: List[dict], complete: bool) -> List[dict]:
                Convert JSON to GeoJSON
            def _remove_padding(cell_dict_wsi: List[dict], cell_dict_detection: List[dict], graph_data: dict, wsi_dimension: Tuple[int, int]) -> Tuple[List[dict], List[dict], dict]:
                Remove padding from the WSI
        """
        # hand over parameters
        self.model_name: str = model_name.upper()
        self.outdir: Path = Path(outdir)
        self.system_configuration: SystemConfiguration = system_configuration
        self.nuclei_taxonomy: str = nuclei_taxonomy.lower()
        self.batch_size: int = batch_size
        self.patch_size: int = patch_size
        self.overlap: int = overlap
        self.geojson: bool = geojson
        self.graph: bool = graph
        self.compression: bool = compression
        self.debug: bool = debug

        # derived parameters
        self.logger: Logger
        self.model: CellViT
        self.model_arch: str
        self.run_conf: dict
        self.inference_transforms: Callable
        self.mixed_precision: bool
        self.num_workers_torch: int
        self.label_map: dict = TYPE_NUCLEI_DICT_PANNUKE
        self.classifier: nn.Module = None
        self.binary: bool = False
        self.device: torch.device = f"cuda:{self.system_configuration['gpu_index']}"

        # setup
        self._instantiate_logger()
        self._load_model()
        self._check_devices()
        self._load_classifier()
        self._load_inference_transforms()
        self._setup_amp(enforce_amp=enforce_amp)
        self._setup_worker()

    def _instantiate_logger(self) -> None:
        """Instantiate logger

        Logger is using no formatters. Logs are stored in the run directory under the filename: inference.log
        """
        if self.debug:
            level = "DEBUG"

        else:
            level = "INFO"
        logger = Logger(
            level=level,
        )
        self.logger = logger.create_logger()
        if self.debug:
            if not ray._private.utils.check_dashboard_dependencies_installed():
                self.logger.error(
                    "Cannot start the Ray dashboard for debugging because required dependencies are missing.\n"
                    "To resolve this, please install the missing dependencies by running:\n"
                    "    pip install -U 'ray[default]'\n\n"
                    "This debug session will not run with ray-dashboard for debugging."
                )

    def _load_model(self) -> None:
        """Load model and checkpoint and load the state_dict"""
        self.logger.info(f"Loading model: {self.model_name}")
        if self.model_name == "SAM":
            model_path = cache_cellvit_sam_h(logger=self.logger)
        elif self.model_name == "HIPT":
            model_path = cache_cellvit_256(logger=self.logger)
        else:
            raise ValueError("Unknown model name. Please select one of ['SAM', 'HIPT']")

        model_checkpoint = torch.load(model_path, map_location="cpu")

        # unpack checkpoint
        self.run_conf = unflatten_dict(model_checkpoint["config"], ".")
        self.model = self._get_model(model_type=model_checkpoint["arch"])
        self.logger.info(
            self.model.load_state_dict(model_checkpoint["model_state_dict"])
        )
        self.model.eval()
        self.model.to(self.device)
        self.run_conf["model"]["token_patch_size"] = self.model.patch_size
        self.model_arch = model_checkpoint["arch"]

    def _get_model(
        self, model_type: Literal["CellViT256", "CellViTSAM"]
    ) -> Union[CellViT256, CellViTSAM]:
        """Return the trained model for inference

        Args:
            model_type (str): Name of the model. Must either be one of:
                CellViT256, CellViTSAM

        Returns:
            Union[CellViT256, CellViTSAM]: Model
        """
        implemented_models = ["CellViT256", "CellViTSAM"]
        if model_type not in implemented_models:
            raise NotImplementedError(
                f"Unknown model type. Please select one of {implemented_models}"
            )
        elif model_type in ["CellViT256"]:
            model = CellViT256(
                model256_path=None,
                num_nuclei_classes=self.run_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=self.run_conf["data"]["num_tissue_classes"],
                regression_loss=self.run_conf["model"].get("regression_loss", False),
            )
        elif model_type in ["CellViTSAM"]:
            model = CellViTSAM(
                model_path=None,
                num_nuclei_classes=self.run_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=self.run_conf["data"]["num_tissue_classes"],
                vit_structure=self.run_conf["model"]["backbone"],
                regression_loss=self.run_conf["model"].get("regression_loss", False),
            )

        return model

    def _check_devices(self) -> None:
        """Check batch size based on GPU memory

        Args:
            gpu (int): GPU-ID
        """
        max_batch_size = 128
        if self.system_configuration["gpu_memory"] < 22:
            if self.model_arch == "CellViTSAM":
                max_batch_size = 2
            elif self.model_arch == "CellViTUNI":
                max_batch_size = 8
            elif self.model_arch == "CellViT256":
                max_batch_size = 8
        elif self.system_configuration["gpu_memory"] < 38:
            if self.model_arch == "CellViTSAM":
                max_batch_size = 4
            elif self.model_arch == "CellViTUNI":
                max_batch_size = 8
            elif self.model_arch == "CellViT256":
                max_batch_size = 8
        elif self.system_configuration["gpu_memory"] < 78:
            if self.model_arch == "CellViTSAM":
                max_batch_size = 8
            elif self.model_arch == "CellViTUNI":
                max_batch_size = 16
            elif self.model_arch == "CellViT256":
                max_batch_size = 24
        else:
            if self.model_arch == "CellViTSAM":
                max_batch_size = 16
            elif self.model_arch == "CellViTUNI":
                max_batch_size = 32
            elif self.model_arch == "CellViT256":
                max_batch_size = 48
        self.logger.info(
            "Based on the hardware we limit the batch size to a maximum of:"
        )
        self.logger.info(max_batch_size)
        if self.batch_size > max_batch_size:
            self.batch_size = max_batch_size
            self.logger.info(f"Apply limits - Batch size: {self.batch_size}")

    def _load_classifier(self) -> None:
        if self.nuclei_taxonomy == "pannuke":
            self.logger.info("Using default PanNuke classes")
            return 0
        elif self.nuclei_taxonomy == "binary":
            self.binary = True
            self.label_map = {0: "background", 1: "nuclei"}
            self.logger.info("Using binary detection")
        else:
            classifier_path = cache_classifier(self.logger)
            if self.model_name == "SAM":
                classifier_path = classifier_path / "sam-h"
            elif self.model_name == "HIPT":
                classifier_path = classifier_path / "vit256"
            else:
                raise ValueError("Unknown model name")
            match self.nuclei_taxonomy:
                case "lizard":
                    self.logger.info("Using Lizard classifier")
                    model_checkpoint = classifier_path / "lizard.pth"
                case "consep":
                    self.logger.info("Using CoNSeP classifier")
                    model_checkpoint = classifier_path / "consep.pth"
                case "midog":
                    self.logger.info("Using MIDOG classifier")
                    model_checkpoint = classifier_path / "midog.pth"
                case "nucls_main":
                    self.logger.info("Using NUCLS Main classifier")
                    model_checkpoint = classifier_path / "nucls_main.pth"
                case "nucls_super":
                    self.logger.info("Using NUCLS Super classifier")
                    model_checkpoint = classifier_path / "nucls_super.pth"
                case "ocelot":
                    self.logger.info("Using Ocelot classifier")
                    model_checkpoint = classifier_path / "ocelot.pth"
                case "panoptils":
                    self.logger.info("Using Panoptils classifier")
                    model_checkpoint = classifier_path / "panoptils.pth"
                case _:
                    self.logger.error(
                        f"Unknown classifier: {self.nuclei_taxonomy}, using default settings"
                    )
                    raise NotImplementedError("Unknown classifier")
            assert (
                model_checkpoint.exists()
            ), f"Classifier cannot be loaded - Expecting {str(model_checkpoint)}"
            model_checkpoint = torch.load(model_checkpoint, map_location="cpu")
            run_conf = unflatten_dict(model_checkpoint["config"], ".")
            model = LinearClassifier(
                embed_dim=model_checkpoint["model_state_dict"]["fc1.weight"].shape[1],
                hidden_dim=run_conf["model"].get("hidden_dim", 100),
                num_classes=run_conf["data"]["num_classes"],
                drop_rate=0,
            )
            self.logger.info(
                model.load_state_dict(model_checkpoint["model_state_dict"])
            )
            model.eval()
            self.label_map = run_conf["data"]["label_map"]
            self.classifier = model

        self.label_map = {int(k): v for k, v in self.label_map.items()}

    def _load_inference_transforms(self):
        """Load the inference transformations from the run_configuration"""
        self.logger.info("Loading inference transformations")

        transform_settings = self.run_conf["transformations"]
        if "normalize" in transform_settings:
            mean = transform_settings["normalize"].get("mean", (0.5, 0.5, 0.5))
            std = transform_settings["normalize"].get("std", (0.5, 0.5, 0.5))
        else:
            mean = (0.5, 0.5, 0.5)
            std = (0.5, 0.5, 0.5)
        self.inference_transforms = T.Compose(
            [T.ToTensor(), T.Normalize(mean=mean, std=std)]
        )

    def _setup_amp(self, enforce_amp: bool = False) -> None:
        """Setup automated mixed precision (amp) for inference.

        Args:
            enforce_amp (bool, optional): Using PyTorch autocasting with dtype float16 to speed up inference. Also good for trained amp networks.
                Can be used to enforce amp inference even for networks trained without amp. Otherwise, the network setting is used.
                Defaults to False.
        """
        if enforce_amp:
            self.mixed_precision = enforce_amp
        else:
            self.mixed_precision = self.run_conf["training"].get(
                "mixed_precision", False
            )

    def _setup_worker(self) -> None:
        """Setup the worker for inference"""
        runtime_env = {"env_vars": {"PYTHONPATH": PYTHON_PATH}}
        # Set the global logging settings

        if self.debug:
            include_dashboard = (
                ray._private.utils.check_dashboard_dependencies_installed()
            )
            logging_level = logging.DEBUG
        else:
            include_dashboard = False
            logging_level = logging.INFO

        # ray logger setup
        formatter = None
        if self.logger.handlers:
            for handler in self.logger.handlers:
                if isinstance(handler, logging.StreamHandler) and hasattr(
                    handler, "formatter"
                ):
                    formatter = handler.formatter
                    break

        # init ray
        ray.init(
            num_cpus=self.system_configuration["cpu_count"] - 2,
            runtime_env=runtime_env,
            object_store_memory=0.3 * self.system_configuration["memory"] * 1024 * 1024,
            include_dashboard=include_dashboard,
            logging_level=logging_level,
            log_to_driver=True,
        )
        # overwrite rays logger style
        if formatter is not None:
            ray_loggers = [
                logging.getLogger("ray"),
                logging.getLogger("ray.worker"),
                logging.getLogger("ray.remote_function"),
                logging.getLogger("ray._private"),  # Covers internal modules
            ]

            for ray_logger in ray_loggers:
                # Modify existing handlers (preserves Ray's logging destinations)
                for handler in ray_logger.handlers:
                    if isinstance(handler, logging.StreamHandler):
                        handler.setFormatter(formatter)

        # workers for loading data
        num_workers = int(3 / 4 * self.system_configuration["cpu_count"])
        if num_workers is None:
            num_workers = 8
        num_workers = int(np.clip(num_workers, 1, 4 * self.batch_size))
        self.num_workers = num_workers

    def _import_postprocessing(self) -> Tuple[Callable, Callable]:
        """Import the postprocessing module

        Returns:
            Tuple[Callable, Callable]: Postprocessing module
        """
        if self.system_configuration["cupy"]:
            from cellvit.inference.postprocessing_cupy import (
                DetectionCellPostProcessor,
                create_batch_pooling_actor,
            )
        else:
            from cellvit.inference.postprocessing_numpy import (
                DetectionCellPostProcessor,
                create_batch_pooling_actor,
            )
        return DetectionCellPostProcessor, create_batch_pooling_actor

    def _post_process_edge_cells(self, cell_list: List[dict]) -> List[int]:
        """Use the CellPostProcessor to remove multiple cells and merge due to overlap

        Args:
            cell_list (List[dict]): List with cell-dictionaries. Required keys:
                * bbox
                * centroid
                * contour
                * type_prob
                * type
                * patch_coordinates
                * cell_status
                * offset_global

        Returns:
            List[int]: List with integers of cells that should be kept
        """
        cell_cleaner = OverlapCellCleaner(cell_list, self.logger)
        cleaned_cells = cell_cleaner.clean_detected_cells()

        return list(cleaned_cells.index.values)

    def _reallign_grid(
        self,
        cell_dict_wsi: list[dict],
        cell_dict_detection: list[dict],
        graph_data,
        rescaling_factor: float,
    ) -> Tuple[list[dict], list[dict]]:
        """Reallign grid if interpolation was used (including target_mpp_tolerance)

        Args:
            cell_dict_wsi (list[dict]):  Input cell dict
            cell_dict_detection (list[dict]): Input cell dict (detection)
            graph_data (_type_): Graph
            rescaling_factor (float): Rescaling Factor

        Returns:
            Tuple[list[dict],list[dict]]:
                * Realligned cell dict (contours)
                * Realligned cell dict (detection)
        """
        for cell in cell_dict_detection:
            cell["bbox"][0][0] = cell["bbox"][0][0] * rescaling_factor
            cell["bbox"][0][1] = cell["bbox"][0][1] * rescaling_factor
            cell["bbox"][1][0] = cell["bbox"][1][0] * rescaling_factor
            cell["bbox"][1][1] = cell["bbox"][1][1] * rescaling_factor
            cell["centroid"][0] = cell["centroid"][0] * rescaling_factor
            cell["centroid"][1] = cell["centroid"][1] * rescaling_factor
        for cell in cell_dict_wsi:
            cell["bbox"][0][0] = cell["bbox"][0][0] * rescaling_factor
            cell["bbox"][0][1] = cell["bbox"][0][1] * rescaling_factor
            cell["bbox"][1][0] = cell["bbox"][1][0] * rescaling_factor
            cell["bbox"][1][1] = cell["bbox"][1][1] * rescaling_factor
            cell["centroid"][0] = cell["centroid"][0] * rescaling_factor
            cell["centroid"][1] = cell["centroid"][1] * rescaling_factor
            cell["contour"] = [
                [round(c[0] * rescaling_factor), round(c[1] * rescaling_factor)]
                for c in cell["contour"]
            ]
        positions_cleaned = []
        for cell in graph_data["positions"]:
            positions_cleaned.append(cell * rescaling_factor)
        graph_data["positions"] = positions_cleaned

        return cell_dict_wsi, cell_dict_detection, graph_data

    def _convert_json_geojson(
        self, cell_list: list[dict], polygons: bool = False
    ) -> List[dict]:
        """Convert a list of cells to a geojson object

        Either a segmentation object (polygon) or detection points are converted

        Args:
            cell_list (list[dict]): Cell list with dict entry for each cell.
                Required keys for detection:
                    * type
                    * centroid
                Required keys for segmentation:
                    * type
                    * contour
            polygons (bool, optional): If polygon segmentations (True) or detection points (False). Defaults to False.

        Returns:
            List[dict]: Geojson like list
        """
        if polygons:
            cell_segmentation_df = pd.DataFrame(cell_list)
            detected_types = sorted(cell_segmentation_df.type.unique())
            geojson_placeholder = []
            for cell_type in detected_types:
                cells = cell_segmentation_df[cell_segmentation_df["type"] == cell_type]
                contours = cells["contour"].to_list()
                final_c = []
                for c in contours:
                    c.append(c[0])
                    final_c.append([c])

                cell_geojson_object = get_template_segmentation()
                cell_geojson_object["id"] = str(uuid.uuid4())
                cell_geojson_object["geometry"]["coordinates"] = final_c
                cell_geojson_object["properties"]["classification"][
                    "name"
                ] = self.label_map[cell_type]
                cell_geojson_object["properties"]["classification"][
                    "color"
                ] = COLOR_DICT_CELLS[cell_type]
                geojson_placeholder.append(cell_geojson_object)
        else:
            cell_detection_df = pd.DataFrame(cell_list)
            detected_types = sorted(cell_detection_df.type.unique())
            geojson_placeholder = []
            for cell_type in detected_types:
                cells = cell_detection_df[cell_detection_df["type"] == cell_type]
                centroids = cells["centroid"].to_list()
                cell_geojson_object = get_template_point()
                cell_geojson_object["id"] = str(uuid.uuid4())
                cell_geojson_object["geometry"]["coordinates"] = centroids
                cell_geojson_object["properties"]["classification"][
                    "name"
                ] = self.label_map[cell_type]
                cell_geojson_object["properties"]["classification"][
                    "color"
                ] = COLOR_DICT_CELLS[cell_type]
                geojson_placeholder.append(cell_geojson_object)
        return geojson_placeholder

    def _remove_padding(
        self, cell_dict_wsi, cell_dict_detection, graph_data, wsi_dimension
    ):
        width, height = wsi_dimension
        correction_width = 1024 - width - 32
        correction_height = 1024 - height - 32
        correction_tensor = torch.tensor([correction_width, correction_height])

        for cell in cell_dict_detection:
            cell["bbox"][0][0] = cell["bbox"][0][0] - correction_height
            cell["bbox"][0][1] = cell["bbox"][0][1] - correction_width
            cell["bbox"][1][0] = cell["bbox"][1][0] - correction_height
            cell["bbox"][1][1] = cell["bbox"][1][1] - correction_width
            cell["centroid"][0] = cell["centroid"][0] - correction_width
            cell["centroid"][1] = cell["centroid"][1] - correction_height
        for cell in cell_dict_wsi:
            cell["bbox"][0][0] = cell["bbox"][0][0] - correction_height
            cell["bbox"][0][1] = cell["bbox"][0][1] - correction_width
            cell["bbox"][1][0] = cell["bbox"][1][0] - correction_height
            cell["bbox"][1][1] = cell["bbox"][1][1] - correction_width
            cell["centroid"][0] = cell["centroid"][0] - correction_width
            cell["centroid"][1] = cell["centroid"][1] - correction_height
            cell["contour"] = [
                [round(c[0] - correction_width), round(c[1] - correction_height)]
                for c in cell["contour"]
            ]
        positions_cleaned = []
        for cell in graph_data["positions"]:
            positions_cleaned.append(cell - correction_tensor)
        graph_data["positions"] = positions_cleaned

        return cell_dict_wsi, cell_dict_detection, graph_data

    def process_wsi(
        self,
        wsi_path: Union[Path, str],
        wsi_mpp: float = None,
        wsi_magnification: float = None,
        apply_prefilter: bool = True,
        filter_patches: bool = False,
        **kwargs,
    ) -> None:
        """Process a whole slide image with CellViT.

        Args:
            wsi_path (Union[Path, str]): Path to the whole slide image.
            wsi_properties (dict, optional): Optional WSI properties,
                Allowed keys are 'slide_mpp' and 'magnification'. Defaults to {}.
            resolution (float, optional): Target resolution. Defaults to 0.25.
            apply_prefilter (bool, optional): Prefilter. Defaults to True.
            filter_patches (bool, optional): Filter patches after processing. Defaults to False.
        """
        wsi_path = Path(wsi_path)
        self.logger.info(f"Processing WSI: {wsi_path.name}")
        self.logger.info(f"Preparing WSI - Loading tissue region and prepare patches")

        # create output directory
        self.outdir.mkdir(exist_ok=True, parents=True)
        wsi_outdir = self.outdir / wsi_path.stem
        wsi_outdir.mkdir(exist_ok=True, parents=True)

        # load metadata
        slide_meta, target_mpp = load_wsi_meta(
            wsi_path=wsi_path,
            wsi_mpp=wsi_mpp,
            wsi_magnification=wsi_magnification,
            logger=self.logger,
        )

        # setup wsi dataloader and postprocessor
        dataset_config = LivePatchWSIConfig(
            wsi_path=str(wsi_path),
            wsi_properties=slide_meta,
            patch_size=self.patch_size,
            patch_overlap=(self.overlap / self.patch_size) * 100,
            target_mpp=target_mpp,
            apply_prefilter=apply_prefilter,
            filter_patches=filter_patches,
            target_mpp_tolerance=0.035,
            **kwargs,
        )
        wsi_inference_dataset = LivePatchWSIDataset(
            slide_processor_config=dataset_config,
            logger=self.logger,
            transforms=self.inference_transforms,
        )
        if self.debug:
            (wsi_outdir / "masks").mkdir(exist_ok=True, parents=True)
            for img_name, img in wsi_inference_dataset.mask_images.items():
                img.save(wsi_outdir / "masks" / f"{img_name}.jpeg", quality=50)
        wsi_inference_dataset.mask_images = None  # clean to free up memory

        wsi_inference_dataloader = LivePatchWSIDataloader(
            dataset=wsi_inference_dataset, batch_size=self.batch_size, shuffle=False
        )
        wsi = WSIMetadata(
            name=wsi_path.name,
            slide_path=wsi_path,
            metadata=wsi_inference_dataset.wsi_metadata,
        )

        # global postprocessor
        (
            DetectionCellPostProcessor,
            create_batch_pooling_actor,
        ) = self._import_postprocessing()
        BatchPoolingActor = create_batch_pooling_actor(
            self.system_configuration["ray_remote_cpus"]
        )

        postprocessor = DetectionCellPostProcessor(
            wsi=wsi,
            nr_types=self.run_conf["data"]["num_nuclei_classes"],
            classifier=self.classifier,
            binary=self.binary,
        )

        # create ray actors for batch-wise postprocessing
        batch_pooling_actors = [
            BatchPoolingActor.remote(postprocessor, self.run_conf)
            for i in range(self.system_configuration["ray_worker"])
        ]

        call_ids = []
        batch_results_list = []

        self.logger.info("Extracting cells using CellViT...")
        with torch.no_grad():
            pbar = tqdm.tqdm(
                wsi_inference_dataloader, total=len(wsi_inference_dataloader)
            )
            pbar.set_postfix(status="Running CellViT...")
            for batch_num, batch in enumerate(wsi_inference_dataloader):
                # check if batch is empty, then continue
                if isinstance(batch[0], list):
                    if len(batch[0]) == 0:
                        pbar.update(1)
                        pbar.total = len(wsi_inference_dataloader)
                        continue
                patches = batch[0].to(self.device)
                metadata = batch[1]
                memory_percentage = (
                    self.system_configuration.get_current_memory_percentage()
                )
                pbar.set_postfix(
                    status="Running CellViT...", mem=f"{memory_percentage:.2f}%"
                )

                # select actor to redirect postprocessing to
                batch_actor = batch_pooling_actors[
                    batch_num % self.system_configuration["ray_worker"]
                ]

                # inference with model
                if self.mixed_precision:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        predictions = self.model.forward(patches, retrieve_tokens=True)
                else:
                    predictions = self.model.forward(patches, retrieve_tokens=True)

                predictions = self.apply_softmax_reorder(predictions)

                # postprocessing

                call_id = batch_actor.convert_batch_to_graph_nodes.remote(
                    predictions, metadata
                )
                call_ids.append(call_id)

                # after 50 batches, retrieve results to lower pressure on memory
                pbar.update(1)
                if len(call_ids) >= 50:
                    pbar.set_postfix(status="Buffering postprocessing... (50 batches)")
                    batch_results = ray.get(call_ids)
                    batch_results_list.append(batch_results)
                    ray.internal.free(call_ids)
                    call_ids = []  # Clear the processed IDs

                percentage_actor_alloc = (
                    sum(retrieve_actor_usage()) / self.system_configuration.memory * 100
                )
                if (
                    percentage_actor_alloc >= 50 or percentage_actor_alloc <= 0.1
                ) and memory_percentage >= 70:
                    pbar.set_postfix(status="Re-register worker")
                    batch_results = ray.get(call_ids)
                    batch_results_list.append(batch_results)
                    ray.internal.free(call_ids)
                    call_ids = []
                    [ray.kill(batch_actor) for batch_actor in batch_pooling_actors]
                    batch_pooling_actors = [
                        BatchPoolingActor.remote(postprocessor, self.run_conf)
                        for i in range(self.system_configuration["ray_worker"])
                    ]

                pbar.total = len(wsi_inference_dataloader)

            self.logger.info("Waiting for final batches to be processed...")
            if len(call_ids) > 0:
                batch_results = ray.get(call_ids)
                batch_results_list.append(batch_results)
        del pbar
        [ray.kill(batch_actor) for batch_actor in batch_pooling_actors]

        # flattend results
        inference_results = list(itertools.chain.from_iterable(batch_results_list))

        # unpack inference results
        cell_dict_wsi = []  # for storing all cell information
        cell_dict_detection = []  # for storing only the centroids

        graph_data = {
            "cell_tokens": [],
            "positions": [],
            "metadata": {
                "wsi_metadata": wsi.metadata,
                "nuclei_types": self.label_map,
            },
            "nuclei_types": [],
        }

        self.logger.info("Unpack Batches")
        for batch_results in inference_results:
            (
                batch_complete_dict,
                batch_detection,
                batch_cell_tokens,
                batch_cell_positions,
            ) = batch_results
            cell_dict_wsi = cell_dict_wsi + batch_complete_dict
            cell_dict_detection = cell_dict_detection + batch_detection
            graph_data["cell_tokens"] = graph_data["cell_tokens"] + batch_cell_tokens
            graph_data["positions"] = graph_data["positions"] + batch_cell_positions
            graph_data["nuclei_types"] = graph_data["nuclei_types"] + [
                v["type"] for v in batch_detection
            ]

        # cleaning overlapping cells
        if len(cell_dict_wsi) == 0:
            self.logger.warning("No cells have been extracted")
            return
        keep_idx = self._post_process_edge_cells(cell_list=cell_dict_wsi)
        cell_dict_wsi = [cell_dict_wsi[idx_c] for idx_c in keep_idx]
        cell_dict_detection = [cell_dict_detection[idx_c] for idx_c in keep_idx]
        graph_data["cell_tokens"] = [
            graph_data["cell_tokens"][idx_c] for idx_c in keep_idx
        ]
        graph_data["positions"] = [graph_data["positions"][idx_c] for idx_c in keep_idx]
        graph_data["nuclei_types"] = [
            graph_data["nuclei_types"][idx_c] for idx_c in keep_idx
        ]
        self.logger.info(f"Detected cells after cleaning: {len(keep_idx)}")

        # reallign grid if interpolation was used (including target_mpp_tolerance)
        if (
            not wsi.metadata["base_mpp"] - 0.035
            <= wsi.metadata["target_patch_mpp"]
            <= wsi.metadata["base_mpp"] + 0.035
        ):
            cell_dict_wsi, cell_dict_detection, graph_data = self._reallign_grid(
                cell_dict_wsi=cell_dict_wsi,
                cell_dict_detection=cell_dict_detection,
                graph_data=graph_data,
                rescaling_factor=wsi_inference_dataset.rescaling_factor,
            )

        # reallign cells if either row or columns is one (dimension in one direction smaller)
        # this has to be done because of white padding
        if (
            wsi.metadata["orig_n_tiles_cols"] == 1
            or wsi.metadata["orig_n_tiles_rows"] == 1
        ):
            self.logger.warning(
                "WSI is smaller than 1024x1024px, we need to remove padding"
            )
            cell_dict_wsi, cell_dict_detection, graph_data = self._remove_padding(
                cell_dict_wsi=cell_dict_wsi,
                cell_dict_detection=cell_dict_detection,
                graph_data=graph_data,
                wsi_dimension=wsi_inference_dataset.tile_extractor.level_dimensions[
                    wsi_inference_dataset.curr_wsi_level
                ],
            )

        # saving/storing
        output_wsi_name = wsi_path.name.split(".")[0]
        cell_dict_wsi = {
            "wsi_metadata": wsi.metadata,
            "type_map": self.label_map,
            "cells": cell_dict_wsi,
        }
        if self.compression:
            with open(str(wsi_outdir / f"cells.json.snappy"), "wb") as outfile:
                compressed_data = snappy.compress(ujson.dumps(cell_dict_wsi, outfile))
                outfile.write(compressed_data)
        else:
            with open(str(wsi_outdir / "cells.json"), "w") as outfile:
                ujson.dump(cell_dict_wsi, outfile)

        if self.geojson:
            self.logger.info("Converting segmentation to geojson")
            geojson_list = self._convert_json_geojson(cell_dict_wsi["cells"], True)
            if self.compression:
                with open(str(wsi_outdir / "cells.geojson.snappy"), "wb") as outfile:
                    compressed_data = snappy.compress(
                        ujson.dumps(geojson_list, outfile)
                    )
                    outfile.write(compressed_data)
            else:
                with open(str(str(wsi_outdir / "cells.geojson")), "w") as outfile:
                    ujson.dump(geojson_list, outfile)

        cell_dict_detection = {
            "wsi_metadata": wsi.metadata,
            "type_map": self.label_map,
            "cells": cell_dict_detection,
        }
        if self.compression:
            with open(str(wsi_outdir / "cell_detection.json.snappy"), "wb") as outfile:
                compressed_data = snappy.compress(
                    ujson.dumps(cell_dict_detection, outfile)
                )
                outfile.write(compressed_data)
        else:
            with open(str(wsi_outdir / "cell_detection.json"), "w") as outfile:
                ujson.dump(cell_dict_detection, outfile)
        if self.geojson:
            self.logger.info("Converting detection to geojson")
            geojson_list = self._convert_json_geojson(
                cell_dict_detection["cells"], False
            )
            if self.compression:
                with open(
                    str(wsi_outdir / "cell_detection.geojson.snappy"),
                    "wb",
                ) as outfile:
                    compressed_data = snappy.compress(
                        ujson.dumps(geojson_list, outfile)
                    )
                    outfile.write(compressed_data)
            else:
                with open(
                    str(str(wsi_outdir / "cell_detection.geojson")),
                    "w",
                ) as outfile:
                    ujson.dump(geojson_list, outfile)

        # store graph
        if self.graph:
            self.logger.info(
                f"Create cell graph with embeddings and save it under: {str(wsi_outdir/ 'cells.pt')}"
            )
            graph = CellGraphDataWSI(
                x=torch.stack(graph_data["cell_tokens"]),
                positions=torch.stack(graph_data["positions"]),
                metadata=graph_data["metadata"],
                nuclei_types=torch.tensor(graph_data["nuclei_types"]),
            )
            torch.save(graph, str(wsi_outdir / "cells.pt"))

        # final output message
        cell_stats_df = pd.DataFrame(cell_dict_wsi["cells"])
        cell_stats = dict(cell_stats_df.value_counts("type"))
        verbose_stats = {self.label_map[k]: v for k, v in cell_stats.items()}
        self.logger.info(f"Finished with cell detection for WSI {output_wsi_name}")
        self.logger.info("Stats:")
        self.logger.info(f"{verbose_stats}")

        # store a json with processed files -> append if already exists
        processed_files = []
        if (self.outdir / "processed_files.json").exists():
            with open(self.outdir / "processed_files.json", "r") as infile:
                processed_files = ujson.load(infile)
        processed_files.append(wsi_path.name)
        processed_files = list(set(processed_files))
        with open(self.outdir / "processed_files.json", "w") as outfile:
            ujson.dump(processed_files, outfile)

    def apply_softmax_reorder(self, predictions: dict) -> dict:
        """Reorder and apply softmax on predictions

        Args:
            predictions(dict): Predictions

        Returns:
            dict: Predictions
        """
        predictions["nuclei_binary_map"] = F.softmax(
            predictions["nuclei_binary_map"], dim=1
        )
        predictions["nuclei_type_map"] = F.softmax(
            predictions["nuclei_type_map"], dim=1
        )
        predictions["nuclei_type_map"] = predictions["nuclei_type_map"].permute(
            0, 2, 3, 1
        )
        predictions["nuclei_binary_map"] = predictions["nuclei_binary_map"].permute(
            0, 2, 3, 1
        )
        predictions["hv_map"] = predictions["hv_map"].permute(0, 2, 3, 1)
        return predictions
