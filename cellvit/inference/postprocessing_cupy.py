# -*- coding: utf-8 -*-
# Postprocessing of cellvit networkm output, tailored for the inference pipeline
#
# Cupy
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen
from os import environ
from typing import List, Tuple, Union
from torch import nn
import cv2
import numpy as np
import ray
import torch
import torch.nn.functional as F
from einops import rearrange
from scipy.ndimage import binary_fill_holes
from skimage.segmentation import watershed

from cellvit.data.dataclass.wsi import WSI, WSIMetadata
from cellvit.utils.tools import get_bounding_box, remap_label
from cellvit.utils.tools_cp import remove_small_objects_cp

# does this work then for subfunctions?
import cupy as cp
from cupyx.scipy.ndimage import label as label_cp


class DetectionCellPostProcessor:
    def __init__(
        self,
        wsi: Union[WSI, WSIMetadata],
        nr_types: int,
        classifier: nn.Module = None,
        binary: bool = False,
    ) -> None:
        """DetectionCellPostProcessor for postprocessing prediction maps and get detected cells, based on cupy

        Args:
            wsi (Union[WSI, WSIMetadata]): WSI object for getting metadata
            nr_types (int):  Number of cell types, including background (background = 0). Defaults to None.
            classifier (nn.Module, optional): Add a token classifier to change the cell types based on a custom cell classifier. Defaults to None.
            binary (bool): If just a binary detection/segmentation should be performed. Defaults to False.

        Raises:
            NotImplementedError: Unknown
        """
        self.wsi = wsi
        self.nr_types = nr_types
        self.classifier = classifier
        self.binary = binary
        self.object_size = 10
        self.k_size = 21

    def check_network_output(self, predictions_: dict) -> None:
        """Check if the network output is valid

        Args:
            predictions_ (dict): Network predictions with tokens. Keys (required):
                * nuclei_binary_map: Binary Nucleus Predictions. Shape: (B, H, W, 2)
                * nuclei_type_map: Type prediction of nuclei. Shape: (B, H, W, self.num_nuclei_classes,)
                * hv_map: Horizontal-Vertical nuclei mapping. Shape: (B, H, W, 2)
        """
        b, h, w, _ = predictions_["nuclei_binary_map"].shape
        assert isinstance(predictions_, dict), "predictions_ must be a dictionary"
        assert (
            "nuclei_binary_map" in predictions_
        ), "nuclei_binary_map must be in predictions_"
        assert (
            "nuclei_type_map" in predictions_
        ), "nuclei_binary_map must be in predictions_"
        assert "hv_map" in predictions_, "nuclei_binary_map must be in predictions_"
        assert predictions_["nuclei_binary_map"].shape == (
            b,
            h,
            w,
            2,
        ), "nuclei_binary_map must have shape (B, H, W, 2)"
        assert predictions_["nuclei_type_map"].shape == (
            b,
            h,
            w,
            self.nr_types,
        ), "nuclei_type_map must have shape (B, H, W, self.nr_types)"
        assert predictions_["hv_map"].shape == (
            b,
            h,
            w,
            2,
        ), "hv_map must have shape (B, H, W, 2)"

    def post_process_batch(self, predictions_: dict) -> Tuple[torch.Tensor, List[dict]]:
        """Post process a batch of predictions and generate cell dictionary and instance predictions for each image in a list

        Args:
            predictions_ (dict): Network predictions with tokens. Keys (required):
                * nuclei_binary_map: Binary Nucleus Predictions. Shape: (B, H, W, 2)
                * nuclei_type_map: Type prediction of nuclei. Shape: (B, H, W, self.num_nuclei_classes,)
                * hv_map: Horizontal-Vertical nuclei mapping. Shape: (B, H, W, 2)

        Returns:
            Tuple[torch.Tensor, List[dict]]:
                * torch.Tensor: Instance map. Each Instance has own integer. Shape: (B, H, W)
                * List of dictionaries. Each List entry is one image. Each dict contains another dict for each detected nucleus.
                    For each nucleus, the following information are returned: "bbox", "centroid", "contour", "type_prob", "type"
        """
        b, h, w, _ = predictions_["nuclei_binary_map"].shape
        # checking
        self.check_network_output(predictions_)

        # batch wise
        pred_maps = self._prepare_pred_maps(predictions_)

        # image wise
        cell_dicts = []
        instance_predictions = []
        for i in range(b):
            pred_inst, cells = self.post_process_single_image(pred_maps[i])
            instance_predictions.append(pred_inst)
            cell_dicts.append(cells)

        return torch.Tensor(np.stack(instance_predictions)), cell_dicts

    def post_process_single_image(
        self, pred_map: cp.ndarray
    ) -> Tuple[np.ndarray, dict[int, dict]]:
        """Process one single image and generate cell dictionary and instance predictions

        Args:
            pred_map (cp.ndarray): Combined output of tp, np and hv branches, in the same order. Shape: (H, W, 4)
        Returns:
            Tuple[np.ndarray, dict[int, dict]]: _description_
        """
        pred_inst, pred_type = self._get_pred_inst_tensor(pred_map)
        cells = self._create_cell_dict(pred_inst, pred_type)
        return (pred_inst, cells)

    def _prepare_pred_maps(self, predictions_: dict) -> cp.ndarray:
        """Prepares the prediction maps for post-processing.

        This function takes a dictionary of PyTorch tensors, clones it,
        moves the tensors to the CPU, converts them to numpy arrays, and
        then stacks them along the last axis.

        Args:
            predictions_ (Dict[str, torch.Tensor]): Network predictions with tokens. Keys (required):
                * nuclei_binary_map: Binary Nucleus Predictions. Shape: (B, H, W, 2)
                * nuclei_type_map: Type prediction of nuclei. Shape: (B, H, W, self.num_nuclei_classes,)
                * hv_map: Horizontal-Vertical nuclei mapping. Shape: (B, H, W, 2)

        Returns:
            cp.ndarray: A numpy array containing the stacked prediction maps.
                * shape: B, H, W, 4
                * The last dimension contains the following maps:
                    * channel 0: Type prediction of nuclei
                    * channel 1: Binary Nucleus Predictions
                    * channel 2: Horizontal-Vertical nuclei mapping (X)
                    * channel 3: Horizontal-Vertical nuclei mapping (Y)
        """
        predictions = predictions_.copy()
        predictions["nuclei_type_map"] = cp.asarray(predictions["nuclei_type_map"])
        predictions["nuclei_binary_map"] = cp.asarray(predictions["nuclei_binary_map"])
        predictions["hv_map"] = cp.asarray(predictions["hv_map"])

        return self._stack_pred_maps(
            predictions["nuclei_type_map"],
            predictions["nuclei_binary_map"],
            predictions["hv_map"],
        )

    def _stack_pred_maps(
        self,
        nuclei_type_map: cp.ndarray,
        nuclei_binary_map: cp.ndarray,
        hv_map: cp.ndarray,
    ) -> cp.ndarray:
        """Creates the prediction map for HoVer-Net post-processing

        Args:
        nuclei_binary_map:
            nuclei_type_map (cp.ndarray):  Type prediction of nuclei. Shape: (B, H, W, self.num_nuclei_classes,)
            nuclei_binary_map (cp.ndarray): Binary Nucleus Predictions. Shape: (B, H, W, 2)
            hv_map (cp.ndarray): Horizontal-Vertical nuclei mapping. Shape: (B, H, W, 2)

        Returns:
            cp.ndarray: A numpy array containing the stacked prediction maps. Shape [B, H, W, 4]
        """
        # Assert that the shapes of the inputs are as expected
        assert (
            nuclei_type_map.ndim == 4
        ), "nuclei_type_map must be a 4-dimensional array"
        assert (
            nuclei_binary_map.ndim == 4
        ), "nuclei_binary_map must be a 4-dimensional array"
        assert hv_map.ndim == 4, "hv_map must be a 4-dimensional array"
        assert (
            nuclei_type_map.shape[:-1]
            == nuclei_binary_map.shape[:-1]
            == hv_map.shape[:-1]
        ), "The first three dimensions of all input arrays must be the same"
        assert (
            nuclei_binary_map.shape[-1] == 2
        ), "The last dimension of nuclei_binary_map must have a size of 2"
        assert (
            hv_map.shape[-1] == 2
        ), "The last dimension of hv_map must have a size of 2"
        assert isinstance(
            nuclei_type_map, cp.ndarray
        ), "nuclei_type_map must be a cupy array"
        assert isinstance(
            nuclei_binary_map, cp.ndarray
        ), "nuclei_binary_map must be a cupy array"
        assert isinstance(hv_map, cp.ndarray), "hv_map must be a cupy array"

        nuclei_type_map = cp.argmax(nuclei_type_map, axis=-1)  # argmax: cupy argmax
        nuclei_binary_map = cp.argmax(nuclei_binary_map, axis=-1)  # argmax: cupy argmax
        pred_map = cp.stack(
            (nuclei_type_map, nuclei_binary_map, hv_map[..., 0], hv_map[..., 1]),
            axis=-1,
        )

        assert (
            pred_map.shape[-1] == 4
        ), "The last dimension of pred_map must have a size of 4"

        return pred_map

    def _get_pred_inst_tensor(
        self,
        pred_map: cp.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Process Nuclei Prediction and generate instance map (each instance has unique integer)

        Args:
            pred_map (cp.ndarray): Combined output of tp, np and hv branches, in the same order. Shape: (H, W, 4)

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                * np.ndarray: Instance array with shape (H, W), each instance has unique integer
                * np.ndarray: Type array with shape (H, W), each pixel has the type of the instance
        """
        assert isinstance(pred_map, cp.ndarray), "pred_map must be a numpy array"
        assert pred_map.ndim == 3, "pred_map must be a 3-dimensional array"
        assert (
            pred_map.shape[-1] == 4
        ), "The last dimension of pred_map must have a size of 4"

        pred_type = pred_map[..., :1]
        pred_inst = pred_map[..., 1:]
        pred_type = pred_type.astype(cp.int32)

        pred_inst = cp.squeeze(pred_inst)
        pred_inst = remap_label(self._proc_np_hv(pred_inst))

        # return as numpy array
        return pred_inst, pred_type.squeeze().get()

    def _proc_np_hv(
        self, pred_inst: cp.ndarray, object_size: int = 10, ksize: int = 21
    ) -> np.ndarray:
        """Process Nuclei Prediction with XY Coordinate Map and generate instance map (each instance has unique integer)

        Separate Instances (also overlapping ones) from binary nuclei map and hv map by using morphological operations and watershed

        Args:
            pred (cp.ndarray): Prediction output, assuming. Shape: (H, W, 3)
                * channel 0 contain probability map of nuclei
                * channel 1 containing the regressed X-map
                * channel 2 containing the regressed Y-map
            object_size (int, optional): Smallest oject size for filtering. Defaults to 10
            k_size (int, optional): Sobel Kernel size. Defaults to 21

        Returns:
            np.ndarray: Instance map for one image. Each nuclei has own integer. Shape: (H, W)
        """

        # Check input types and values
        assert isinstance(pred_inst, cp.ndarray), "pred_inst must be a numpy array"
        assert pred_inst.ndim == 3, "pred_inst must be a 3-dimensional array"
        assert (
            pred_inst.shape[2] == 3
        ), "The last dimension of pred_inst must have a size of 3"
        assert isinstance(object_size, int), "object_size must be an integer"
        assert object_size > 0, "object_size must be greater than 0"
        assert isinstance(ksize, int), "ksize must be an integer"
        assert ksize > 0, "ksize must be greater than 0"

        # ensure dtype and extract individual channels
        pred = cp.array(pred_inst, dtype=cp.float32)
        blb_raw = pred[..., 0]
        h_dir_raw = pred[..., 1].get()
        v_dir_raw = pred[..., 2].get()

        blb = cp.array(blb_raw >= 0.5, dtype=cp.int32)
        blb = label_cp(blb)[0]
        blb = remove_small_objects_cp(blb, min_size=10)
        blb[blb > 0] = 1  # background is 0 already

        h_dir = cv2.normalize(
            h_dir_raw,
            None,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )
        v_dir = cv2.normalize(
            v_dir_raw,
            None,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )

        # Apply Sobel filter to the direction maps
        sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=ksize)
        sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=ksize)

        sobelh = 1 - (
            cv2.normalize(
                sobelh,
                None,
                alpha=0,
                beta=1,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_32F,
            )
        )
        sobelv = 1 - (
            cv2.normalize(
                sobelv,
                None,
                alpha=0,
                beta=1,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_32F,
            )
        )

        # Combine the Sobel filtered images
        overall = cp.maximum(cp.asarray(sobelh), cp.asarray(sobelv))
        overall = overall - (1 - blb)
        overall[overall < 0] = 0

        # Create distance map
        dist = (1.0 - overall) * blb
        dist = -cv2.GaussianBlur(dist.get(), (3, 3), 0)

        overall = cp.array(overall >= 0.4, dtype=cp.int32)
        marker = blb - overall
        marker[marker < 0] = 0

        # Apply all
        marker = binary_fill_holes(marker.get()).astype("uint8")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
        marker = label_cp(cp.asarray(marker))[0]
        marker = remove_small_objects_cp(marker, min_size=object_size).get()

        # Separate instances
        proced_pred = watershed(dist, markers=marker, mask=blb.get())

        return proced_pred

    ### Methods related to cell dictionary
    def _create_cell_dict(
        self, pred_inst: np.ndarray, pred_type: np.ndarray
    ) -> dict[int, dict]:
        """Create cell dictionary from instance and type predictions

        Keys of the dictionary:
            * bbox: Bounding box of the cell
            * centroid: Centroid of the cell
            * contour: Contour of the cell
            * type_prob: Probability of the cell type
            * type: Type of the cell

        Args:
            pred_inst (np.ndarray): Instance array with shape (H, W), each instance has unique integer
            pred_type (np.ndarray): Type array with shape (H, W), each pixel has the type of the instance

        Returns:
            dict [int, dict]: Dictionary containing the cell information
        """
        assert isinstance(pred_inst, np.ndarray), "pred_inst must be a numpy array"
        assert pred_inst.ndim == 2, "pred_inst must be a 2-dimensional array"
        assert isinstance(pred_type, np.ndarray), "pred_type must be a numpy array"
        assert pred_type.ndim == 2, "pred_type must be a 2-dimensional array"
        assert (
            pred_inst.shape == pred_type.shape
        ), "pred_inst and pred_type must have the same shape"

        inst_id_list = np.unique(pred_inst)[1:]  # exlcude background
        inst_info_dict = {}

        for inst_id in inst_id_list:
            inst_id, cell_dict = self._create_single_instance_entry(
                inst_id, pred_inst, pred_type
            )
            if cell_dict is not None:
                inst_info_dict[inst_id] = cell_dict

        return inst_info_dict

    def _create_single_instance_entry(
        self, inst_id: int, pred_inst: np.ndarray, pred_type: np.ndarray
    ) -> Tuple[int, dict]:
        """Create a single cell dictionary entry from instance and type predictions

        Args:
            inst_id (int): _description_
            pred_inst (np.ndarray): Instance array with shape (H, W), each instance has unique integer
            pred_type (np.ndarray): Type array with shape (H, W), each pixel has the type of the instance

        Returns:
            Tuple[int, dict]:
                * int: Instance ID
                * dict: Dictionary containing the cell information
                    Keys are: "bbox", "centroid", "contour", "type_prob", "type"
        """
        inst_map_global = pred_inst == inst_id
        inst_bbox = self._get_instance_bbox(inst_map_global)
        inst_map_local = self._get_local_instance_map(inst_map_global, inst_bbox)
        inst_centroid_local, inst_contour_local = self._get_instance_centroid_contour(
            inst_map_local
        )

        if inst_centroid_local is None:
            return inst_id, None

        inst_centroid, inst_contour = self._correct_instance_position(
            inst_centroid_local, inst_contour_local, inst_bbox
        )
        inst_type, inst_type_prob = self._get_instance_type(
            inst_bbox, pred_type, inst_map_local
        )

        return inst_id, {  # inst_id should start at 1
            "bbox": inst_bbox,
            "centroid": inst_centroid,
            "contour": inst_contour,
            "type_prob": inst_type_prob,
            "type": inst_type,
        }

    def _get_instance_bbox(self, inst_map_global: np.ndarray) -> np.ndarray:
        """Get the bounding box of an instance from global instance map (instance map is binary)

        Args:
            inst_map_global (np.ndarray): Binary instance map, Shape: (H, W)

        Returns:
            np.ndarray: Bounding box of the instance. Shape: (2, 2)
                Interpretation: [[rmin, cmin], [rmax, cmax]]
        """
        rmin, rmax, cmin, cmax = get_bounding_box(inst_map_global)
        inst_bbox = np.array([[rmin, cmin], [rmax, cmax]])
        return inst_bbox

    def _get_local_instance_map(
        self, inst_map_global: np.ndarray, inst_bbox: np.ndarray
    ) -> np.ndarray:
        """Get the local instance map from the global instance map, crop it with the bounding box

        Args:
            inst_map_global (np.ndarray): Binary instance map, Shape: (H, W)
            inst_bbox (np.ndarray): Bounding box of the instance. Shape: (2, 2)

        Returns:
            np.ndarray: Local instance map. Shape: (H', W')
        """
        inst_map_local = inst_map_global[
            inst_bbox[0][0] : inst_bbox[1][0], inst_bbox[0][1] : inst_bbox[1][1]
        ]
        inst_map_local = inst_map_local.astype(np.uint8)
        return inst_map_local

    def _get_instance_centroid_contour(
        self, inst_map_local: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get the centroid and contour of an instance from the local instance map

        Coordinates are relative to the local instance map

        Args:
            inst_map_local (np.ndarray): Local instance map. Shape: (H', W')

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                * np.ndarray: Centroid of the instance. Shape: (2,)
                * np.ndarray: Contour of the instance. Shape: (N, 2)
        """
        inst_moment = cv2.moments(inst_map_local)
        inst_contour = cv2.findContours(
            inst_map_local, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        inst_contour = np.squeeze(inst_contour[0][0].astype("int32"))

        if inst_contour.shape[0] < 3 or len(inst_contour.shape) != 2:
            return None, None

        inst_centroid = [
            (inst_moment["m10"] / inst_moment["m00"]),
            (inst_moment["m01"] / inst_moment["m00"]),
        ]
        inst_centroid = np.array(inst_centroid)

        return inst_centroid, inst_contour

    def _correct_instance_position(
        self, inst_centroid: np.ndarray, inst_contour: np.ndarray, inst_bbox: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Correct the position of the centroid and contour of an instance to the global image

        Args:
            inst_centroid (np.ndarray): Centroid of the instance. Shape: (2,)
            inst_contour (np.ndarray): Contour of the instance. Shape: (N, 2)
            inst_bbox (np.ndarray): Bounding box of the instance. Shape: (2, 2)

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                * np.ndarray: Centroid of the instance (global cs). Shape: (2,)
                * np.ndarray: Contour of the instance (global cs). Shape: (N, 2)
        """
        inst_contour[:, 0] += inst_bbox[0][1]  # X
        inst_contour[:, 1] += inst_bbox[0][0]  # Y
        inst_centroid[0] += inst_bbox[0][1]  # X
        inst_centroid[1] += inst_bbox[0][0]  # Y

        return inst_centroid, inst_contour

    def _get_instance_type(
        self, inst_bbox: np.ndarray, pred_type: np.ndarray, inst_map_local: np.ndarray
    ) -> Tuple[int, float]:
        """Get the type of an instance from the local instance map and the type prediction map

        Args:
            inst_bbox (np.ndarray): Bounding box of the instance. Shape: (2, 2)
            pred_type (np.ndarray): Type prediction of nuclei. Shape: (H, W)
            inst_map_local (np.ndarray): Local instance map. Shape: (H', W')

        Returns:
            Tuple[int, float]:
                * int: Type of the instance
                * float: Probability of the instance type
        """
        inst_map_local = inst_map_local.astype(bool)
        inst_type_local = pred_type[
            inst_bbox[0][0] : inst_bbox[1][0], inst_bbox[0][1] : inst_bbox[1][1]
        ][inst_map_local]
        type_list, type_pixels = np.unique(inst_type_local, return_counts=True)
        type_list = list(zip(type_list, type_pixels))
        type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
        inst_type = type_list[0][0]

        # if type is background, select, the 2nd most dominant if exist
        if inst_type == 0:
            if len(type_list) > 1:
                inst_type = type_list[1][0]
        type_dict = {v[0]: v[1] for v in type_list}
        type_prob = type_dict[inst_type] / (np.sum(inst_map_local) + 1.0e-6)

        return int(inst_type), float(type_prob)


def create_batch_pooling_actor(num_cpus: int = 8):
    num_gpus = 0.1
    if environ.get("RAY_GPUS_DEACTIVATION") is not None:
        if environ.get("RAY_GPUS_DEACTIVATION") == "1":
            num_gpus = 0

    @ray.remote(num_cpus=num_cpus, num_gpus=num_gpus)
    class BatchPoolingActor:
        def __init__(
            self,
            detection_cell_postprocessor: DetectionCellPostProcessor,
            run_conf: dict,
        ) -> None:
            """Ray Actor for coordinating the postprocessing of **one** batch

            The postprocessing is done in a separate process to avoid blocking the main process.
            The calculation is done with the help of the `DetectionCellPostProcessorCupy` class.
            This actor acts as a coordinator for the postprocessing of one batch and a wrapper for the `DetectionCellPostProcessorCupy` class.

            Args:
                detection_cell_postprocessor (DetectionCellPostProcessorCupy): Instance of the `DetectionCellPostProcessorCupy` class
                run_conf (dict): Run configuration
            """
            assert "dataset_config" in run_conf, "dataset_config must be in run_conf"
            assert (
                "nuclei_types" in run_conf["dataset_config"]
            ), "nuclei_types must be in run_conf['dataset_config']"
            assert "model" in run_conf, "model must be in run_conf"
            assert (
                "token_patch_size" in run_conf["model"]
            ), "token_patch_size must be in run_conf['model']"

            self.detection_cell_postprocessor = detection_cell_postprocessor
            self.run_conf = run_conf

        def convert_batch_to_graph_nodes(
            self, predictions: dict, metadata: List[dict]
        ) -> Tuple[List[dict], List[dict], List[torch.Tensor], List[torch.Tensor]]:
            """Postprocess a batch of predictions and convert it to graph nodes

            Returns the complete graph nodes (cell dictionary), the detection nodes (cell detection dictionary), the cell tokens and the cell positions


            Args:
                predictions (dict): predictions_ (dict): Network predictions with tokens. Keys (required):
                    * nuclei_binary_map: Binary Nucleus Predictions. Shape: (B, H, W, 2)
                    * nuclei_type_map: Type prediction of nuclei. Shape: (B, H, W, self.num_nuclei_classes,)
                    * hv_map: Horizontal-Vertical nuclei mapping. Shape: (B, H, W, 2)
                metadata List[(dict)]: List of metadata dictionaries for each patch.
                    Each dictionary needs to contain the following keys:
                    * row: Row index of the patch
                    * col: Column index of the patch
                    Other keys are optional

            Returns:
                Tuple[List[dict], List[dict], List[torch.Tensor], List[torch.Tensor]]:
                    * List[dict]: Complete graph nodes (cell dictionary)
                    * List[dict]: Detection nodes (cell detection dictionary)
                    * List[torch.Tensor]: Cell tokens
                    * List[torch.Tensor]: Cell positions (centroid)
            """
            _, cell_dict_batch = self.detection_cell_postprocessor.post_process_batch(
                predictions
            )
            tokens = predictions["tokens"].detach().to("cpu")

            batch_complete = []
            batch_detection = []
            batch_cell_tokens = []
            batch_cell_positions = []

            for idx, (patch_cell_dict, patch_metadata) in enumerate(
                zip(cell_dict_batch, metadata)
            ):
                (
                    patch_complete,
                    patch_detection,
                    patch_cell_tokens,
                    patch_cell_positions,
                ) = self.convert_patch_to_graph_nodes(
                    patch_cell_dict, patch_metadata, tokens[idx]
                )
                batch_complete = batch_complete + patch_complete
                batch_detection = batch_detection + patch_detection
                batch_cell_tokens = batch_cell_tokens + patch_cell_tokens
                batch_cell_positions = batch_cell_positions + patch_cell_positions

            if self.detection_cell_postprocessor.classifier is not None:
                if len(batch_cell_tokens) > 0:
                    batch_cell_tokens_pt = torch.stack(batch_cell_tokens)
                    updated_preds = self.detection_cell_postprocessor.classifier(
                        batch_cell_tokens_pt
                    )
                    updated_preds = F.softmax(updated_preds, dim=1)
                    updated_classes = torch.argmax(updated_preds, dim=1)
                    updated_class_preds = updated_preds[
                        torch.arange(updated_classes.shape[0]), updated_classes
                    ]

                    for f, z in zip(batch_complete, updated_classes):
                        f["type"] = int(z)
                    for f, z in zip(batch_complete, updated_class_preds):
                        f["type_prob"] = int(z)
                    for f, z in zip(batch_detection, updated_classes):
                        f["type"] = int(z)
            if self.detection_cell_postprocessor.binary:
                for f in batch_complete:
                    f["type"] = 1
                for f in batch_detection:
                    f["type"] = 1
                pass

            return (
                batch_complete,
                batch_detection,
                batch_cell_tokens,
                batch_cell_positions,
            )

        def convert_patch_to_graph_nodes(
            self,
            patch_cell_dict: dict,
            patch_metadata: dict,
            patch_tokens: torch.Tensor,
        ) -> Tuple[List[dict], List[dict], List[torch.Tensor], List[torch.Tensor]]:
            """Extract information from a single patch and convert it to graph nodes for a global view

            Args:
                patch_cell_dict (dict): Dictionary containing the cell information.
                    Each dictionary needs to contain the following keys:
                    * bbox: Bounding box of the cell
                    * centroid: Centroid of the cell
                    * contour: Contour of the cell
                    * type_prob: Probability of the cell type
                    * type: Type of the cell
                patch_metadata (dict): Metadata dictionary for the patch.
                    Each dictionary needs to contain the following keys:
                    * row: Row index of the patch
                    * col: Column index of the patch
                    Other keys are optional but are stored in the graph nodes for later use
                patch_tokens (torch.Tensor): Tokens of the patch. Shape: (D, H, W)

            Returns:
                Tuple[List[dict], List[dict], List[torch.Tensor], List[torch.Tensor]]:
                    * List[dict]: Complete graph nodes (cell dictionary) of the patch
                    * List[dict]: Detection nodes (cell detection dictionary) of the patch
                    * List[torch.Tensor]: Cell tokens of the patch
                    * List[torch.Tensor]: Cell positions (centroid) of the patch
            """
            wsi = self.detection_cell_postprocessor.wsi
            patch_cell_detection = {}
            patch_cell_detection["patch_metadata"] = patch_metadata
            patch_cell_detection["type_map"] = self.run_conf["dataset_config"][
                "nuclei_types"
            ]

            wsi_scaling_factor = wsi.metadata["downsampling"]
            patch_size = wsi.metadata["patch_size"]
            x_global = int(
                patch_metadata["row"] * patch_size
                - (patch_metadata["row"] + 0.5) * wsi.metadata["patch_overlap"]
            )
            y_global = int(
                patch_metadata["col"] * patch_size
                - (patch_metadata["col"] + 0.5) * wsi.metadata["patch_overlap"]
            )

            cell_tokens = []
            cell_positions = []
            cell_complete = []
            cell_detections = []

            # extract cell information
            for cell in patch_cell_dict.values():
                if (
                    cell["type"]
                    == self.run_conf["dataset_config"]["nuclei_types"]["Background"]
                ):
                    continue
                offset_global = np.array([x_global, y_global])
                centroid_global = np.rint(
                    (cell["centroid"] + np.flip(offset_global)) * wsi_scaling_factor
                )  # TODO: check for 0.499 mpp slides
                contour_global = (
                    cell["contour"] + np.flip(offset_global)
                ) * wsi_scaling_factor
                bbox_global = (cell["bbox"] + offset_global) * wsi_scaling_factor
                cell_dict = {
                    "bbox": bbox_global.tolist(),
                    "centroid": centroid_global.tolist(),
                    "contour": contour_global.tolist(),
                    "type_prob": cell["type_prob"],
                    "type": cell["type"],
                    "patch_coordinates": [
                        patch_metadata["row"],
                        patch_metadata["col"],
                    ],
                    "cell_status": get_cell_position_marging(
                        bbox=cell["bbox"],
                        patch_size=wsi.metadata["patch_size"],
                        margin=64,
                    ),
                    "offset_global": offset_global.tolist(),
                }
                cell_detection = {
                    "bbox": bbox_global.tolist(),
                    "centroid": centroid_global.tolist(),
                    "type": cell["type"],
                }
                if (
                    np.max(cell["bbox"]) == wsi.metadata["patch_size"]
                    or np.min(cell["bbox"]) == 0
                ):  # Use overlap and patch size
                    position = get_cell_position(
                        cell["bbox"], wsi.metadata["patch_size"]
                    )
                    cell_dict["edge_position"] = True
                    cell_dict["edge_information"] = {}
                    cell_dict["edge_information"]["position"] = position
                    cell_dict["edge_information"]["edge_patches"] = get_edge_patch(
                        position, patch_metadata["row"], patch_metadata["col"]
                    )
                else:
                    cell_dict["edge_position"] = False

                bb_index = cell["bbox"] / self.run_conf["model"]["token_patch_size"]
                bb_index[0, :] = np.floor(bb_index[0, :])
                bb_index[1, :] = np.ceil(bb_index[1, :])
                bb_index = bb_index.astype(np.uint8)
                cell_token = patch_tokens[
                    :, bb_index[0, 0] : bb_index[1, 0], bb_index[0, 1] : bb_index[1, 1]
                ]
                cell_token = torch.mean(
                    rearrange(cell_token, "D H W -> (H W) D"), dim=0
                )

                cell_tokens.append(cell_token)
                cell_positions.append(torch.Tensor(centroid_global))
                cell_complete.append(cell_dict)
                cell_detections.append(cell_detection)

            return cell_complete, cell_detections, cell_tokens, cell_positions

    return BatchPoolingActor


def get_cell_position(bbox: np.ndarray, patch_size: int = 1024) -> List[int]:
    """Get cell position as a list

    Entry is 1, if cell touches the border: [top, right, down, left]

    Args:
        bbox (np.ndarray): Bounding-Box of cell
        patch_size (int, optional): Patch-size. Defaults to 1024.

    Returns:
        List[int]: List with 4 integers for each position
    """
    # bbox = 2x2 array in h, w style
    # bbox[0,0] = upper position (height)
    # bbox[1,0] = lower dimension (height)
    # boox[0,1] = left position (width)
    # bbox[1,1] = right position (width)
    # bbox[:,0] -> x dimensions
    top, left, down, right = False, False, False, False
    if bbox[0, 0] == 0:
        top = True
    if bbox[0, 1] == 0:
        left = True
    if bbox[1, 0] == patch_size:
        down = True
    if bbox[1, 1] == patch_size:
        right = True
    position = [top, right, down, left]
    position = [int(pos) for pos in position]

    return position


def get_cell_position_marging(
    bbox: np.ndarray, patch_size: int = 1024, margin: int = 64
) -> int:
    """Get the status of the cell, describing the cell position

    A cell is either in the mid (0) or at one of the borders (1-8)

    # Numbers are assigned clockwise, starting from top left
    # i.e., top left = 1, top = 2, top right = 3, right = 4, bottom right = 5 bottom = 6, bottom left = 7, left = 8
    # Mid status is denoted by 0

    Args:
        bbox (np.ndarray): Bounding Box of cell
        patch_size (int, optional): Patch-Size. Defaults to 1024.
        margin (int, optional): Margin-Size. Defaults to 64.

    Returns:
        int: Cell Status
    """
    cell_status = None
    if np.max(bbox) > patch_size - margin or np.min(bbox) < margin:
        if bbox[0, 0] < margin:
            # top left, top or top right
            if bbox[0, 1] < margin:
                # top left
                cell_status = 1
            elif bbox[1, 1] > patch_size - margin:
                # top right
                cell_status = 3
            else:
                # top
                cell_status = 2
        elif bbox[1, 1] > patch_size - margin:
            # top right, right or bottom right
            if bbox[1, 0] > patch_size - margin:
                # bottom right
                cell_status = 5
            else:
                # right
                cell_status = 4
        elif bbox[1, 0] > patch_size - margin:
            # bottom right, bottom, bottom left
            if bbox[0, 1] < margin:
                # bottom left
                cell_status = 7
            else:
                # bottom
                cell_status = 6
        elif bbox[0, 1] < margin:
            # bottom left, left, top left, but only left is left
            cell_status = 8
    else:
        cell_status = 0

    return cell_status


def get_edge_patch(position: List[int], row: int, col: int) -> List[List[int]]:
    """Get the edge patches of a cell located at the border

    Args:
        position (List[int]): Position of the cell encoded as a list
            -> See below for a list of positions (1-8)
        row (int): Row position of the patch
        col (int): Col position of the patch

    Returns:
        List[List[int]]: List of edge patches, each patch encoded as list of row and col
    """
    # row starting on bottom or on top?
    if position == [1, 0, 0, 0]:
        # top
        return [[row - 1, col]]
    if position == [1, 1, 0, 0]:
        # top and right
        return [[row - 1, col], [row - 1, col + 1], [row, col + 1]]
    if position == [0, 1, 0, 0]:
        # right
        return [[row, col + 1]]
    if position == [0, 1, 1, 0]:
        # right and down
        return [[row, col + 1], [row + 1, col + 1], [row + 1, col]]
    if position == [0, 0, 1, 0]:
        # down
        return [[row + 1, col]]
    if position == [0, 0, 1, 1]:
        # down and left
        return [[row + 1, col], [row + 1, col - 1], [row, col - 1]]
    if position == [0, 0, 0, 1]:
        # left
        return [[row, col - 1]]
    if position == [1, 0, 0, 1]:
        # left and top
        return [[row, col - 1], [row - 1, col - 1], [row - 1, col]]
