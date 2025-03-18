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


from typing import List, Tuple

import cv2
import numpy as np
import torch
from numba import jit
from scipy.ndimage import binary_fill_holes, measurements
from skimage.segmentation import watershed

from cellvit.utils.tools import get_bounding_box, remove_small_objects


class DetectionCellPostProcessor:
    def __init__(
        self,
        nr_types: int,
        magnification: int = 40,
        gt: bool = False,
    ) -> None:
        """DetectionCellPostProcessor for postprocessing prediction maps and get detected cells

        Args:
            nr_types (int, optional): Number of cell types, including background (background = 0). Defaults to None.
            magnification (Literal[20, 40], optional): Which magnification the data has. Defaults to 40.
            gt (bool, optional): If this is gt data (used that we do not suppress tiny cells that may be noise in a prediction map).
                Defaults to False.

        Raises:
            NotImplementedError: Unknown magnification
        """
        self.nr_types = nr_types
        self.magnification = magnification
        self.gt = gt

        if magnification == 40:
            self.object_size = 10
            self.k_size = 21
        elif magnification == 20:
            self.object_size = 3  # 3 or 40, we used 5
            self.k_size = 11  # 11 or 41, we used 13
        else:
            raise NotImplementedError("Unknown magnification")
        if gt:  # to not supress something in gt!
            self.object_size = 100
            self.k_size = 21

    def check_network_output(self, predictions_: dict) -> None:
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
        self, pred_map: np.ndarray
    ) -> Tuple[np.ndarray, dict[int, dict]]:
        """Process one single image and generate cell dictionary and instance predictions

        Args:
            pred_map (np.ndarray): Combined output of tp, np and hv branches, in the same order. Shape: (H, W, 4)
        Returns:
            Tuple[np.ndarray, dict[int, dict]]: _description_
        """
        pred_inst, pred_type = self._get_pred_inst_tensor(pred_map)
        cells = self._create_cell_dict(pred_inst, pred_type)
        return (pred_inst, cells)

    def _prepare_pred_maps(self, predictions_: dict):
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
            np.ndarray: A numpy array containing the stacked prediction maps.
                * shape: B, H, W, 4
                * The last dimension contains the following maps:
                    * channel 0: Type prediction of nuclei
                    * channel 1: Binary Nucleus Predictions
                    * channel 2: Horizontal-Vertical nuclei mapping (X)
                    * channel 3: Horizontal-Vertical nuclei mapping (Y)
        """
        predictions = predictions_.copy()
        # NOTE: Is it possible to convert a torch array to a cupy array without moving from device to device?
        # convert all to numpy and device
        predictions["nuclei_type_map"] = (
            predictions["nuclei_type_map"].detach().cpu().numpy()
        )
        predictions["nuclei_binary_map"] = (
            predictions["nuclei_binary_map"].detach().cpu().numpy()
        )
        predictions["hv_map"] = predictions["hv_map"].detach().cpu().numpy()

        return stack_pred_maps(
            predictions["nuclei_type_map"],
            predictions["nuclei_binary_map"],
            predictions["hv_map"],
        )

    def _get_pred_inst_tensor(
        self,
        pred_map: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Process Nuclei Prediction and generate instance map (each instance has unique integer)

        Args:
            pred_map (np.ndarray): Combined output of tp, np and hv branches, in the same order. Shape: (H, W, 4)

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                * np.ndarray: Instance array with shape (H, W), each instance has unique integer
                * np.ndarray: Type array with shape (H, W), each pixel has the type of the instance
        """
        assert isinstance(pred_map, np.ndarray), "pred_map must be a numpy array"
        assert pred_map.ndim == 3, "pred_map must be a 3-dimensional array"
        assert (
            pred_map.shape[-1] == 4
        ), "The last dimension of pred_map must have a size of 4"

        pred_type = pred_map[..., :1]
        pred_inst = pred_map[..., 1:]
        pred_type = pred_type.astype(np.int32)

        pred_inst = np.squeeze(pred_inst)
        pred_inst = self._proc_np_hv(pred_inst)

        return pred_inst, pred_type.squeeze()

    def _proc_np_hv(
        self, pred_inst: np.ndarray, object_size: int = 10, ksize: int = 21
    ):
        """Process Nuclei Prediction with XY Coordinate Map and generate instance map (each instance has unique integer)

        Separate Instances (also overlapping ones) from binary nuclei map and hv map by using morphological operations and watershed

        Args:
            pred (np.ndarray): Prediction output, assuming. Shape: (H, W, 3)
                * channel 0 contain probability map of nuclei
                * channel 1 containing the regressed X-map
                * channel 2 containing the regressed Y-map
            object_size (int, optional): Smallest oject size for filtering. Defaults to 10
            k_size (int, optional): Sobel Kernel size. Defaults to 21

        Returns:
            np.ndarray: Instance map for one image. Each nuclei has own integer. Shape: (H, W)
        """

        # Check input types and values
        assert isinstance(pred_inst, np.ndarray), "pred_inst must be a numpy array"
        assert pred_inst.ndim == 3, "pred_inst must be a 3-dimensional array"
        assert (
            pred_inst.shape[2] == 3
        ), "The last dimension of pred_inst must have a size of 3"
        assert isinstance(object_size, int), "object_size must be an integer"
        assert object_size > 0, "object_size must be greater than 0"
        assert isinstance(ksize, int), "ksize must be an integer"
        assert ksize > 0, "ksize must be greater than 0"

        # ensure dtype and extract individual channels
        pred = np.array(pred_inst, dtype=np.float32)
        blb_raw = pred[..., 0]
        h_dir_raw = pred[..., 1]
        v_dir_raw = pred[..., 2]

        blb = np.array(blb_raw >= 0.5, dtype=np.int32)
        blb = measurements.label(blb)[0]
        blb = remove_small_objects(blb, min_size=10)
        blb[blb > 0] = 1  # background is 0 already

        # Normalize the horizontal and vertical direction maps to [0, 1]
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

        # Normalize and invert the Sobel filtered images
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
        overall = np.maximum(sobelh, sobelv)
        overall = overall - (1 - blb)
        overall[overall < 0] = 0

        # Create distance map
        dist = (1.0 - overall) * blb
        dist = -cv2.GaussianBlur(dist, (3, 3), 0)

        # Apply all
        overall = np.array(overall >= 0.4, dtype=np.int32)
        marker = blb - overall
        marker[marker < 0] = 0
        marker = binary_fill_holes(marker).astype("uint8")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
        marker = measurements.label(marker)[0]
        marker = remove_small_objects(marker, min_size=object_size)

        # Separate instances
        proced_pred = watershed(dist, markers=marker, mask=blb)

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


@jit(nopython=True)
def stack_pred_maps(
    nuclei_type_map: np.ndarray, nuclei_binary_map: np.ndarray, hv_map: np.ndarray
) -> np.ndarray:
    """Creates the prediction map for HoVer-Net post-processing

    Args:
    nuclei_binary_map:
        nuclei_type_map (np.ndarray):  Type prediction of nuclei. Shape: (B, H, W, self.num_nuclei_classes,)
        nuclei_binary_map (np.ndarray): Binary Nucleus Predictions. Shape: (B, H, W, 2)
        hv_map (np.ndarray): Horizontal-Vertical nuclei mapping. Shape: (B, H, W, 2)

    Returns:
        np.ndarray: A numpy array containing the stacked prediction maps. Shape [B, H, W, 4]
    """
    # Assert that the shapes of the inputs are as expected
    assert nuclei_type_map.ndim == 4, "nuclei_type_map must be a 4-dimensional array"
    assert (
        nuclei_binary_map.ndim == 4
    ), "nuclei_binary_map must be a 4-dimensional array"
    assert hv_map.ndim == 4, "hv_map must be a 4-dimensional array"
    assert (
        nuclei_type_map.shape[:-1] == nuclei_binary_map.shape[:-1] == hv_map.shape[:-1]
    ), "The first three dimensions of all input arrays must be the same"
    assert (
        nuclei_binary_map.shape[-1] == 2
    ), "The last dimension of nuclei_binary_map must have a size of 2"
    assert hv_map.shape[-1] == 2, "The last dimension of hv_map must have a size of 2"

    nuclei_type_map = np.argmax(nuclei_type_map, axis=-1)
    nuclei_binary_map = np.argmax(nuclei_binary_map, axis=-1)
    pred_map = np.stack(
        (nuclei_type_map, nuclei_binary_map, hv_map[..., 0], hv_map[..., 1]), axis=-1
    )

    assert (
        pred_map.shape[-1] == 4
    ), "The last dimension of pred_map must have a size of 4"

    return pred_map
