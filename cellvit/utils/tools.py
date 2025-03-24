# -*- coding: utf-8 -*-
#
# Helper functions for CellViT
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import logging
import sys
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd

from scipy import ndimage


def get_bounding_box(img: np.ndarray) -> List[int]:
    """Get the bounding box of a binary image.

    Args:
        img (np.ndarray): Binary image (2D array) where non-zero values represent the object.

    Returns:
        List[int]: Bounding box coordinates in the format [rmin, rmax, cmin, cmax].
    """
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)

    # if the image is empty, return empty list
    if not np.any(rows) or not np.any(cols):
        return []

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]


def remove_small_objects(pred, min_size=64, connectivity=1):
    """Remove connected components smaller than the specified size.

    This function is taken from skimage.morphology.remove_small_objects, but the warning
    is removed when a single label is provided.

    Args:
        pred: input labelled array
        min_size: minimum size of instance in output array
        connectivity: The connectivity defining the neighborhood of a pixel.

    Returns:
        out: output array with instances removed under min_size

    """
    out = pred

    if min_size == 0:  # shortcut for efficiency
        return out

    if out.dtype == bool:
        selem = ndimage.generate_binary_structure(
            pred.ndim, connectivity
        )  # numpy function
        ccs = np.zeros_like(pred, dtype=np.int32)
        ndimage.label(pred, selem, output=ccs)  # generate_binary_structure
    else:
        ccs = out

    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError(
            "Negative value labels are not supported. Try "
            "relabeling the input with `scipy.ndimage.label` or "
            "`skimage.morphology.label`."
        )

    too_small = component_sizes < min_size
    too_small_mask = too_small[ccs]
    out[too_small_mask] = 0

    return out


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten a nested dictionary and insert the sep to seperate keys

    Args:
        d (dict): dict to flatten
        parent_key (str, optional): parent key name. Defaults to ''.
        sep (str, optional): Seperator. Defaults to '.'.

    Returns:
        dict: Flattened dict
    """
    items = []
    for k, v in d.items():
        if type(k) != str:
            k = str(k)
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: dict, sep: str = ".") -> dict:
    """Unflatten a flattened dictionary (created a nested dictionary)

    Args:
        d (dict): Dict to be nested
        sep (str, optional): Seperator of flattened keys. Defaults to '.'.

    Returns:
        dict: Nested dict
    """
    output_dict = {}

    # Sortieren der Schlüssel nach Länge, damit tiefere Pfade zuletzt verarbeitet werden
    sorted_items = sorted(d.items(), key=lambda x: len(x[0].split(sep)))

    for key, value in sorted_items:
        keys = key.split(sep)
        current = output_dict

        for k in keys[:-1]:
            # Wenn der Schlüssel bereits existiert aber kein Dictionary ist,
            # ersetzen wir ihn durch ein Dictionary
            if k in current and not isinstance(current[k], dict):
                current[k] = {}

            # Hinzufügen des Schlüssels falls nicht vorhanden
            if k not in current:
                current[k] = {}

            current = current[k]

        # Setzen des endgültigen Wertes
        current[keys[-1]] = value

    return output_dict


def get_size_of_dict(d: dict) -> int:
    size = sys.getsizeof(d)
    for key, value in d.items():
        size += sys.getsizeof(key)
        size += sys.getsizeof(value)
    return size


def load_wsi_files_from_csv(csv_path: Union[Path, str], wsi_extension: str) -> List:
    """Load filenames from csv file with column name "Filename"

    Args:
        csv_path (Union[Path, str]): Path to csv file
        wsi_extension (str): WSI file ending (suffix)

    Returns:
        List: List of WSI
    """
    wsi_filelist = pd.read_csv(csv_path)
    wsi_filelist = wsi_filelist["Filename"].to_list()
    wsi_filelist = [f for f in wsi_filelist if Path(f).suffix == f".{wsi_extension}"]

    return wsi_filelist


def close_logger(logger: logging.Logger) -> None:
    """Closing a logger savely

    Args:
        logger (logging.Logger): Logger to close
    """
    handlers = logger.handlers[:]
    for handler in handlers:
        logger.removeHandler(handler)
        handler.close()

    logger.handlers.clear()
    logging.shutdown()


def remove_parameter_tag(d: dict, sep: str = ".") -> dict:
    """Remove all paramter tags from dictionary

    Args:
        d (dict): Dict must be flattened with defined seperator
        sep (str, optional): Seperator used during flattening. Defaults to ".".

    Returns:
        dict: Dict with parameter tag removed
    """
    param_dict = {}
    for k, _ in d.items():
        unflattened_keys = k.split(sep)
        new_keys = []
        max_num_insert = len(unflattened_keys) - 1
        for i, k in enumerate(unflattened_keys):
            if i < max_num_insert and k != "parameters":
                new_keys.append(k)
        joined_key = sep.join(new_keys)
        param_dict[joined_key] = {}
    print(param_dict)
    for k, v in d.items():
        unflattened_keys = k.split(sep)
        new_keys = []
        max_num_insert = len(unflattened_keys) - 1
        for i, k in enumerate(unflattened_keys):
            if i < max_num_insert and k != "parameters":
                new_keys.append(k)
        joined_key = sep.join(new_keys)
        param_dict[joined_key][unflattened_keys[-1]] = v

    return param_dict


def remap_label(pred, by_size=False):
    """
    Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3]
    not [0, 2, 4, 6]. The ordering of instances (which one comes first)
    is preserved unless by_size=True, then the instances will be reordered
    so that bigger nucler has smaller ID

    Args:
        pred    : the 2d array contain instances where each instances is marked
                  by non-zero integer
        by_size : renaming with larger nuclei has smaller id (on-top)
    """
    pred_id = list(np.unique(pred))
    if 0 in pred_id:
        pred_id.remove(0)
    if len(pred_id) == 0:
        return pred  # no label
    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        # sort the id by size in descending order
        pair_list = zip(pred_id, pred_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
    return new_pred
