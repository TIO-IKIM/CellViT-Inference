# -*- coding: utf-8 -*-
#
# Helper functions for CellViT with cupy
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen


import cupy as cp

from cupyx.scipy import ndimage as ndimage_cp


def remove_small_objects_cp(pred: cp.ndarray, min_size=64, connectivity=1):
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
        selem = ndimage_cp.generate_binary_structure(pred.ndim, connectivity)
        ccs = cp.zeros_like(pred, dtype=cp.int32)
        ndimage_cp.label(
            pred, selem, output=ccs
        )  # https://docs.rapids.ai/api/cucim/stable/api/#cucim.skimage.measure.label
    else:
        ccs = out

    try:
        component_sizes = cp.bincount(ccs.ravel())
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
