# -*- coding: utf-8 -*-
#
# Help functions to load WSI metadata
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

from typing import Tuple
import logging
from openslide import OpenSlide
from pathlib import Path
from pathopatch.utils.patch_util import target_mpp_to_downsample
import math
import re


def load_wsi_meta(
    wsi_path: Path,
    logger: logging.Logger,
    wsi_mpp: float = None,
    wsi_magnification: float = None,
) -> Tuple[dict, float]:
    """Load WSI metadata from OpenSlide object or from regex

    Args:
        wsi_path (Path): Path to the WSI file
        logger (logging.Logger): Logger object
        wsi_mpp (float, optional): Microns per pixel, overrides OpenSlide metadata. Defaults to None.
        wsi_magnification (float, optional): Magnification, overrides OpenSlide metadata. Defaults to None.

    Raises:
        NotImplementedError: MPP must be defined either by metadata or by config file!
        NotImplementedError: Magnification must be defined either by metadata or by config file!
        NotImplementedError: MPP must be defined either by metadata or by config file!
        RuntimeError: See previous stacktrace, there is an error in the WSI MPP or metadata

    Returns:
        Tuple[dict, float]: WSI metadata and target MPP.
            Keys: mpp, magnification, slide_mpp
    """
    slide_openslide = OpenSlide(str(wsi_path))
    if wsi_mpp is not None:
        slide_mpp = wsi_mpp
    elif "openslide.mpp-x" in slide_openslide.properties:
        slide_mpp = float(slide_openslide.properties["openslide.mpp-x"])
    else:  # last option is to use regex
        try:
            pattern = re.compile(r"MPP(?: =)? (\d+\.\d+)")
            # Use the pattern to find the match in the string
            match = pattern.search(slide_openslide.properties["openslide.comment"])
            # Extract the float value
            if match:
                slide_mpp = float(match.group(1))
            else:
                raise NotImplementedError(
                    "MPP must be defined either by metadata or by config file!"
                )
        except:
            raise NotImplementedError(
                "MPP must be defined either by metadata or by config file!"
            )

    if wsi_magnification is not None:
        slide_mag = wsi_magnification
    elif "openslide.objective-power" in slide_openslide.properties:
        slide_mag = float(slide_openslide.properties.get("openslide.objective-power"))
    elif 0.20 <= slide_mpp <= 0.30:
        logger.warning(
            "Based on resolution we assume x40 magnification. If not, please manually define magnification."
        )
        slide_mag = 40.0
    elif 0.40 <= slide_mpp <= 0.60:
        logger.warning(
            "Based on resolution we assume x20 magnification. If not, please manually define magnification."
        )
        slide_mag = 20.0
    else:
        raise NotImplementedError(
            "Magnification must be defined either by metadata or by config file!"
        )

    slide_properties = {"mpp": slide_mpp, "magnification": slide_mag}

    if slide_mpp > 0.75:
        logger.error(
            "Slide MPP must be smaller than 0.75 to use CellViT. Check your images for MPP and check if you provided the right WSI metadata."
        )
        logger.error(
            "An example for customized metadata is given in the examples.sh file"
        )
        raise RuntimeError(
            "See previous stacktrace, there is an error in the WSI MPP or metadata"
        )

    if slide_mpp >= 0.20 and slide_mpp <= 0.30:
        target_mpp = slide_mpp
        logger.info(f"Using target_mpp: {target_mpp} instead of 0.25")
    else:
        # target_mpp = 0.25
        downsample, _ = target_mpp_to_downsample(slide_mpp, 0.25)
        target_mpp = find_target_mpp(slide_mpp, downsample)
        logger.warning(
            f"We need to rescale to {target_mpp:<.5f}, handle with care! Check the final results!"
        )

    slide_properties["slide_mpp"] = slide_properties[
        "mpp"
    ]  # to align with pathopatch...
    return slide_properties, target_mpp


def find_target_mpp(base_mpp: float, downsample: int) -> float:
    """Find target mpp closest to 0.25 MPP

    Args:
        base_mpp (float): Base mpp of the slide
        downsample (int): Downsample

    Returns:
        float: Target MPP
    """
    # find the target mpp closest to 0.25 given that the following holds:
    product = base_mpp * downsample
    if product >= 0.20 and product <= 0.30:
        return product
    k_ideal = 4.0 / product
    if k_ideal < 1:
        k = 1
    else:
        k_floor = math.floor(k_ideal)
        k_ceil = math.ceil(k_ideal)

        if math.isclose(k_ideal, k_floor, abs_tol=1e-9):
            k = int(k_floor)
        elif math.isclose(k_ideal, k_ceil, abs_tol=1e-9):
            k = int(k_ceil)
        else:
            target_floor = (k_floor / 16.0) * product
            target_ceil = (k_ceil / 16.0) * product

            diff_floor = abs(target_floor - 0.25)
            diff_ceil = abs(target_ceil - 0.25)

            if diff_floor < diff_ceil:
                k = k_floor
            elif diff_ceil < diff_floor:
                k = k_ceil
            else:
                k = k_ceil  # Prefer higher k if equidistant

    target_mpp = (k / 16.0) * product

    rescaling_factor = target_mpp / (base_mpp * downsample)
    patch_size = rescaling_factor * 1024
    overlap = rescaling_factor * 32
    assert patch_size % 2 == 0, f"Patch size {patch_size} must be even"
    assert overlap % 2 == 0, f"Overlap {overlap} must be even"

    return target_mpp
