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
            match = pattern.search(self.slide_openslide.properties["openslide.comment"])
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
        target_mpp = 0.25
        logger.warning(
            f"We need to rescale to {0.25}, handle with care! Check the final results!"
        )
    slide_properties["slide_mpp"] = slide_properties[
        "mpp"
    ]  # to align with pathopatch...
    return slide_properties, target_mpp
