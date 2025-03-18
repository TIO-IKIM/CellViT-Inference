# -*- coding: utf-8 -*-
#
# Download example files for testing
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essenfrom pathlib import Pathfrom pathlib import Path

from pathlib import Path
from cellvit.utils.download import check_and_download
import pyvips


def convert_pyramid(base_path: Path, inname: str, outname: str) -> None:
    """Convert an image to a pyramid image.

    Args:
        base_path (Path): Path to the image
        inname (str): Image name
        outname (str): Output image name
    """
    print("Converting to pyramid image")
    image = pyvips.Image.new_from_file(base_path / inname, access="sequential")
    image.tiffsave(base_path / outname, tile=True, pyramid=True)


def check_test_database() -> None:
    """Check if the test database exists, and download it if it does not exist."""
    print("Checking Test Database")
    base_path = Path(__file__).parent.parent.parent / "test_database" / "MIDOG"
    check_and_download(
        base_path,
        "001.tiff",
        "https://springernature.figshare.com/ndownloader/files/40282099",
    )
    convert_pyramid(base_path, "001.tiff", "001_pyramid.tiff")
    base_path = Path(__file__).parent.parent.parent / "test_database" / "x20_svs"
    check_and_download(
        base_path,
        "CMU-1-Small-Region.svs",
        "https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1-Small-Region.svs",
    )
    base_path = Path(__file__).parent.parent.parent / "test_database" / "x40_svs"
    check_and_download(
        base_path,
        "JP2K-33003-2.svs",
        "https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/JP2K-33003-2.svs",
    )
    print("Test Database is now cached on local machine.")


if __name__ == "__main__":
    check_test_database()
