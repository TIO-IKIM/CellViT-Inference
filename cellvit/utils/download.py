# -*- coding: utf-8 -*-
#
# Download scripts
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essenfrom pathlib import Pathfrom pathlib import Path


from pathlib import Path
import requests
from tqdm import tqdm
from typing import Union, Optional
from cellvit.utils.logger import NullLogger
import logging


def file_exists(directory_path: Path, file_name: str) -> bool:
    """Check if a file exists in a specific directory.

    Args:
        directory_path (Path): The path of the directory to check.
        file_name (str): The name of the file to check.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    file_path = directory_path / file_name
    return file_path.exists()


def download_file(
    download_link: str, file_path: Path, logger: Optional[logging.Logger] = None
) -> None:
    """Download a file from a link and save it to a specific path.

    Args:
        download_link (str): The link to download the file from.
        file_path (Path): The path to save the downloaded file to.

    Raises:
        HTTPError: If the download request fails.
    """
    logger = logger or NullLogger()
    response = requests.get(download_link, stream=True)

    # Ensure the request was successful
    response.raise_for_status()

    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 KiloByte
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)

    with open(file_path, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        logger.error("ERROR, something went wrong")


def check_and_download(
    directory_path: Union[Path, str],
    file_name: str,
    download_link: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Check if a file exists, and download it if it does not exist.

    Args:
        directory_path (Path): The path of the directory to check.
        file_name (str): The name of the file to check.
        download_link (str): The link to download the file from if it does not exist.
    """
    logger = logger or NullLogger()
    directory_path = Path(directory_path)
    directory_path.mkdir(exist_ok=True, parents=True)
    if not file_exists(directory_path, file_name):
        file_path = directory_path / file_name
        logger.info(f"Downloading file to {file_path}")
        download_file(download_link, file_path)
    else:
        logger.info(f"The file {file_name} already exists in {directory_path}.")
