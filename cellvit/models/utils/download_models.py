# -*- coding: utf-8 -*-
# Download models from server, TBD in future after publication
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

# import argparse
# import os
# from pathlib import Path

# import requests
# import tqdm


# def file_exists(directory_path: Path, file_name: str) -> bool:
#     """Check if a file exists in a specific directory.

#     Args:
#         directory_path (Path): The path of the directory to check.
#         file_name (str): The name of the file to check.

#     Returns:
#         bool: True if the file exists, False otherwise.
#     """
#     file_path = directory_path / file_name
#     return file_path.exists()


# def download_file(download_link: str, file_path: Path) -> None:
#     """Download a file from a link and save it to a specific path.

#     Args:
#         download_link (str): The link to download the file from.
#         file_path (Path): The path to save the downloaded file to.

#     Raises:
#         HTTPError: If the download request fails.
#     """
#     response = requests.get(download_link, stream=True)

#     # Ensure the request was successful
#     response.raise_for_status()

#     total_size_in_bytes = int(response.headers.get("content-length", 0))
#     block_size = 1024  # 1 KiloByte
#     progress_bar = tqdm.tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)

#     with open(file_path, "wb") as file:
#         for data in response.iter_content(block_size):
#             progress_bar.update(len(data))
#             file.write(data)
#     progress_bar.close()

#     if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
#         print("ERROR, something went wrong")


# def check_and_download(
#     directory_path: Path, file_name: str, download_link: str
# ) -> None:
#     """Check if a file exists, and download it if it does not exist.

#     Args:
#         directory_path (Path): The path of the directory to check.
#         file_name (str): The name of the file to check.
#         download_link (str): The link to download the file from if it does not exist.
#     """
#     if not file_exists(directory_path, file_name):
#         file_path = directory_path / file_name
#         print("Downloading file...")
#         download_file(download_link, file_path)
#         print(
#             f"The file {file_name} has been successfully downloaded and is located in {directory_path}."
#         )
#     else:
#         print(f"The file {file_name} already exists in {directory_path}.")


# def download_all(checkpoint_path):
#     check_and_download(
#         checkpoint_path,
#         "CellViT-256-x40.pth",
#         "https://figshare.com/ndownloader/files/45351919",
#     )
#     check_and_download(
#         checkpoint_path,
#         "CellViT-256-x20.pth",
#         "https://figshare.com/ndownloader/files/45351922",
#     )
#     check_and_download(
#         checkpoint_path,
#         "CellViT-SAM-H-x40.pth",
#         "https://figshare.com/ndownloader/files/45351934",
#     )
#     check_and_download(
#         checkpoint_path,
#         "CellViT-SAM-H-x20.pth",
#         "https://figshare.com/ndownloader/files/45351940",
#     )


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Download checkpoint files")
#     parser.add_argument(
#         "--all",
#         action="store_true",
#         help="Download all checkpoint files",
#     )
#     parser.add_argument(
#         "--file",
#         type=str,
#         choices=[
#             "CellViT-256-x40.pth",
#             "CellViT-256-x20.pth",
#             "CellViT-SAM-H-x40.pth",
#             "CellViT-SAM-H-x20.pth",
#         ],
#         help="Download a specific checkpoint file",
#     )

#     args = parser.parse_args()

#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     project_root = os.path.dirname(os.path.abspath(current_dir))
#     project_root = os.path.dirname(os.path.abspath(project_root))
#     project_root = os.path.dirname(os.path.abspath(project_root))

#     checkpoint_path = Path(project_root) / "checkpoints"
#     assert checkpoint_path.exists(), "Cannot find checkpoint folder"
#     if args.all:
#         download_all(checkpoint_path)
#     elif args.file:
#         filename = args.file
#         url_map = {
#             "CellViT-256-x40.pth": "https://figshare.com/ndownloader/files/45351919",
#             "CellViT-256-x20.pth": "https://figshare.com/ndownloader/files/45351922",
#             "CellViT-SAM-H-x40.pth": "https://figshare.com/ndownloader/files/45351934",
#             "CellViT-SAM-H-x20.pth": "https://figshare.com/ndownloader/files/45351940",
#         }
#         if filename in url_map:
#             check_and_download(checkpoint_path, filename, url_map[filename])
#         else:
#             print("Invalid filename provided.")
#     else:
#         print(
#             "Please specify either --all to download all files or --file to download a specific file."
#         )
