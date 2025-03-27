# -*- coding: utf-8 -*-
# CLI for CellViT inference
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import argparse
from pathlib import Path
import torch
import pandas as pd
import json
import yaml
import pandas as pd
import threading


def parse_wsi_properties(wsi_properties_str):
    try:
        return json.loads(wsi_properties_str)
    except json.JSONDecodeError:
        raise argparse.ArgumentTypeError(f"Invalid JSON format: {wsi_properties_str}")


class InferenceConfiguration:
    def __init__(self, config: dict) -> None:
        """Initializes the InferenceConfiguration object

        Args:
            config (dict): Configuration dictionary

        Attributes:
            model (str): Model to use for inference. Allowed values: 'SAM' or 'HIPT'
            nuclei_taxonomy (str): Nuclei taxonomy to use for inference. Allowed values: 'binary', 'pannuke', 'consep', 'lizard', 'midog', 'nucls_main', 'nucls_super', 'ocelot', 'panoptils'
            gpu (int): Cuda-GPU ID for inference. Default: 0
            enforce_amp (bool): Whether to use mixed precision for inference (enforced). Otherwise network default training settings are used. Default: False
            batch_size (int): Inference batch-size. Default: 8
            outdir (Path): Output directory to store results
            geojson (bool): Set this flag to export results as additional geojson files for loading them into Software like QuPath
            graph (bool): Set this flag to export results as pytorch graph including embeddings (.pt) file
            compression (bool): Set this flag to export results as snappy compressed file
            command (str): Main run command for either performing inference on single WSI-file or on whole dataset
            wsi_path (Path): Path to WSI file
            wsi_folder (Path): Path to the folder where all WSI are stored
            wsi_filelist (pd.DataFrame): Filelist with WSI to process. Must be a .csv file with one row 'path' denoting the paths to all WSI to process. In addition, WSI properties can be provided by adding two additional columns, named 'slide_mpp' and 'magnification'. Other cols are discarded
            wsi_extension (str): The extension types used for the WSI files, see configs.python.config (WSI_EXT)
            wsi_mpp (float): The microns per pixel (mpp) of the WSI
            wsi_magnification (float): The magnification of the WSI
            cpu_count (int): Number of CPU cores to use/available. Recommend to first test automatic derivation, and just change if problems occur. Default: System configuration is used
            ray_worker (int): Number of Ray workers to use
            ray_remote_cpus (int): Number of CPUs to use for Ray workers
            memory (int): RAM ot use
            debug (bool): If debug should be used
        """
        self.model: str
        self.nuclei_taxonomy: str = "pannuke"
        self.gpu: int = 0
        self.enforce_amp: bool = False
        self.batch_size: int = 8
        self.outdir: Path
        self.geojson: bool = False
        self.graph: bool = False
        self.compression: bool = False
        self.command: str
        self.wsi_path: Path = None
        self.wsi_folder: Path = None
        self.wsi_filelist: pd.DataFrame = None
        self.wsi_extension: str = "svs"
        self.wsi_mpp: float = None
        self.wsi_magnification: float = None
        self.cpu_count: int = None
        self.ray_worker: int = None
        self.ray_remote_cpus: int = None
        self.memory: int = None
        self.debug: int = False

        assert isinstance(config, dict), "Config must be of type dict"

        # set model and classifier type
        self.__set_model(config)
        self.__set_nuclei_taxonomy(config)

        # set cellvit related inference properties
        self.__set_gpu(config)
        self.__set_amp(config)
        self.__set_batch_size(config)

        # set output information
        self.__set_outdir(config)
        self.__set_geojson(config)
        self.__set_graph(config)
        self.__set_compression(config)

        # set command
        self.__set_command(config)

        # set system related properties
        self.__set_cpu_count(config)
        self.__set_ray_worker(config)
        self.__set_ray_remote_cpus(config)
        self.__set_memory(config)

        # set debug
        self.__set_debug(config)

    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise KeyError(f"Key '{key}' not found in InferenceConfiguration")

    def __setitem__(self, key, value):
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise KeyError(f"Key '{key}' not found in InferenceConfiguration")

    def __set_model(self, config: dict) -> None:
        """Sets the model to use for inference

        Args:
            config (dict): Configuration dictionary

        Raises:
            AssertionError: If model is not provided in config
            AssertionError: If model is not of type string
            AssertionError: If model is not 'SAM' or 'HIPT'
        """
        assert "model" in config, "Model must be provided in config"
        assert isinstance(config["model"], str), "Model must be of type string"
        assert config["model"].upper() in [
            "SAM",
            "HIPT",
        ], "Model must be either 'SAM' or 'HIPT'"
        self.model = config["model"].upper()

    def __set_nuclei_taxonomy(self, config: dict) -> None:
        """Sets the nuclei taxonomy to use for inference

        Args:
            config (dict): Configuration dictionary

        Raises:
            ValueError: If nuclei taxonomy is not provided in config
            AssertionError: If nuclei taxonomy is not of type string
            AssertionError: If nuclei taxonomy is not one of 'binary', 'pannuke', 'consep', 'lizard', 'midog', 'nucls_main', 'nucls_super', 'ocelot', 'panoptils'
        """
        if "nuclei_taxonomy" in config:
            if config["nuclei_taxonomy"] is not None:
                config["nuclei_taxonomy"] = config["nuclei_taxonomy"].lower()
                assert isinstance(
                    config["nuclei_taxonomy"], str
                ), "Nuclei taxonomy must be of type string"
                assert config["nuclei_taxonomy"] in [
                    "binary",
                    "pannuke",
                    "consep",
                    "lizard",
                    "midog",
                    "nucls_main",
                    "nucls_super",
                    "ocelot",
                    "panoptils",
                ], "Nuclei taxonomy must be one of binary, pannuke, lizard, midog, nucls_main, nucls_super, ocelot, panoptils"
                if self.model == "HIPT" and config["nuclei_taxonomy"] == "midog":
                    raise ValueError("HIPT model does not support midog taxonomy")
                self.nuclei_taxonomy = config["nuclei_taxonomy"]

    def __set_gpu(self, config: dict) -> None:
        """Sets the GPU to use for inference

        Args:
            config (dict): Configuration dictionary

        Raises:
            AssertionError: If GPU is not of type integer
            AssertionError: If GPU is not between 0 and number of available GPUs
        """
        inference_config = config.get("inference")
        if inference_config is None:
            return

        gpu = inference_config.get("gpu")
        if gpu is not None:
            assert isinstance(gpu, int), "GPU must be of type integer"
            assert (
                0 <= gpu < torch.cuda.device_count()
            ), f"GPU must be between 0 and {torch.cuda.device_count()-1}"
            self.gpu = gpu

    def __set_amp(self, config: dict) -> None:
        """Sets the AMP to use for inference

        Args:
            config (dict): Configuration dictionary

        Raises:
            AssertionError: If AMP is not of type boolean
        """
        inference_config = config.get("inference")
        if inference_config is None:
            return

        enforce_amp = inference_config.get("enforce_amp")
        if enforce_amp is not None:
            assert isinstance(enforce_amp, bool), "AMP must be of type boolean"
            self.enforce_amp = enforce_amp

    def __set_batch_size(self, config: dict) -> None:
        """Sets the batch size to use for inference

        Args:
            config (dict): Configuration dictionary

        Raises:
            AssertionError: If batch size is not of type integer
            AssertionError: If batch size is not between 2 and 32
        """
        inference_config = config.get("inference")
        if inference_config is None:
            return

        batch_size = inference_config.get("batch_size")
        if batch_size is not None:
            assert isinstance(batch_size, int), "Batch size must be of type integer"
            assert 1 < batch_size <= 48, "Batch size must be between 2 and 48"
            self.batch_size = batch_size

    def __set_outdir(self, config: dict) -> None:
        """Sets the output directory to store results

        Args:
            config (dict): Configuration dictionary

        Raises:
            AssertionError: If output format is not provided
            AssertionError: If output format is not of type dict
            AssertionError: If output directory is not provided
            AssertionError: If output directory is not of type string
        """
        output_format = config.get("output_format")
        assert output_format is not None, "Output format must be provided"
        assert isinstance(output_format, dict), "Output format must be of type dict"
        assert "outdir" in output_format, "Output directory must be provided"
        assert isinstance(
            output_format["outdir"], str
        ), "Output directory must be of type string"
        self.outdir = Path(output_format["outdir"])

    def __set_geojson(self, config: dict) -> None:
        """Sets the geojson flag to export results as additional geojson files for loading them into Software like QuPath

        Args:
            config (dict): Configuration dictionary

        Raises:
            AssertionError: If geojson is not of type boolean
        """
        output_format = config.get("output_format")
        if "geojson" in output_format and output_format["geojson"] is not None:
            assert isinstance(
                output_format["geojson"], bool
            ), "Geojson must be of type boolean"
            self.geojson = output_format["geojson"]

    def __set_graph(self, config: dict) -> None:
        """Sets the graph flag to export results as pytorch graph including embeddings (.pt) file

        Args:
            config (dict): Configuration dictionary

        Raises:
            AssertionError: If graph is not of type boolean
        """
        output_format = config.get("output_format")
        if "graph" in output_format and output_format["graph"] is not None:
            assert isinstance(
                output_format["graph"], bool
            ), "Graph must be of type boolean"
            self.graph = output_format["graph"]

    def __set_compression(self, config: dict) -> None:
        """Sets the compression flag to export results as snappy compressed file

        Args:
            config (dict): Configuration dictionary

        Raises:
            AssertionError: If compression is not of type boolean
        """
        output_format = config.get("output_format")
        if "compression" in output_format and output_format["compression"] is not None:
            assert isinstance(
                output_format["compression"], bool
            ), "Compression must be of type boolean"
            self.compression = output_format["compression"]

    def __set_cpu_count(self, config: dict) -> None:
        """Sets the number of CPU cores to use/available

        Args:
            config (dict): Configuration dictionary

        Raises:
            AssertionError: If CPU count is not of type integer
            AssertionError: If CPU count is not greater than 0
        """
        system_config = config.get("system")
        if system_config is None:
            return

        cpu_count = system_config.get("cpu_count")
        if cpu_count is not None:
            assert isinstance(cpu_count, int), "CPU count must be of type integer"
            assert cpu_count > 0, "CPU count must be greater than 0"
            self.cpu_count = cpu_count

    def __set_ray_worker(self, config: dict) -> None:
        """Sets the number of Ray workers to use

        Args:
            config (dict): Configuration dictionary

        Raises:
            AssertionError: If Ray worker is not of type integer
            AssertionError: If Ray worker is not greater than 0
        """
        system_config = config.get("system")
        if system_config is None:
            return

        ray_worker = system_config.get("ray_worker")
        if ray_worker is not None:
            assert isinstance(ray_worker, int), "Ray worker must be of type integer"
            assert ray_worker > 0, "Ray worker must be greater than 0"
            assert ray_worker <= 10, "Please provide less then 10 worker"
            self.ray_worker = ray_worker

    def __set_ray_remote_cpus(self, config: dict) -> None:
        """Sets the number of CPUs to use for Ray workers

        Args:
            config (dict): Configuration dictionary

        Raises:
            AssertionError: If Ray remote CPUs is not of type integer
            AssertionError: If Ray remote CPUs is not greater than 0
        """
        system_config = config.get("system")
        if system_config is None:
            return

        ray_remote_cpus = system_config.get("ray_remote_cpus")
        if ray_remote_cpus is not None:
            assert isinstance(
                ray_remote_cpus, int
            ), "Ray remote CPUs must be of type integer"
            assert ray_remote_cpus > 0, "Ray remote worker must be greater than 0"
            self.ray_remote_cpus = ray_remote_cpus

    def __set_memory(self, config: dict) -> None:
        system_config = config.get("system")
        if system_config is None:
            return

        memory = system_config.get("memory")
        if memory is not None:
            assert isinstance(memory, int), "Memory must be integer"
            assert memory >= 8192, "Memory must be larger than 8GB (8192MB)"
            self.memory = memory

    def __set_debug(self, config: dict) -> None:
        debug = config.get("debug")
        if debug is None:
            return
        assert isinstance(debug, bool), "Debug must be true or false (bool)"
        if debug:
            self.debug = True
        else:
            self.debug = False

    def __set_command(self, config: dict) -> None:
        """Sets the main run command for either performing inference on single WSI-file or on whole dataset

        Args:
            config (dict): Configuration dictionary

        Raises:
            AssertionError: If command is not provided
            AssertionError: If command is not of type dict
            AssertionError: If command is not 'process_wsi' or 'process_dataset'
            AssertionError: If WSI path is not provided
            AssertionError: If WSI path is not of type string
            AssertionError: If WSI path does not exist
            AssertionError: If WSI path is not a file
            AssertionError: If WSI MPP is not of type float
            AssertionError: If WSI MPP is not greater than 0
            AssertionError: If WSI magnification is not greater than 0
            AssertionError: If WSI magnification is not greater than 0
            AssertionError: If WSI folder is not provided
            AssertionError: If WSI folder is not of type string
            AssertionError: If WSI folder does not exist
            AssertionError: If WSI folder is not a directory
            AssertionError: If WSI filelist is not of type string
            AssertionError: If WSI filelist does not exist
            AssertionError: If WSI filelist is not a file
            AssertionError: If WSI filelist is not a .csv file
            AssertionError: If WSI filelist does not contain a 'path' column
        """
        assert (
            "process_wsi" in config or "process_dataset" in config
        ), "Command must be provided"
        if "process_wsi" in config:
            assert isinstance(
                config["process_wsi"], dict
            ), "Command must be of type dict"
            self.command = "process_wsi"

            assert "wsi_path" in config["process_wsi"], "WSI path must be provided"
            assert isinstance(
                config["process_wsi"]["wsi_path"], str
            ), "WSI path must be of type string"
            assert Path(
                config["process_wsi"]["wsi_path"]
            ).exists(), "WSI path does not exist"
            if not Path(config["process_wsi"]["wsi_path"]).is_file():
                wsi_path = Path(config["process_wsi"]["wsi_path"])
                files_sorted = sorted(
                    wsi_path.glob("*.dcm"), key=lambda f: f.stat().st_size, reverse=True
                )

                if not len(files_sorted) > 0:
                    raise FileNotFoundError("WSI path is not a file")
                self.wsi_path = files_sorted[0]
            else:
                self.wsi_path = Path(config["process_wsi"]["wsi_path"])

            wsi_mpp = config["process_wsi"].get("wsi_mpp")
            if wsi_mpp is not None:
                assert isinstance(wsi_mpp, float), "WSI MPP must be of type float"
                assert wsi_mpp > 0, "WSI MPP must be greater than 0"
                self.wsi_mpp = wsi_mpp
            wsi_magnification = config["process_wsi"].get("wsi_magnification")
            if wsi_magnification is not None:
                assert wsi_magnification > 0, "WSI magnification must be greater than 0"
                self.wsi_magnification = wsi_magnification
        if "process_dataset" in config:
            assert isinstance(
                config["process_dataset"], dict
            ), "Command must be of type dict"
            process_dataset = config["process_dataset"]
            self.command = "process_dataset"

            assert (
                "wsi_folder" in process_dataset or "wsi_filelist" in process_dataset
            ), "WSI folder or filelist must be provided"
            assert (
                "wsi_folder" in process_dataset
                and process_dataset["wsi_folder"] is not None
            ) ^ (
                "wsi_filelist" in process_dataset
                and process_dataset["wsi_filelist"] is not None
            ), "Either 'wsi_folder' or 'wsi_filelist' must be provided, but not both."
            if (
                "wsi_folder" in process_dataset
                and process_dataset["wsi_folder"] is not None
            ):
                assert isinstance(
                    process_dataset["wsi_folder"], str
                ), "WSI folder must be of type string"
                assert Path(
                    process_dataset["wsi_folder"]
                ).exists(), "WSI folder does not exist"
                assert Path(
                    process_dataset["wsi_folder"]
                ).is_dir(), "WSI folder is not a directory"
                self.wsi_folder = Path(process_dataset["wsi_folder"])

                wsi_extension = process_dataset.get("wsi_extension")
                if wsi_extension is not None:
                    assert isinstance(
                        wsi_extension, str
                    ), "WSI extension must be of type string"
                    self.wsi_extension = wsi_extension
            else:
                assert isinstance(
                    process_dataset["wsi_filelist"], str
                ), "WSI filelist must be of type string"
                assert Path(
                    process_dataset["wsi_filelist"]
                ).exists(), "WSI filelist does not exist"
                assert Path(
                    process_dataset["wsi_filelist"]
                ).is_file(), "WSI filelist is not a file"
                assert Path(process_dataset["wsi_filelist"]).suffix in [
                    ".csv"
                ], "WSI filelist must be a .csv file"
                self.wsi_filelist = Path(process_dataset["wsi_filelist"])
                self.wsi_filelist = pd.read_csv(self.wsi_filelist, delimiter=",")
                filelist_header = self.wsi_filelist.columns.tolist()
                assert (
                    "path" in filelist_header
                ), "Filelist must contain a 'path' column"

            wsi_mpp = process_dataset.get("wsi_mpp")
            if wsi_mpp is not None:
                assert isinstance(wsi_mpp, float), "WSI MPP must be of type float"
                assert wsi_mpp > 0, "WSI MPP must be greater than 0"
                self.wsi_mpp = wsi_mpp
            wsi_magnification = process_dataset.get("wsi_magnification")
            if wsi_magnification is not None:
                assert wsi_magnification > 0, "WSI magnification must be greater than 0"
                self.wsi_magnification = wsi_magnification


class InferenceWSIParser:
    """Parser for in-memory calculation"""

    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="Perform CellViT++ inference",
        )
        parser.add_argument(
            "--config",
            type=str,
            help="Path to a YAML configuration file. If provided, CLI arguments are ignored.",
        )

        parser.add_argument(
            "--model",
            type=str,
            choices=["SAM", "HIPT"],
            help="Segmentation model to use",
        )

        # Nuclei classification taxonomy (OPTIONAL)
        parser.add_argument(
            "--nuclei_taxonomy",
            type=str,
            default="pannuke",
            choices=[
                "binary",
                "pannuke",
                "consep",
                "lizard",
                "midog",
                "nucls_main",
                "nucls_super",
                "ocelot",
                "panoptils",
            ],
            help="Defines the nuclei classification taxonomy",
        )

        # Inference Settings
        inference_group = parser.add_argument_group("Inference Settings")
        inference_group.add_argument(
            "--gpu", type=int, default=0, help="GPU ID to use for inference"
        )
        inference_group.add_argument(
            "--enforce_amp",
            action="store_true",
            help="Whether to use Automatic Mixed Precision (AMP) for inference",
        )
        inference_group.add_argument(
            "--batch_size",
            type=int,
            default=8,
            help="Number of images processed per batch",
        )

        # Output Settings
        output_group = parser.add_argument_group("Output Settings")
        output_group.add_argument(
            "--outdir",
            type=str,
            help="Path to the output directory where results will be stored",
        )
        output_group.add_argument(
            "--geojson",
            action="store_true",
            help="Whether to export results in GeoJSON format (for QuPath or other tools)",
        )
        output_group.add_argument(
            "--graph",
            action="store_true",
            help="Whether to generate a cell graph representation",
        )
        output_group.add_argument(
            "--compression",
            action="store_true",
            help="Whether to use Snappy compression for output files",
        )

        # Processing Mode
        mode_group = parser.add_argument_group("Processing Mode (Choose One)")
        mode_subparsers = parser.add_subparsers(
            dest="mode", help="Select processing mode"
        )

        # Single WSI processing
        wsi_parser = mode_subparsers.add_parser(
            "process_wsi", help="Process a single Whole Slide Image"
        )
        wsi_parser.add_argument(
            "--wsi_path",
            type=str,
            required=True,
            help="Path to the Whole Slide Image (WSI) file",
        )
        wsi_parser.add_argument(
            "--wsi_mpp",
            type=float,
            help="Microns per pixel (spatial resolution of the slide)",
        )
        wsi_parser.add_argument(
            "--wsi_magnification",
            type=int,
            help="Magnification level of the slide (e.g., 20x, 40x)",
        )

        # Dataset processing
        dataset_parser = mode_subparsers.add_parser(
            "process_dataset", help="Process multiple WSI files"
        )
        wsi_source_group = dataset_parser.add_mutually_exclusive_group(required=True)
        wsi_source_group.add_argument(
            "--wsi_folder",
            type=str,
            help="Path to a folder containing multiple WSI files",
        )
        wsi_source_group.add_argument(
            "--wsi_filelist",
            type=str,
            help="Path to a CSV file listing WSI files (must have a 'path' column)",
        )
        dataset_parser.add_argument(
            "--wsi_extension",
            type=str,
            default="svs",
            help="File extension of WSI files (used for detection)",
        )
        dataset_parser.add_argument(
            "--wsi_mpp", type=float, help="Microns per pixel (spatial resolution)"
        )
        dataset_parser.add_argument(
            "--wsi_magnification", type=int, help="Magnification level of the slides"
        )

        # System Settings
        system_group = parser.add_argument_group("System Settings")
        system_group.add_argument(
            "--cpu_count", type=int, help="Number of CPU cores to use for inference"
        )
        system_group.add_argument(
            "--ray_worker",
            type=int,
            help="Number of ray worker to use for inference (limited by cpu-count)",
        )
        system_group.add_argument(
            "--ray_remote_cpus", type=int, help="Number of CPUs per ray worker"
        )
        system_group.add_argument("--memory", type=int, help="RAM in MB to use")

        # Debug Settings
        parser.add_argument(
            "--debug",
            action="store_true",
            help="Enable debug mode (changes logger level and requires ray[default])",
        )

        self.parser = parser

    def _transform_cli_to_yaml_structure(self, opt) -> dict:
        opt_yaml_style = {}
        # general
        opt_yaml_style["model"] = opt["model"]
        opt_yaml_style["nuclei_taxonomy"] = opt["nuclei_taxonomy"]

        # inference
        opt_yaml_style["inference"] = {}
        opt_yaml_style["inference"]["gpu"] = opt["gpu"]
        opt_yaml_style["inference"]["enforce_amp"] = opt["enforce_amp"]
        opt_yaml_style["inference"]["batch_size"] = opt["batch_size"]

        # output format
        opt_yaml_style["output_format"] = {}
        opt_yaml_style["output_format"]["outdir"] = opt["outdir"]
        opt_yaml_style["output_format"]["geojson"] = opt["geojson"]
        opt_yaml_style["output_format"]["graph"] = opt["graph"]
        opt_yaml_style["output_format"]["compression"] = opt["compression"]

        # system setting
        opt_yaml_style["system"] = {}
        opt_yaml_style["system"]["cpu_count"] = opt["cpu_count"]
        opt_yaml_style["system"]["ray_worker"] = opt["ray_worker"]
        opt_yaml_style["system"]["ray_remote_cpus"] = opt["ray_remote_cpus"]
        opt_yaml_style["system"]["memory"] = opt["memory"]
        opt_yaml_style["debug"] = opt["debug"]

        if opt["mode"] == "process_wsi":
            opt_yaml_style["process_wsi"] = {}
            opt_yaml_style["process_wsi"]["wsi_path"] = opt["wsi_path"]
            opt_yaml_style["process_wsi"]["wsi_mpp"] = opt["wsi_mpp"]
            opt_yaml_style["process_wsi"]["wsi_magnification"] = opt[
                "wsi_magnification"
            ]
        elif opt["mode"] == "process_dataset":
            opt_yaml_style["process_dataset"] = {}
            opt_yaml_style["process_dataset"]["wsi_folder"] = opt["wsi_folder"]
            opt_yaml_style["process_dataset"]["wsi_extension"] = opt["wsi_extension"]
            opt_yaml_style["process_dataset"]["wsi_filelist"] = opt["wsi_filelist"]
            opt_yaml_style["process_dataset"]["wsi_mpp"] = opt["wsi_mpp"]
            opt_yaml_style["process_dataset"]["wsi_magnification"] = opt[
                "wsi_magnification"
            ]
        else:
            raise NotImplementedError(
                "Problem occured - use either process_wsi or process_dataset"
            )

        return opt_yaml_style

    def parse_arguments(self) -> dict:
        opt = self.parser.parse_args()
        opt = vars(opt)
        if opt["config"] is not None:
            with open(opt["config"], "r") as f:
                config = yaml.safe_load(f)
        else:
            config = self._transform_cli_to_yaml_structure(opt)

        inf_conf = InferenceConfiguration(config)
        return inf_conf


def get_user_input_with_timeout(
    prompt: str, timeout: int = 5, default: bool = True
) -> bool:
    """Prompt user input, return after timeout

    Args:
        prompt (str): Prompt to ask
        timeout (int, optional): Timeout. Defaults to 5.
        default (bool, optional): Default return value. Defaults to 1 ("yes").

    Returns:
        bool: Yes (1) or No (0)
    """
    print(
        f"{prompt} (yes/no) [default: {['no','yes'][default]} in {timeout} seconds]: ",
        end="",
        flush=True,
    )

    result = [default]

    def user_input():
        response = input().strip().lower()
        if response in ["yes", "y"]:
            result[0] = True
        elif response in ["no", "n"]:
            result[0] = False

    thread = threading.Thread(target=user_input)
    thread.daemon = True
    thread.start()
    thread.join(timeout)

    return result[0]
