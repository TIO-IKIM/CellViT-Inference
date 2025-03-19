# -*- coding: utf-8 -*-
# CellViT Inference Pipeline for Whole Slide Images (WSI) in Memory
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

from cellvit.inference.cli import InferenceWSIParser, get_user_input_with_timeout
from cellvit.inference.inference import CellViTInference
from cellvit.utils.ressource_manager import SystemConfiguration
from pathlib import Path
import pandas as pd
import ujson as json


def main():
    # argparse
    configuration_parser = InferenceWSIParser()
    args = configuration_parser.parse_arguments()
    command = args["command"]

    # set up ressource manager
    system_configuration = SystemConfiguration(gpu=args["gpu"])
    if args["cpu_count"] is not None:
        system_configuration.overwrite_available_cpus(args["cpu_count"])
    if args["ray_remote_cpus"] is not None:
        system_configuration.overwrite_ray_remote_cpus(args["ray_remote_cpus"])
    if args["ray_worker"] is not None:
        system_configuration.overwrite_ray_worker(args["ray_worker"])
    if args["memory"] is not None:
        system_configuration.overwrite_memory(args["memory"])
    system_configuration.log_system_configuration()

    # set up inference pipeline
    celldetector = CellViTInference(
        model_name=args["model"],
        outdir=args["outdir"],
        system_configuration=system_configuration,
        nuclei_taxonomy=args["nuclei_taxonomy"],
        batch_size=args["batch_size"],
        geojson=args["geojson"],
        graph=args["graph"],
        compression=args["compression"],
        enforce_amp=args["enforce_amp"],
        debug=args["debug"],
    )

    if command.lower() == "process_wsi":
        celldetector.logger.info("Processing single WSI file")
        celldetector.process_wsi(
            wsi_path=args["wsi_path"],
            wsi_mpp=args["wsi_mpp"],
            wsi_magnification=args["wsi_magnification"],
        )

    elif command.lower() == "process_dataset":
        celldetector.logger.info("Processing whole dataset")
        if args["wsi_filelist"] is not None:
            # TODO: do this
            celldetector.logger.info(f"Loading files from filelist {args['filelist']}")
            wsi_filelist = pd.read_csv(args["filelist"], delimiter=",")
            wsi_filelist = wsi_filelist.to_dict(orient="records")

            # already processed files

            for wsi_index, wsi in enumerate(wsi_filelist):
                celldetector.logger.info(f"Progress: {wsi_index+1}/{len(wsi_filelist)}")
                wsi_path = Path(wsi["path"])
                wsi_properties = {}
                if "slide_mpp" in wsi:
                    wsi_properties["slide_mpp"] = wsi["slide_mpp"]
                if "magnification" in wsi:
                    wsi_properties["magnification"] = wsi["magnification"]
                celldetector.process_wsi(
                    wsi_path=wsi_path,
                    wsi_properties=wsi_properties,
                    resolution=args["resolution"],
                )

        elif args["wsi_folder"] is not None:
            celldetector.logger.info(
                f"Loading all files from folder {args['wsi_folder']}. No filelist provided."
            )
            wsi_filelist = [
                f
                for f in sorted(
                    Path(args["wsi_folder"]).glob(f"**/*.{args['wsi_extension']}")
                )
            ]
            celldetector.logger.info(f"Found {len(wsi_filelist)} files inside folder")

            # check if files are already processed
            if (celldetector.outdir / "processed_files.json").exists():
                processed_files = []
                with open(celldetector.outdir / "processed_files.json", "r") as f:
                    processed_files = json.load(f)
                if len(processed_files) != 0:
                    celldetector.logger.info(
                        f"Found processed files: {len(processed_files)}"
                    )
                    remove_files = get_user_input_with_timeout(
                        f"Should the {len(processed_files)} found files be removed from the processing stack?"
                    )
                    if remove_files:
                        print("\n")
                        wsi_filelist = [
                            f for f in wsi_filelist if f.name not in processed_files
                        ]
                        celldetector.logger.info(
                            f"New processing amount: {len(wsi_filelist)}"
                        )

            for wsi_index, wsi in enumerate(wsi_filelist):
                celldetector.logger.info(f"Progress: {wsi_index+1}/{len(wsi_filelist)}")
                wsi_path = Path(wsi)

                celldetector.process_wsi(
                    wsi_path=wsi_path,
                    wsi_mpp=args["wsi_mpp"],
                    wsi_magnification=args["wsi_magnification"],
                )
        else:
            raise ValueError("Provide either filelist or wsi_folder.")
    celldetector.logger.info("Finished processing")


if __name__ == "__main__":
    main()
