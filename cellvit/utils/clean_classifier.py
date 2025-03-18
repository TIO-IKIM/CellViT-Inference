# -*- coding: utf-8 -*-
# Clean a CellViT classifier checkpoint by removing unnecessary keys.
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import torch
import argparse


def clean_checkpoint(input_path, output_path, keys_to_remove, config_keys_to_remove):
    # Load the checkpoint
    checkpoint = torch.load(input_path, map_location="cpu")

    # Remove specified keys from the main checkpoint
    for key in keys_to_remove:
        checkpoint.pop(key, None)

    # Clean the config dictionary if it exists
    if "config" in checkpoint and isinstance(checkpoint["config"], dict):
        for key in config_keys_to_remove:
            checkpoint["config"].pop(key, None)

    # Save the cleaned checkpoint
    torch.save(checkpoint, output_path)
    print(f"Cleaned checkpoint saved to {output_path}")


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser(
        description="Clean a CellViT classifier checkpoint by removing unnecessary keys."
    )
    parser.add_argument("input_path", type=str, help="Path to the input checkpoint")
    parser.add_argument(
        "output_path", type=str, help="Path to save the cleaned checkpoint"
    )
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path

    keys_to_remove = [
        "optimizer_state_dict",
        "scheduler_state_dict",
        "best_metric",
        "wandb_id",
        "logdir",
        "run_name",
        "scaler_state_dict",
    ]

    config_keys_to_remove = [
        "logging.mode",
        "logging.project",
        "logging.notes",
        "logging.log_comment",
        "logging.wandb_dir",
        "logging.log_dir",
        "logging.level",
        "logging.run_id",
        "logging.wandb_file",
        "random_seed",
        "gpu",
        "data.dataset_path",
        "data.train_fold",
        "data.val_fold",
        "data.network_name",
        "cellvit_path",
        "training.weighted_sampling",
        "training.cache_cell_dataset",
        "training.batch_size",
        "training.epochs",
        "training.drop_rate",
        "training.optimizer",
        "training.optimizer_hyperparameter.betas",
        "training.optimizer_hyperparameter.lr",
        "training.optimizer_hyperparameter.weight_decay",
        "training.early_stopping_patience",
        "training.eval_every",
        "training.weight_list",
        "training.scheduler.scheduler_type",
        "run_sweep",
        "agent",
        "just_load_model",
    ]

    clean_checkpoint(input_path, output_path, keys_to_remove, config_keys_to_remove)
