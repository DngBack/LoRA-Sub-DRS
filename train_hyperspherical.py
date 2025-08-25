#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training script for DRS-Hyperspherical method

This script provides comprehensive training capabilities for the DRS-Hyperspherical
continual learning method, including:
- YAML configuration support
- Comprehensive logging and analysis
- Ablation study generation
- Automatic experiment management
"""

import argparse
import yaml
import json
import os
import sys
import logging
import time
import torch
import numpy as np
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trainer import train
from utils.hyperspherical_analysis import (
    HypersphericalAnalyzer,
    run_comprehensive_ablation,
)


def setup_logging(config: dict) -> str:
    """Setup comprehensive logging for the experiment"""

    # Create logs directory
    dataset_name = config.get("dataset", {}).get("name", "unknown")
    model_name = config.get("model", {}).get("model_name", "unknown")
    prefix = config.get("experiment", {}).get("prefix", "exp")

    log_dir = f"logs/{dataset_name}/{model_name}"
    os.makedirs(log_dir, exist_ok=True)

    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{prefix}_{timestamp}.log")

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )

    logging.info(f"Logging initialized: {log_file}")
    return log_file


def load_config(config_path: str) -> dict:
    """Load configuration from YAML or JSON file"""

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        if config_path.endswith((".yaml", ".yml")):
            config = yaml.safe_load(f)
        else:
            config = json.load(f)

    logging.info(f"Configuration loaded from: {config_path}")
    return config


def flatten_config(config: dict, parent_key: str = "", sep: str = "_") -> dict:
    """Flatten nested configuration dictionary for trainer compatibility"""

    flattened = {}

    for key, value in config.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key

        if isinstance(value, dict):
            flattened.update(flatten_config(value, new_key, sep))
        else:
            flattened[new_key] = value

    return flattened


def setup_experiment_environment(config: dict) -> dict:
    """Setup experiment environment and update configuration"""

    # Set random seeds for reproducibility
    seed = config.get("seed", 1337)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Setup device
    device_config = config.get("device", {})
    gpu_id = device_config.get("gpu_id", "0")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    # Create experiment directory
    save_path = config.get("logging", {}).get("save_path", "./experiments/")
    experiment_name = f"{config.get('experiment', {}).get('prefix', 'exp')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    experiment_dir = os.path.join(save_path, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # Save configuration to experiment directory
    config_save_path = os.path.join(experiment_dir, "config.yaml")
    with open(config_save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    logging.info(f"Experiment directory created: {experiment_dir}")
    logging.info(f"Configuration saved: {config_save_path}")

    # Update config with experiment info
    config["experiment_dir"] = experiment_dir
    config["config_path"] = config_save_path

    return config


def convert_config_for_trainer(config: dict) -> dict:
    """Convert YAML config structure to format expected by trainer"""

    # Start with flattened config
    trainer_config = flatten_config(config)

    # Handle special mappings for backwards compatibility
    mapping = {
        # Model settings
        "model_backbone": "backbone",
        "model_net_type": "net_type",
        "model_model_name": "model_name",
        "model_embd_dim": "embd_dim",
        "model_class_num": "class_num",
        # LoRA settings
        "lora_rank": "lora_rank",
        "lora_alpha": "lora_alpha",
        "lora_dropout": "lora_dropout",
        # Dataset settings
        "dataset_name": "dataset",
        "dataset_init_cls": "init_cls",
        "dataset_increment": "increment",
        "dataset_total_sessions": "total_sessions",
        # Training settings
        "train_batch_size": "batch_size",
        "train_lr": "lrate",
        "train_init_lr": "init_lr",
        "train_fc_lrate": "fc_lrate",
        "train_weight_decay": "weight_decay",
        "train_epochs": "epochs",
        "train_init_epoch": "init_epoch",
        "train_optim": "optim",
        # Hardware settings
        "device_gpu_id": "device",
        "device_num_workers": "num_workers",
        "device_multiple_gpus": "multiple_gpus",
        # Loss settings
        "loss_lambada": "lambada",
        "loss_margin_inter": "margin_inter",
        # Experimental settings
        "experiment_eval": "eval",
        "experiment_debug": "debug",
        "advanced_EPSILON": "EPSILON",
    }

    # Apply mappings
    for yaml_key, trainer_key in mapping.items():
        if yaml_key in trainer_config:
            trainer_config[trainer_key] = trainer_config[yaml_key]

    # Ensure required fields have defaults
    defaults = {
        "net_type": "sip",
        "model_name": "drs_hyperspherical",
        "device": "0",
        "multiple_gpus": [],
        "eval": False,
        "debug": False,
        "EPSILON": 1e-8,
    }

    for key, default_value in defaults.items():
        if key not in trainer_config:
            trainer_config[key] = default_value

    return trainer_config


def run_single_experiment(config_path: str) -> dict:
    """Run a single DRS-Hyperspherical experiment"""

    # Load and setup configuration
    config = load_config(config_path)
    log_file = setup_logging(config)
    config = setup_experiment_environment(config)

    # Convert to trainer format
    trainer_config = convert_config_for_trainer(config)

    # Log experiment start
    logging.info("=" * 70)
    logging.info("Starting DRS-Hyperspherical Experiment")
    logging.info("=" * 70)
    logging.info(f"Configuration: {config_path}")
    logging.info(f"Model: {trainer_config.get('model_name', 'unknown')}")
    logging.info(f"Dataset: {trainer_config.get('dataset', 'unknown')}")
    logging.info(f"Device: {trainer_config.get('device', 'unknown')}")
    logging.info(f"Experiment directory: {config['experiment_dir']}")
    logging.info("=" * 70)

    start_time = time.time()

    try:
        # Initialize analyzer
        experiment_name = config.get("experiment", {}).get(
            "prefix", "DRS-Hyperspherical"
        )
        analyzer = HypersphericalAnalyzer(experiment_name, config["experiment_dir"])

        # Add analyzer to config so the method can use it
        trainer_config["analyzer"] = analyzer

        # Run training
        train(trainer_config)

        # Generate final analysis
        analyzer.generate_visualizations()
        results = analyzer.save_results()

        # Calculate total time
        total_time = time.time() - start_time

        logging.info("=" * 70)
        logging.info("Experiment Completed Successfully")
        logging.info(f"Total time: {total_time:.2f} seconds")
        logging.info(
            f"Final average accuracy: {results['summary']['final_average_accuracy']:.2f}%"
        )
        logging.info(
            f"Average geodesic drift: {results['summary']['average_geodesic_drift']:.4f}"
        )
        logging.info(f"Results saved to: {config['experiment_dir']}")
        logging.info("=" * 70)

        return results

    except Exception as e:
        logging.error(f"Experiment failed with error: {e}")
        logging.error("Full traceback:", exc_info=True)
        raise


def main():
    """Main function for training script"""

    parser = argparse.ArgumentParser(
        description="DRS-Hyperspherical: Spherical Geometry for Continual Learning"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/drs_hyperspherical_cifar100.yaml",
        help="Path to configuration file (YAML or JSON)",
    )

    parser.add_argument(
        "--ablation", action="store_true", help="Generate ablation study configurations"
    )

    parser.add_argument(
        "--ablation-dir",
        type=str,
        default="./ablation_configs/",
        help="Directory for ablation study configurations",
    )

    parser.add_argument(
        "--run-ablation",
        action="store_true",
        help="Run complete ablation study (generates and runs all configs)",
    )

    args = parser.parse_args()

    # Handle ablation study generation
    if args.ablation or args.run_ablation:
        print("=" * 70)
        print("DRS-Hyperspherical Ablation Study Generator")
        print("=" * 70)

        config_paths = run_comprehensive_ablation(
            args.config, args.ablation_dir, run_experiments=args.run_ablation
        )

        if not args.run_ablation:
            print("\nAblation configurations generated!")
            print("\nTo run individual experiments:")
            for config_path in config_paths:
                print(f"  python train_hyperspherical.py --config {config_path}")
            print("\nTo run all experiments:")
            print("  python train_hyperspherical.py --run-ablation")

        return

    # Run single experiment
    try:
        results = run_single_experiment(args.config)
        print(f"\nExperiment completed successfully!")
        print(f"Final accuracy: {results['summary']['final_average_accuracy']:.2f}%")

    except Exception as e:
        print(f"\nExperiment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
