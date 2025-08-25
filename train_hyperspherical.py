#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training script for DRS-Hyperspherical method

This script provides comprehensive training capabilities for the DRS-Hyperspherical
continual learning method, including:
- JSON configuration support
- Comprehensive logging and analysis
- Ablation study generation
- Automatic experiment management
"""

import argparse
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
    """Load configuration from JSON file"""

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
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
    config_save_path = os.path.join(experiment_dir, "config.json")
    with open(config_save_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    logging.info(f"Experiment directory created: {experiment_dir}")
    logging.info(f"Configuration saved: {config_save_path}")

    # Update config with experiment info
    config["experiment_dir"] = experiment_dir
    config["config_path"] = config_save_path

    return config


def convert_config_for_trainer(config: dict) -> dict:
    """Convert JSON config structure to format expected by trainer"""

    trainer_config = {}

    # ========== BASIC SETTINGS ==========
    trainer_config["seed"] = [config.get("seed", 1337)]  # trainer expects a list

    # ========== MODEL SETTINGS ==========
    model_config = config.get("model", {})
    trainer_config["backbone"] = model_config.get("backbone", "vit_b16")
    trainer_config["net_type"] = model_config.get("net_type", "sip")
    trainer_config["model_name"] = model_config.get("model_name", "drs_hyperspherical")
    trainer_config["embd_dim"] = model_config.get("embd_dim", 768)
    trainer_config["class_num"] = model_config.get("class_num", 5)

    # ========== LORA SETTINGS ==========
    lora_config = config.get("lora", {})
    trainer_config["lora_rank"] = lora_config.get("rank", 16)
    trainer_config["lora_alpha"] = lora_config.get("alpha", 16)
    trainer_config["lora_dropout"] = lora_config.get("dropout", 0.0)

    # ========== DATASET SETTINGS ==========
    dataset_config = config.get("dataset", {})
    trainer_config["dataset"] = dataset_config.get("name", "cifar100")
    trainer_config["init_cls"] = dataset_config.get("init_cls", 50)
    trainer_config["increment"] = dataset_config.get("increment", 5)
    trainer_config["total_sessions"] = dataset_config.get("total_sessions", 11)

    # ========== TRAINING SETTINGS ==========
    train_config = config.get("train", {})
    trainer_config["epochs_warm"] = train_config.get("epochs_warm", 4)
    trainer_config["epochs_main"] = train_config.get("epochs_main", 26)
    trainer_config["epochs"] = train_config.get("epochs", 30)
    trainer_config["init_epoch"] = train_config.get("init_epoch", 20)
    trainer_config["batch_size"] = train_config.get("batch_size", 128)
    trainer_config["lrate"] = train_config.get("lr", 0.001)
    trainer_config["init_lr"] = train_config.get("init_lr", 0.001)
    trainer_config["fc_lrate"] = train_config.get("fc_lrate", 0.001)
    trainer_config["weight_decay"] = train_config.get("weight_decay", 1e-5)
    trainer_config["lrate_decay"] = train_config.get("lrate_decay", 0.1)
    trainer_config["init_lr_decay"] = train_config.get("init_lr_decay", 0.1)
    trainer_config["optim"] = train_config.get("optim", "AdamProj")
    trainer_config["ema_momentum"] = train_config.get("ema_momentum", 0.97)

    # ========== SPHERICAL GEOMETRY SETTINGS ==========
    spherical_config = config.get("spherical", {})
    trainer_config["per_class_prototypes"] = spherical_config.get(
        "per_class_prototypes", True
    )
    trainer_config["multi_anchor"] = spherical_config.get("multi_anchor", False)
    trainer_config["num_anchors"] = spherical_config.get("num_anchors", 3)
    trainer_config["pca_energy"] = spherical_config.get("pca_energy", 0.90)
    trainer_config["k_max"] = spherical_config.get("k_max", 128)

    # ========== LOSS SETTINGS ==========
    loss_config = config.get("loss", {})
    trainer_config["s_start"] = loss_config.get("s_start", 10.0)
    trainer_config["s_end"] = loss_config.get("s_end", 30.0)
    trainer_config["m_start"] = loss_config.get("m_start", 0.0)
    trainer_config["m_end"] = loss_config.get("m_end", 0.2)
    trainer_config["triplet_lambda"] = loss_config.get("triplet_lambda", 0.5)
    trainer_config["triplet_margin"] = loss_config.get("triplet_margin", 0.2)
    trainer_config["triplet_mining"] = loss_config.get("triplet_mining", "hard")
    trainer_config["label_smoothing"] = loss_config.get("label_smoothing", 0.05)
    trainer_config["lambada"] = loss_config.get(
        "lambada", 0.5
    )  # Original LoRA-Sub loss weight
    trainer_config["margin_inter"] = loss_config.get("margin_inter", 0.5)

    # ========== DEVICE SETTINGS ==========
    device_config = config.get("device", {})
    trainer_config["device"] = device_config.get("gpu_id", "0")
    trainer_config["num_workers"] = device_config.get("num_workers", 4)
    trainer_config["multiple_gpus"] = device_config.get("multiple_gpus", [])

    # ========== EXPERIMENT SETTINGS ==========
    experiment_config = config.get("experiment", {})
    trainer_config["eval"] = experiment_config.get("eval", False)
    trainer_config["debug"] = experiment_config.get("debug", False)
    trainer_config["prefix"] = experiment_config.get("prefix", "DRS-Hyperspherical")

    # ========== ADVANCED SETTINGS ==========
    advanced_config = config.get("advanced", {})
    trainer_config["EPSILON"] = advanced_config.get("EPSILON", 1e-8)
    trainer_config["fixed_lambda"] = advanced_config.get("fixed_lambda", None)
    trainer_config["use_classifier_alignment"] = advanced_config.get(
        "use_classifier_alignment", False
    )
    trainer_config["use_self_distillation"] = advanced_config.get(
        "use_self_distillation", False
    )
    trainer_config["shuffle"] = advanced_config.get("shuffle", True)

    # ========== LOG CONFIGURATION MAPPING ==========
    logging.info("=== Configuration Mapping ===")
    logging.info(f"Model: {trainer_config['model_name']}")
    logging.info(f"Dataset: {trainer_config['dataset']}")
    logging.info(f"Device: {trainer_config['device']}")
    logging.info(f"Batch size: {trainer_config['batch_size']}")
    logging.info(f"Learning rate: {trainer_config['lrate']}")
    logging.info(f"Epochs: {trainer_config['epochs']}")
    logging.info(f"LoRA rank: {trainer_config['lora_rank']}")
    logging.info(f"PCA energy: {trainer_config['pca_energy']}")
    logging.info(f"EMA momentum: {trainer_config['ema_momentum']}")
    logging.info("============================")

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
        default="configs/drs_hyperspherical_cifar100.json",
        help="Path to configuration file (JSON format)",
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
