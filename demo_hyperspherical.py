#!/usr/bin/env python3
"""
Demo script for running Hyperspherical DRS on CIFAR-100
This demonstrates how to use the enhanced LoRA-Sub-DRS with hyperspherical space
"""

import json
import argparse
import os
import sys
from trainer import train


def create_demo_config():
    """Create a demo configuration for testing H-DRS"""
    config = {
        "prefix": "demo_hsphere",
        "dataset": "cifar100",
        "data_path": "data/",
        "memory_size": 0,
        "memory_per_class": 0,
        "fixed_memory": True,
        "shuffle": False,
        "seed": [42],  # Use different seed for demo
        "init_cls": 5,
        "increment": 5,
        "total_sessions": 5,  # Reduced for demo (25 classes total)
        "model_name": "lorasub_drs",
        "net_type": "sip",
        "embd_dim": 768,
        "num_heads": 12,
        "EPSILON": 1e-8,
        "init_epoch": 5,  # Reduced for demo
        "optim": "Adam",
        "init_lr": 0.0005,
        "init_lr_decay": 0.1,
        "init_weight_decay": 0.0,
        "epochs": 5,  # Reduced for demo
        "fc_lrate": 0.002,
        "lrate": 0.0005,
        "lrate_decay": 0.1,
        "batch_size": 64,  # Reduced for demo
        "weight_decay": 0.0,
        "rank": 10,
        "margin_inter": 1.0,
        "lambada": 0.05,
        "num_workers": 4,  # Reduced for demo
        # Hyperspherical parameters
        "use_hyperspherical": True,
        "spcauchy_rho": 0.5,
        "sphere_dim": 768,
        "kl_beta": 0.1,
        "angular_margin": 0.1,
        "variance_threshold": 0.95,
        "enable_spherical_projection": True,
        "save_prototypes": True,
        "prototype_dir": "./demo_prototypes",
    }

    return config


def create_baseline_config():
    """Create baseline configuration (original LoRA-Sub-DRS)"""
    config = create_demo_config()
    config["prefix"] = "demo_baseline"
    config["use_hyperspherical"] = False
    config["enable_spherical_projection"] = False
    config["prototype_dir"] = "./demo_prototypes_baseline"
    return config


def setup_parser():
    parser = argparse.ArgumentParser(description="Demo Hyperspherical DRS vs Baseline")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["baseline", "hyperspherical", "both"],
        default="both",
        help="Which version to run",
    )
    parser.add_argument("--device", type=str, default="0", help="GPU device")
    parser.add_argument("--eval", action="store_true", help="Evaluation only")
    parser.add_argument(
        "--quick", action="store_true", help="Quick demo with minimal epochs"
    )

    return parser


def save_config(config, filename):
    """Save configuration to JSON file"""
    os.makedirs("configs", exist_ok=True)
    filepath = f"configs/{filename}"
    with open(filepath, "w") as f:
        json.dump(config, f, indent=2)
    return filepath


def run_experiment(config, name, device="0", eval_only=False):
    """Run a single experiment"""
    print(f"\n{'=' * 60}")
    print(f"Running {name}")
    print(f"{'=' * 60}")

    # Save config
    config_file = save_config(config, f"{name.lower().replace(' ', '_')}.json")

    # Setup arguments
    class Args:
        def __init__(self):
            self.config = config_file
            self.device = device
            self.eval = eval_only

    args = Args()

    try:
        # Run training
        train(vars(args))
        print(f"✅ {name} completed successfully!")
        return True

    except Exception as e:
        print(f"❌ {name} failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


def print_comparison_info():
    """Print information about the comparison"""
    print("""
🔬 HYPERSPHERICAL DRS DEMO
=========================

This demo compares:

1. 📊 BASELINE: Original LoRA-Sub-DRS
   - Uses Euclidean distance and standard normalization
   - Prototypes stored in Euclidean space
   - Standard triplet loss with L2 distance

2. 🌐 HYPERSPHERICAL: Enhanced LoRA-Sub-DRS with H-DRS
   - Features normalized to unit hypersphere
   - Angular distance for similarity computation
   - Spherical Cauchy distribution for robust projection
   - Möbius transformations for directional updates
   - Angular triplet loss for better separation

Expected Benefits of H-DRS:
- 🎯 Reduced feature drift (angular distances more stable)
- 📈 Better performance on later tasks
- 💾 More compact prototype storage (unit norm constraint)
- 🔄 Enhanced plasticity-stability trade-off

Watch the logs for:
- Training accuracy trends
- Spherical drift scores (H-DRS only)
- Final task accuracies
""")


def main():
    args = setup_parser().parse_args()

    print_comparison_info()

    # Modify configs for quick demo if requested
    if args.quick:
        print("🚀 Quick demo mode enabled - reducing epochs for faster execution")

    device = args.device
    results = {}

    if args.mode in ["baseline", "both"]:
        config = create_baseline_config()
        if args.quick:
            config["init_epoch"] = 2
            config["epochs"] = 2
            config["total_sessions"] = 3

        success = run_experiment(config, "Baseline LoRA-Sub-DRS", device, args.eval)
        results["baseline"] = success

    if args.mode in ["hyperspherical", "both"]:
        config = create_demo_config()
        if args.quick:
            config["init_epoch"] = 2
            config["epochs"] = 2
            config["total_sessions"] = 3

        success = run_experiment(config, "Hyperspherical DRS", device, args.eval)
        results["hyperspherical"] = success

    # Print results summary
    print(f"\n{'=' * 60}")
    print("DEMO RESULTS SUMMARY")
    print(f"{'=' * 60}")

    for name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{name.upper():20s}: {status}")

    if len(results) == 2:
        print(f"\n📊 Compare the final accuracies and observe:")
        print(f"   - How H-DRS maintains better performance on early tasks")
        print(f"   - Spherical drift scores showing reduced feature drift")
        print(f"   - Overall accuracy improvements with hyperspherical space")

    print(f"\n🔍 Check the experiment logs in:")
    print(f"   - logs/cifar100/ (training logs)")
    print(f"   - ./demo_prototypes/ (spherical prototypes)")
    print(f"   - ./demo_prototypes_baseline/ (baseline prototypes)")


if __name__ == "__main__":
    main()
