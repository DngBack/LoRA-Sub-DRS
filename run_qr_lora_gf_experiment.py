#!/usr/bin/env python3
"""
QR-LoRA-GF Experiment Runner

This script runs experiments comparing QR-LoRA-GF with baseline methods
on CIFAR-100 dataset.
"""

import json
import subprocess
import time
import os
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_experiment(config_path, method_name, seed=1):
    """Run a single experiment"""
    logger.info(f"Running {method_name} with seed {seed}")

    # Update seed in config
    with open(config_path, "r") as f:
        config = json.load(f)

    config["seed"] = [seed]

    # Create temporary config file
    temp_config_path = f"temp_config_{method_name}_{seed}.json"
    with open(temp_config_path, "w") as f:
        json.dump(config, f, indent=2)

    try:
        # Run experiment
        start_time = time.time()
        result = subprocess.run(
            ["python", "main.py", "--config", temp_config_path],
            capture_output=True,
            text=True,
            timeout=3600,
        )  # 1 hour timeout

        end_time = time.time()
        duration = end_time - start_time

        if result.returncode == 0:
            logger.info(f"✓ {method_name} completed successfully in {duration:.1f}s")
            return True, duration
        else:
            logger.error(f"❌ {method_name} failed: {result.stderr}")
            return False, duration

    except subprocess.TimeoutExpired:
        logger.error(f"❌ {method_name} timed out after 1 hour")
        return False, 3600
    except Exception as e:
        logger.error(f"❌ {method_name} failed with exception: {e}")
        return False, 0
    finally:
        # Clean up temporary config
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)


def run_comparison_experiments():
    """Run comparison experiments between different methods"""
    logger.info("Starting QR-LoRA-GF Comparison Experiments")
    logger.info("=" * 60)

    # Define experiments to run
    experiments = [
        {
            "name": "QR-LoRA-GF",
            "config": "configs/cifar100_qr_lora_gf.json",
            "description": "QR-LoRA Subtraction with Gated Fusion",
        },
        {
            "name": "Neuro-LoRA",
            "config": "configs/cifar100_neuro_lora.json",
            "description": "Neuro-LoRA (biologically-inspired)",
        },
        {
            "name": "LoRA-Sub-DRS",
            "config": "configs/cifar100.json",
            "description": "Original LoRA-Sub-DRS",
        },
    ]

    # Run experiments for different seeds
    seeds = [1, 3, 5]
    results = {}

    for exp in experiments:
        method_name = exp["name"]
        config_path = exp["config"]

        logger.info(f"\nRunning {method_name}: {exp['description']}")
        logger.info("-" * 40)

        method_results = []

        for seed in seeds:
            success, duration = run_experiment(config_path, method_name, seed)
            method_results.append(
                {"seed": seed, "success": success, "duration": duration}
            )

        results[method_name] = method_results

        # Summary for this method
        successful_runs = sum(1 for r in method_results if r["success"])
        total_duration = sum(r["duration"] for r in method_results)

        logger.info(f"\n{method_name} Summary:")
        logger.info(f"  Successful runs: {successful_runs}/{len(seeds)}")
        logger.info(f"  Total duration: {total_duration:.1f}s")
        logger.info(f"  Average duration: {total_duration / len(seeds):.1f}s")

    # Overall summary
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 60)

    for method_name, method_results in results.items():
        successful_runs = sum(1 for r in method_results if r["success"])
        total_duration = sum(r["duration"] for r in method_results)
        avg_duration = total_duration / len(method_results)

        logger.info(f"{method_name}:")
        logger.info(
            f"  Success rate: {successful_runs}/{len(method_results)} ({successful_runs / len(method_results) * 100:.1f}%)"
        )
        logger.info(f"  Average duration: {avg_duration:.1f}s")

    return results


def analyze_results(results):
    """Analyze experiment results"""
    logger.info("\n" + "=" * 60)
    logger.info("DETAILED ANALYSIS")
    logger.info("=" * 60)

    for method_name, method_results in results.items():
        logger.info(f"\n{method_name} Detailed Results:")

        for result in method_results:
            status = "✓ SUCCESS" if result["success"] else "❌ FAILED"
            logger.info(
                f"  Seed {result['seed']}: {status} ({result['duration']:.1f}s)"
            )

    # Performance comparison
    logger.info("\nPerformance Comparison:")
    for method_name, method_results in results.items():
        successful_runs = [r for r in method_results if r["success"]]
        if successful_runs:
            avg_duration = sum(r["duration"] for r in successful_runs) / len(
                successful_runs
            )
            logger.info(f"  {method_name}: {avg_duration:.1f}s average")
        else:
            logger.info(f"  {method_name}: No successful runs")


def check_prerequisites():
    """Check if all prerequisites are met"""
    logger.info("Checking prerequisites...")

    # Check if config files exist
    config_files = [
        "configs/cifar100_qr_lora_gf.json",
        "configs/cifar100_neuro_lora.json",
        "configs/cifar100.json",
    ]

    for config_file in config_files:
        if not os.path.exists(config_file):
            logger.error(f"❌ Config file not found: {config_file}")
            return False
        else:
            logger.info(f"✓ Found config file: {config_file}")

    # Check if main.py exists
    if not os.path.exists("main.py"):
        logger.error("❌ main.py not found")
        return False
    else:
        logger.info("✓ Found main.py")

    # Check if data directory exists
    if not os.path.exists("data"):
        logger.warning(
            "⚠️  data/ directory not found. Make sure CIFAR-100 is downloaded."
        )
    else:
        logger.info("✓ Found data/ directory")

    logger.info("✓ All prerequisites checked")
    return True


def main():
    """Main function"""
    print("QR-LoRA-GF Experiment Runner")
    print("=" * 40)
    print("This script will run comparison experiments between:")
    print("1. QR-LoRA-GF (our new method)")
    print("2. Neuro-LoRA (biologically-inspired)")
    print("3. LoRA-Sub-DRS (original method)")
    print()

    # Check prerequisites
    if not check_prerequisites():
        logger.error("❌ Prerequisites not met. Please check the setup.")
        return

    # Ask for confirmation
    response = input("Do you want to proceed with the experiments? (y/n): ")
    if response.lower() != "y":
        logger.info("Experiments cancelled by user.")
        return

    # Run experiments
    try:
        results = run_comparison_experiments()
        analyze_results(results)

        logger.info("\n🎉 All experiments completed!")
        logger.info("Check the logs/ directory for detailed results.")

    except KeyboardInterrupt:
        logger.info("\nExperiments interrupted by user.")
    except Exception as e:
        logger.error(f"\nExperiments failed with error: {e}")


if __name__ == "__main__":
    main()
