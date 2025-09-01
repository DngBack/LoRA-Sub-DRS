#!/usr/bin/env python3
"""
Neuro-LoRA Usage Example

This example demonstrates how to use the Neuro-LoRA implementation
for continual learning experiments.
"""

import json
import torch
import logging
from utils.data_manager import DataManager
from methods.neuro_lora import NeuroLoRA

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Example usage of Neuro-LoRA"""

    # Load configuration
    with open("configs/cifar100_neuro_lora.json", "r") as f:
        config = json.load(f)

    logger.info("Loaded Neuro-LoRA configuration")

    # Initialize data manager
    data_manager = DataManager(
        dataset_name=config["dataset"],
        shuffle=True,
        seed=config["seed"],
        init_cls=config["init_cls"],
        increment=config["increment"],
    )

    logger.info(f"Initialized data manager for {config['dataset']}")

    # Create Neuro-LoRA instance
    neuro_lora = NeuroLoRA(config)

    logger.info("Created Neuro-LoRA instance")

    # Get number of tasks
    total_tasks = data_manager.total_tasks
    logger.info(f"Total tasks: {total_tasks}")

    # Training loop
    for task in range(total_tasks):
        logger.info(f"Starting task {task + 1}/{total_tasks}")

        # Incremental training
        neuro_lora.incremental_train(data_manager)

        # Evaluation
        results = neuro_lora.eval_task()

        logger.info(f"Task {task + 1} results: {results}")

        # Optional: Save checkpoint
        if task < total_tasks - 1:  # Don't save after last task
            neuro_lora.after_task()

    logger.info("Training completed!")

    # Final evaluation
    logger.info("Final evaluation results:")
    final_results = neuro_lora.eval_task()
    logger.info(f"Final accuracy: {final_results['top1']:.2f}%")


if __name__ == "__main__":
    print("Neuro-LoRA Example")
    print("==================")
    print("This example demonstrates Neuro-LoRA usage.")
    print("Make sure you have:")
    print("1. CIFAR-100 dataset downloaded")
    print("2. Required dependencies installed")
    print("3. GPU available (recommended)")
    print()

    try:
        main()
    except Exception as e:
        logger.error(f"Example failed: {e}")
        print(f"\nError: {e}")
        print("Please check your setup and try again.")
