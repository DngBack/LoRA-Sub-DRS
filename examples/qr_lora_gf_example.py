#!/usr/bin/env python3
"""
QR-LoRA-GF Usage Example

This example demonstrates how to use the QR-LoRA Subtraction with Gated Fusion method
for continual learning experiments.
"""

import json
import torch
import logging
import numpy as np
from utils.data_manager import DataManager
from methods.qr_lora_gf import QRLoRA_GF

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Example usage of QR-LoRA-GF"""

    # Load configuration
    with open("configs/cifar100_qr_lora_gf.json", "r") as f:
        config = json.load(f)

    logger.info("Loaded QR-LoRA-GF configuration")

    # Initialize data manager
    data_manager = DataManager(
        dataset_name=config["dataset"],
        shuffle=True,
        seed=config["seed"][0],  # Use first seed for example
        init_cls=config["init_cls"],
        increment=config["increment"],
    )

    logger.info(f"Initialized data manager for {config['dataset']}")

    # Create QR-LoRA-GF instance
    qr_lora_gf = QRLoRA_GF(config)

    logger.info("Created QR-LoRA-GF instance")

    # Get number of tasks
    total_tasks = data_manager.total_tasks
    logger.info(f"Total tasks: {total_tasks}")

    # Training loop
    for task in range(min(total_tasks, 3)):  # Run first 3 tasks for example
        logger.info(f"Starting task {task + 1}/{total_tasks}")

        # Incremental training
        qr_lora_gf.incremental_train(data_manager)

        # Evaluation
        results = qr_lora_gf.eval_task()

        logger.info(f"Task {task + 1} results: {results}")

        # Optional: Save checkpoint
        if task < total_tasks - 1:  # Don't save after last task
            qr_lora_gf.after_task()

    logger.info("Training completed!")

    # Final evaluation
    logger.info("Final evaluation results:")
    final_results = qr_lora_gf.eval_task()
    logger.info(f"Final accuracy: {final_results['top1']:.2f}%")

    # Print QR-LoRA-GF specific information
    logger.info("\nQR-LoRA-GF Method Information:")
    logger.info(f"✓ QR decomposition with pivoting: {qr_lora_gf.use_pivoting}")
    logger.info(f"✓ Gated fusion enabled: {qr_lora_gf.use_gated_fusion}")
    logger.info(f"✓ Learnable subtraction strength: {qr_lora_gf.learnable_subtraction}")
    logger.info(f"✓ Energy threshold: {qr_lora_gf.energy_threshold}")
    logger.info(f"✓ Fusion strength: {qr_lora_gf.fusion_strength}")
    logger.info(
        f"✓ Gate regularization weight: {qr_lora_gf.gate_regularization_weight}"
    )


def compare_methods():
    """Compare QR-LoRA-GF with other methods"""
    logger.info("\nComparing QR-LoRA-GF with other methods...")

    # Load configurations
    with open("configs/cifar100_qr_lora_gf.json", "r") as f:
        qr_config = json.load(f)

    with open("configs/cifar100_neuro_lora.json", "r") as f:
        neuro_config = json.load(f)

    # Compare key parameters
    logger.info("Parameter Comparison:")
    logger.info(f"QR-LoRA-GF k_per_task: {qr_config['qr_lora_gf']['k_per_task']}")
    logger.info(f"Neuro-LoRA k_per_task: {neuro_config['neuro_lora']['k_per_task']}")

    logger.info(f"QR-LoRA-GF K_max: {qr_config['qr_lora_gf']['K_max']}")
    logger.info(f"Neuro-LoRA K_max: {neuro_config['neuro_lora']['K_max']}")

    # Key differences
    logger.info("\nKey Differences:")
    logger.info("✓ QR-LoRA-GF uses QR decomposition instead of SVD")
    logger.info("✓ QR-LoRA-GF includes gated fusion mechanism")
    logger.info("✓ QR-LoRA-GF has learnable subtraction strength")
    logger.info("✓ QR-LoRA-GF includes gate regularization")


def demonstrate_qr_advantages():
    """Demonstrate advantages of QR decomposition"""
    logger.info("\nDemonstrating QR decomposition advantages...")

    # Create test matrices
    d, r = 768, 64
    B = torch.randn(d, r) * 0.1
    A = torch.randn(r, d) * 0.1

    # Test QR decomposition
    from utils.qr_lora_utils import extract_subspace_qr_from_BA

    S_qr, imp_qr = extract_subspace_qr_from_BA(B, A, k=8, use_pivoting=True)

    # Check orthonormality
    S_norms = S_qr.norm(dim=0)
    orthogonality_error = torch.norm(torch.mm(S_qr.T, S_qr) - torch.eye(8))

    logger.info(f"QR orthonormality error: {orthogonality_error:.2e}")
    logger.info(f"QR importance scores shape: {imp_qr.shape}")
    logger.info(f"QR importance scores range: [{imp_qr.min():.3f}, {imp_qr.max():.3f}]")

    # Demonstrate gated fusion
    from utils.qr_lora_utils import gated_fusion_subspaces

    S_old = torch.randn(d, 16)
    S_new = torch.randn(d, 8)

    # Orthonormalize
    Q_old, _ = torch.linalg.qr(S_old)
    Q_new, _ = torch.linalg.qr(S_new)
    S_old = Q_old[:, :16]
    S_new = Q_new[:, :8]

    S_fused, gate_weights = gated_fusion_subspaces(S_old, S_new, fusion_strength=0.5)

    logger.info(f"Gated fusion gate weights: {gate_weights}")
    logger.info(f"Average gate weight: {gate_weights.mean():.3f}")
    logger.info(f"Gate weight std: {gate_weights.std():.3f}")


if __name__ == "__main__":
    print("QR-LoRA-GF Example")
    print("==================")
    print("This example demonstrates QR-LoRA-GF usage.")
    print("Make sure you have:")
    print("1. CIFAR-100 dataset downloaded")
    print("2. Required dependencies installed")
    print("3. GPU available (recommended)")
    print()

    try:
        main()
        compare_methods()
        demonstrate_qr_advantages()

        print("\n🎉 QR-LoRA-GF example completed successfully!")

    except Exception as e:
        logger.error(f"Example failed: {e}")
        print(f"\nError: {e}")
        print("Please check your setup and try again.")
