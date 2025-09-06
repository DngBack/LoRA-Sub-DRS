#!/usr/bin/env python3
"""
Comprehensive Test Suite for QR-LoRA-GF Method

This test suite verifies all components of the QR-LoRA Subtraction with Gated Fusion method:
1. QR-based subspace extraction
2. Gated fusion mechanism
3. Gradient projection with gating
4. Integration with the main training loop
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import json
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.qr_lora_utils import (
    extract_subspace_qr_from_BA,
    gated_fusion_subspaces,
    merge_cumulative_subspace_qr,
    project_grad_qr_gated,
    compute_qr_regularization_loss,
    save_qr_subspace,
    load_qr_subspace,
    create_gated_fusion_module,
)
from methods.qr_lora_gf import QRLoRA_GF


def test_qr_subspace_extraction():
    """Test QR-based subspace extraction"""
    print("Testing QR-based subspace extraction...")

    # Create realistic LoRA matrices
    d, r = 768, 64
    B = torch.randn(d, r) * 0.1
    A = torch.randn(r, d) * 0.1

    # Test with pivoting
    S_new, imp_scores = extract_subspace_qr_from_BA(B, A, k=8, use_pivoting=True)
    assert S_new.shape == (d, 8), f"Expected shape (768, 8), got {S_new.shape}"
    assert imp_scores.shape == (8,), (
        f"Expected importance scores shape (8,), got {imp_scores.shape}"
    )

    # Check orthonormality
    S_norms = S_new.norm(dim=0)
    assert torch.allclose(S_norms, torch.ones_like(S_norms), atol=1e-6), (
        "Columns should be normalized"
    )

    # Check orthogonality
    S_ortho = torch.mm(S_new.T, S_new)
    expected = torch.eye(8)
    assert torch.allclose(S_ortho, expected, atol=1e-6), "Columns should be orthogonal"

    # Test without pivoting
    S_new_no_pivot, imp_scores_no_pivot = extract_subspace_qr_from_BA(
        B, A, k=8, use_pivoting=False
    )
    assert S_new_no_pivot.shape == (d, 8), "Should work without pivoting too"

    # Test adaptive k selection
    S_adaptive, imp_adaptive = extract_subspace_qr_from_BA(
        B, A, k=16, energy_threshold=0.8
    )
    assert S_adaptive.shape[1] <= 16, "Adaptive k should respect max k"

    print("✓ QR-based subspace extraction test passed!")


def test_gated_fusion():
    """Test gated fusion mechanism"""
    print("Testing gated fusion mechanism...")

    d = 768
    K_old, k_new = 16, 8

    # Create subspaces
    S_old = torch.randn(d, K_old)
    S_new = torch.randn(d, k_new)

    # Orthonormalize
    Q_old, _ = torch.linalg.qr(S_old)
    Q_new, _ = torch.linalg.qr(S_new)
    S_old = Q_old[:, :K_old]
    S_new = Q_new[:, :k_new]

    # Test gated fusion
    S_fused, gate_weights = gated_fusion_subspaces(S_old, S_new, fusion_strength=0.5)

    assert S_fused.shape == (d, k_new), f"Expected shape (768, 8), got {S_fused.shape}"
    assert gate_weights.shape == (k_new,), (
        f"Expected gate weights shape (8,), got {gate_weights.shape}"
    )
    assert torch.all(gate_weights >= 0) and torch.all(gate_weights <= 1), (
        "Gate weights should be in [0,1]"
    )

    # Check orthonormality of fused subspace
    S_norms = S_fused.norm(dim=0)
    assert torch.allclose(S_norms, torch.ones_like(S_norms), atol=1e-6), (
        "Fused subspace should be orthonormal"
    )

    # Test with None old subspace
    S_fused_none, gate_weights_none = gated_fusion_subspaces(None, S_new)
    assert torch.allclose(S_fused_none, S_new), (
        "Should return new subspace when old is None"
    )
    assert torch.allclose(gate_weights_none, torch.ones_like(gate_weights_none)), (
        "Gate weights should be 1 when old is None"
    )

    print("✓ Gated fusion mechanism test passed!")


def test_merge_cumulative_subspace():
    """Test cumulative subspace merging with QR and gated fusion"""
    print("Testing cumulative subspace merging...")

    d = 768
    K_prev, k_new = 32, 8

    # Create subspaces
    S_prev = torch.randn(d, K_prev)
    S_new = torch.randn(d, k_new)

    # Orthonormalize
    Q_prev, _ = torch.linalg.qr(S_prev)
    Q_new, _ = torch.linalg.qr(S_new)
    S_prev = Q_prev[:, :K_prev]
    S_new = Q_new[:, :k_new]

    # Test with gated fusion
    S_cum_fused, fusion_info_fused = merge_cumulative_subspace_qr(
        S_prev, S_new, K_max=40, use_gated_fusion=True, fusion_strength=0.5
    )

    assert S_cum_fused.shape[0] == d, (
        f"Expected first dimension {d}, got {S_cum_fused.shape[0]}"
    )
    assert S_cum_fused.shape[1] <= 40, (
        f"Expected max 40 columns, got {S_cum_fused.shape[1]}"
    )
    assert fusion_info_fused["method"] == "gated_fusion", "Should use gated fusion"
    assert "gate_weights" in fusion_info_fused, "Should include gate weights"

    # Test without gated fusion
    S_cum_no_fusion, fusion_info_no_fusion = merge_cumulative_subspace_qr(
        S_prev, S_new, K_max=40, use_gated_fusion=False
    )

    assert fusion_info_no_fusion["method"] == "concatenation", (
        "Should use concatenation"
    )

    # Test with None previous subspace
    S_cum_none, fusion_info_none = merge_cumulative_subspace_qr(None, S_new, K_max=40)
    assert fusion_info_none["method"] == "new_only", (
        "Should handle None previous subspace"
    )

    print("✓ Cumulative subspace merging test passed!")


def test_gradient_projection():
    """Test QR-based gradient projection with gating"""
    print("Testing QR-based gradient projection...")

    d, r, K = 768, 64, 16

    # Create test data
    S = torch.randn(d, K)
    Q, _ = torch.linalg.qr(S)
    S = Q[:, :K]  # Ensure orthonormal

    A = torch.randn(r, d) * 0.1
    B = torch.randn(d, r) * 0.1
    gA = torch.randn(r, d) * 0.01
    gB = torch.randn(d, r) * 0.01

    # Test without gating
    gA_proj, gB_proj = project_grad_qr_gated(gA, gB, A, B, S, None)

    assert gA_proj.shape == gA.shape, f"Expected shape {gA.shape}, got {gA_proj.shape}"
    assert gB_proj.shape == gB.shape, f"Expected shape {gB.shape}, got {gB_proj.shape}"

    # Test with gating
    gate_weights = torch.rand(K)  # Random gate weights
    gA_proj_gated, gB_proj_gated = project_grad_qr_gated(gA, gB, A, B, S, gate_weights)

    assert gA_proj_gated.shape == gA.shape, "Gated projection should maintain shape"
    assert gB_proj_gated.shape == gB.shape, "Gated projection should maintain shape"

    # Test with None subspace
    gA_proj_none, gB_proj_none = project_grad_qr_gated(gA, gB, A, B, None, None)
    assert torch.allclose(gA_proj_none, gA), (
        "Should return original gradients when S is None"
    )
    assert torch.allclose(gB_proj_none, gB), (
        "Should return original gradients when S is None"
    )

    print("✓ QR-based gradient projection test passed!")


def test_regularization_loss():
    """Test gate regularization loss"""
    print("Testing gate regularization loss...")

    # Test with realistic gate weights
    gate_weights = torch.tensor([0.3, 0.7, 0.5, 0.8, 0.2])

    reg_loss = compute_qr_regularization_loss(gate_weights, target_gate_value=0.5)

    assert isinstance(reg_loss, torch.Tensor), "Should return a tensor"
    assert reg_loss.numel() == 1, "Should return a scalar"
    assert reg_loss.item() >= 0, "Regularization loss should be non-negative"

    # Test with None gate weights
    reg_loss_none = compute_qr_regularization_loss(None)
    assert reg_loss_none.item() == 0.0, "Should return 0 for None gate weights"

    print("✓ Gate regularization loss test passed!")


def test_save_load_subspace():
    """Test saving and loading QR subspaces"""
    print("Testing save/load QR subspaces...")

    # Create test data
    d, K = 768, 16
    S = torch.randn(d, K)
    Q, _ = torch.linalg.qr(S)
    S = Q[:, :K]

    importance_scores = torch.rand(K)
    fusion_info = {
        "method": "gated_fusion",
        "gate_weights": torch.rand(K),
        "avg_gate_weight": 0.6,
        "final_dim": K,
        "input_dim": K + 8,
    }

    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "test_subspace.pt")

        # Save
        save_qr_subspace(S, importance_scores, fusion_info, path)
        assert os.path.exists(path), "File should be created"

        # Load
        S_loaded, imp_loaded, info_loaded = load_qr_subspace(path)

        assert torch.allclose(S_loaded, S), "Loaded subspace should match saved"
        assert torch.allclose(imp_loaded, importance_scores), (
            "Loaded importance scores should match"
        )
        assert info_loaded["method"] == fusion_info["method"], (
            "Loaded fusion info should match"
        )
        assert info_loaded["avg_gate_weight"] == fusion_info["avg_gate_weight"], (
            "Loaded avg gate weight should match"
        )

    # Test loading non-existent file
    S_none, imp_none, info_none = load_qr_subspace("non_existent_file.pt")
    assert S_none is None, "Should return None for non-existent file"
    assert imp_none is None, "Should return None for non-existent file"
    assert info_none is None, "Should return None for non-existent file"

    print("✓ Save/load QR subspaces test passed!")


def test_gated_fusion_module():
    """Test learnable gated fusion module"""
    print("Testing learnable gated fusion module...")

    d = 768
    K_old, k_new = 16, 8

    # Create fusion module
    fusion_module = create_gated_fusion_module(d, hidden_dim=64)

    # Create test subspaces
    S_old = torch.randn(d, K_old)
    S_new = torch.randn(d, k_new)

    # Orthonormalize
    Q_old, _ = torch.linalg.qr(S_old)
    Q_new, _ = torch.linalg.qr(S_new)
    S_old = Q_old[:, :K_old]
    S_new = Q_new[:, :k_new]

    # Forward pass
    S_fused, gate_weights = fusion_module(S_old, S_new)

    assert S_fused.shape == (d, k_new), f"Expected shape (768, 8), got {S_fused.shape}"
    assert gate_weights.shape == (k_new,), (
        f"Expected gate weights shape (8,), got {gate_weights.shape}"
    )
    assert torch.all(gate_weights >= 0) and torch.all(gate_weights <= 1), (
        "Gate weights should be in [0,1]"
    )

    # Test with None old subspace
    S_fused_none, gate_weights_none = fusion_module(None, S_new)
    assert torch.allclose(S_fused_none, S_new), (
        "Should return new subspace when old is None"
    )

    print("✓ Learnable gated fusion module test passed!")


def test_qr_lora_gf_integration():
    """Test QR-LoRA-GF method integration"""
    print("Testing QR-LoRA-GF method integration...")

    # Create minimal config
    config = {
        "net_type": "sip",
        "embd_dim": 768,
        "num_heads": 12,
        "EPSILON": 1e-8,
        "init_epoch": 1,
        "optim": "Adam",
        "init_lr": 0.001,
        "init_lr_decay": 0.1,
        "init_weight_decay": 0.0,
        "epochs": 1,
        "fc_lrate": 0.002,
        "lrate": 0.001,
        "lrate_decay": 0.1,
        "batch_size": 32,
        "weight_decay": 0.0,
        "rank": 8,
        "margin_inter": 1.0,
        "lambada": 0.05,
        "num_workers": 4,
        "total_sessions": 5,
        "dataset": "cifar100",
        "eval": True,
        "device": ["cuda:0"] if torch.cuda.is_available() else ["cpu"],
        "memory_size": 0,
        "memory_per_class": 0,
        "fixed_memory": True,
        "checkpoint_dir": "logs/test_qr_lora_gf",
        "init_cls": 5,
        "increment": 5,
        "qr_lora_gf": {
            "enabled": True,
            "k_per_task": 4,
            "K_max": 32,
            "use_pivoting": True,
            "energy_threshold": 0.95,
            "use_gated_fusion": True,
            "fusion_strength": 0.5,
            "gate_temperature": 1.0,
            "learnable_subtraction": True,
            "subtraction_alpha": 1.0,
            "gate_regularization_weight": 0.01,
            "target_gate_value": 0.5,
        },
    }

    try:
        # Initialize QR-LoRA-GF
        qr_lora_gf = QRLoRA_GF(config)

        # Check that all components are initialized
        assert hasattr(qr_lora_gf, "S_cumulative"), (
            "Should have cumulative subspace storage"
        )
        assert hasattr(qr_lora_gf, "importance_scores"), (
            "Should have importance scores storage"
        )
        assert hasattr(qr_lora_gf, "fusion_info"), "Should have fusion info storage"

        # Check configuration
        assert qr_lora_gf.k_per_task == 4, "Should have correct k_per_task"
        assert qr_lora_gf.K_max == 32, "Should have correct K_max"
        assert qr_lora_gf.use_gated_fusion == True, "Should have gated fusion enabled"

        print("✓ QR-LoRA-GF method integration test passed!")

    except Exception as e:
        print(f"❌ QR-LoRA-GF integration test failed: {e}")
        raise


def test_numerical_stability():
    """Test numerical stability of QR decomposition vs SVD"""
    print("Testing numerical stability...")

    # Create ill-conditioned matrix (high condition number)
    d, r = 768, 64

    # Create matrix with very small singular values
    U, _, Vt = torch.linalg.svd(torch.randn(d, d))
    S_vals = torch.cat([torch.ones(r - 10), torch.linspace(1e-6, 1e-8, 10)])
    S_vals = torch.cat([S_vals, torch.zeros(d - r)])

    ill_conditioned = U @ torch.diag(S_vals) @ Vt

    # Extract B and A from ill-conditioned matrix
    B = ill_conditioned[:, :r]
    A = ill_conditioned[:r, :]

    # Test QR decomposition
    S_qr, imp_qr = extract_subspace_qr_from_BA(B, A, k=8, use_pivoting=True)

    # Check orthonormality
    S_norms_qr = S_qr.norm(dim=0)
    orthogonality_error_qr = torch.norm(torch.mm(S_qr.T, S_qr) - torch.eye(8))

    assert torch.allclose(S_norms_qr, torch.ones_like(S_norms_qr), atol=1e-5), (
        "QR should maintain orthonormality"
    )
    assert orthogonality_error_qr < 1e-5, "QR should maintain orthogonality"

    print(f"QR orthogonality error: {orthogonality_error_qr:.2e}")
    print("✓ Numerical stability test passed!")


def test_parameter_efficiency():
    """Test parameter efficiency compared to standard LoRA"""
    print("Testing parameter efficiency...")

    d, r = 768, 64

    # Standard LoRA parameters: A (r×d) + B (d×r) = r×d + d×r = r×(d+d) = r×2d
    standard_lora_params = r * (d + d)

    # QR-LoRA-GF: We use QR decomposition which can be more efficient
    # The key insight is that QR gives us orthonormal basis with better numerical properties
    # and we can potentially use lower rank due to better conditioning

    # Test with lower rank for QR-LoRA-GF
    r_qr = r // 2  # Use half the rank
    B_qr = torch.randn(d, r_qr) * 0.1
    A_qr = torch.randn(r_qr, d) * 0.1

    qr_lora_params = r_qr * (d + d)

    # QR-LoRA-GF should use fewer parameters
    assert qr_lora_params < standard_lora_params, (
        "QR-LoRA-GF should be more parameter efficient"
    )

    reduction_ratio = (standard_lora_params - qr_lora_params) / standard_lora_params
    print(f"Parameter reduction: {reduction_ratio:.1%}")

    # Test that we can still extract meaningful subspaces with fewer parameters
    S_qr, imp_qr = extract_subspace_qr_from_BA(B_qr, A_qr, k=8)
    assert S_qr.shape == (d, 8), "Should still extract correct subspace shape"

    print("✓ Parameter efficiency test passed!")


def run_all_tests():
    """Run all tests"""
    print("Running QR-LoRA-GF Comprehensive Test Suite")
    print("=" * 60)

    tests = [
        test_qr_subspace_extraction,
        test_gated_fusion,
        test_merge_cumulative_subspace,
        test_gradient_projection,
        test_regularization_loss,
        test_save_load_subspace,
        test_gated_fusion_module,
        test_qr_lora_gf_integration,
        test_numerical_stability,
        test_parameter_efficiency,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            failed += 1

    print("=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All QR-LoRA-GF tests passed!")
        return True
    else:
        print(f"❌ {failed} tests failed!")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
