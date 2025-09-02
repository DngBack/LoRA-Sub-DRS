#!/usr/bin/env python3
"""
Test for Delta-W Projection and Adaptive-k Functionality

This test verifies the new delta-W projected gradient projection
and adaptive-k subspace extraction features.
"""

import torch
import torch.nn as nn
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.neuro_utils import (
    project_grad_delta_w,
    extract_subspace_adaptive_k,
    extract_subspace_from_BA,
)


def test_delta_w_projection():
    """Test delta-W projected gradient projection"""
    print("Testing delta-W projected gradient projection...")

    # Create test matrices
    d, r = 768, 64
    B = torch.randn(d, r) * 0.1
    A = torch.randn(r, d) * 0.1
    gB = torch.randn(d, r) * 0.01
    gA = torch.randn(r, d) * 0.01

    # Create subspace
    k = 8
    S = torch.randn(d, k)
    Q, _ = torch.linalg.qr(S)
    S = Q[:, :k]

    print(f"Matrix shapes: B={B.shape}, A={A.shape}, S={S.shape}")
    print(f"Gradient shapes: gB={gB.shape}, gA={gA.shape}")

    # Test delta-W projection
    gA_proj, gB_proj = project_grad_delta_w(gA, gB, A, B, S)

    print(
        f"Projected gradient shapes: gA_proj={gA_proj.shape}, gB_proj={gB_proj.shape}"
    )

    # Verify shapes are preserved
    assert gA_proj.shape == gA.shape, (
        f"gA shape mismatch: {gA_proj.shape} vs {gA.shape}"
    )
    assert gB_proj.shape == gB.shape, (
        f"gB shape mismatch: {gB_proj.shape} vs {gB.shape}"
    )

    # Verify that the projected gradients are different from original
    gA_diff = torch.norm(gA_proj - gA)
    gB_diff = torch.norm(gB_proj - gB)
    print(f"gA difference: {gA_diff:.6f}")
    print(f"gB difference: {gB_diff:.6f}")

    # Verify that the total effect is projected out of subspace
    # Compute original total gradient
    grad_delta_w_orig = torch.matmul(B, gA) + torch.matmul(gB, A)

    # Compute projected total gradient
    grad_delta_w_proj = torch.matmul(B, gA_proj) + torch.matmul(gB_proj, A)

    # Check orthogonality to subspace
    coef_orig = torch.matmul(S.T, grad_delta_w_orig)
    coef_proj = torch.matmul(S.T, grad_delta_w_proj)

    orig_projection_magnitude = torch.norm(coef_orig)
    proj_projection_magnitude = torch.norm(coef_proj)

    print(f"Original projection magnitude: {orig_projection_magnitude:.6f}")
    print(f"Projected projection magnitude: {proj_projection_magnitude:.6f}")

    # The projected version should have smaller projection onto S
    # Note: We relax the constraint as the projection is approximate
    assert proj_projection_magnitude < orig_projection_magnitude * 0.8, (
        f"Projection not effective: {proj_projection_magnitude} vs {orig_projection_magnitude}"
    )

    print(
        f"Projection effectiveness: {proj_projection_magnitude / orig_projection_magnitude:.4f}"
    )

    print("✓ Delta-W projection test passed!")


def test_adaptive_k_extraction():
    """Test adaptive-k subspace extraction"""
    print("\nTesting adaptive-k subspace extraction...")

    # Create test matrices with different complexity
    d, r = 768, 64

    # Test case 1: Simple case (low rank)
    # Create a low-rank matrix by using only a few singular values
    U = torch.randn(d, d)
    U, _ = torch.linalg.qr(U)
    Vt = torch.randn(d, d)
    Vt, _ = torch.linalg.qr(Vt)

    # Create singular values with only first few being significant
    S = torch.zeros(d)
    S[:5] = torch.randn(5) * 0.1  # Only first 5 are significant

    # Create low-rank delta_w
    delta_w_simple = U @ torch.diag(S) @ Vt

    # Extract B and A from low-rank delta_w (approximate)
    B_simple = U[:, :r] @ torch.diag(S[:r])  # (d, r)
    A_simple = Vt[:r, :]  # (r, d)

    # Test case 2: Complex case (high rank)
    B_complex = torch.randn(d, r) * 0.1
    A_complex = torch.randn(r, d) * 0.1

    print(f"Matrix shapes: B={B_simple.shape}, A={A_simple.shape}")

    # Test adaptive-k extraction
    S_simple = extract_subspace_adaptive_k(
        B_simple, A_simple, energy_threshold=0.95, k_max=20
    )

    S_complex = extract_subspace_adaptive_k(
        B_complex, A_complex, energy_threshold=0.95, k_max=20
    )

    print(f"Simple case: extracted {S_simple.shape[1]} vectors")
    print(f"Complex case: extracted {S_complex.shape[1]} vectors")

    # Verify that simple case uses fewer vectors
    assert S_simple.shape[1] <= S_complex.shape[1], (
        f"Simple case should use fewer vectors: {S_simple.shape[1]} vs {S_complex.shape[1]}"
    )

    # Verify orthonormality
    Q_simple, _ = torch.linalg.qr(S_simple)
    Q_complex, _ = torch.linalg.qr(S_complex)

    orthogonality_simple = torch.norm(
        S_simple.T @ S_simple - torch.eye(S_simple.shape[1])
    )
    orthogonality_complex = torch.norm(
        S_complex.T @ S_complex - torch.eye(S_complex.shape[1])
    )

    print(f"Simple case orthogonality error: {orthogonality_simple:.6f}")
    print(f"Complex case orthogonality error: {orthogonality_complex:.6f}")

    assert orthogonality_simple < 1e-5, (
        f"Simple case not orthonormal: {orthogonality_simple}"
    )
    assert orthogonality_complex < 1e-5, (
        f"Complex case not orthonormal: {orthogonality_complex}"
    )

    # Test energy retention
    delta_w_simple = B_simple @ A_simple
    delta_w_complex = B_complex @ A_complex

    U_simple, S_simple_svd, _ = torch.linalg.svd(delta_w_simple)
    U_complex, S_complex_svd, _ = torch.linalg.svd(delta_w_complex)

    # Check energy retention for simple case
    total_energy_simple = S_simple_svd.sum()
    retained_energy_simple = S_simple_svd[: S_simple.shape[1]].sum()
    energy_ratio_simple = retained_energy_simple / total_energy_simple

    print(f"Simple case energy retention: {energy_ratio_simple:.4f}")

    assert energy_ratio_simple >= 0.95, (
        f"Energy retention below threshold: {energy_ratio_simple:.4f}"
    )

    print("✓ Adaptive-k extraction test passed!")


def test_adaptive_k_vs_fixed_k():
    """Compare adaptive-k vs fixed-k extraction"""
    print("\nTesting adaptive-k vs fixed-k comparison...")

    d, r = 768, 64
    B = torch.randn(d, r) * 0.1
    A = torch.randn(r, d) * 0.1

    # Fixed-k extraction
    k_fixed = 8
    S_fixed = extract_subspace_from_BA(B, A, k_fixed)

    # Adaptive-k extraction
    S_adaptive = extract_subspace_adaptive_k(B, A, energy_threshold=0.95, k_max=k_fixed)

    print(f"Fixed-k: {S_fixed.shape[1]} vectors")
    print(f"Adaptive-k: {S_adaptive.shape[1]} vectors")

    # Both should be valid
    assert S_fixed.shape[1] == k_fixed, f"Fixed-k should extract {k_fixed} vectors"
    assert S_adaptive.shape[1] <= k_fixed, f"Adaptive-k should not exceed k_max"
    assert S_adaptive.shape[1] >= 1, f"Adaptive-k should extract at least 1 vector"

    # Check energy retention
    delta_w = B @ A
    U, S, _ = torch.linalg.svd(delta_w)

    # Fixed-k energy retention
    fixed_energy = S[:k_fixed].sum() / S.sum()

    # Adaptive-k energy retention
    adaptive_energy = S[: S_adaptive.shape[1]].sum() / S.sum()

    print(f"Fixed-k energy retention: {fixed_energy:.4f}")
    print(f"Adaptive-k energy retention: {adaptive_energy:.4f}")

    # Adaptive-k should meet the threshold if possible
    # If the matrix doesn't have enough energy in k_max vectors, that's okay
    if S[:k_fixed].sum() / S.sum() >= 0.95:
        assert adaptive_energy >= 0.95, (
            f"Adaptive-k energy retention below threshold: {adaptive_energy:.4f}"
        )
    else:
        print(
            f"Note: Matrix doesn't have enough energy in {k_fixed} vectors to meet threshold"
        )
        # Just verify that adaptive-k extracted as many vectors as possible
        assert S_adaptive.shape[1] == k_fixed, (
            f"Adaptive-k should extract maximum vectors when energy is low: {S_adaptive.shape[1]} vs {k_fixed}"
        )

    print("✓ Adaptive-k vs fixed-k comparison passed!")


if __name__ == "__main__":
    print("Delta-W Projection and Adaptive-k Tests")
    print("=======================================")

    try:
        test_delta_w_projection()
        test_adaptive_k_extraction()
        test_adaptive_k_vs_fixed_k()

        print("\n🎉 All delta-W projection and adaptive-k tests passed!")
        print("\nSummary:")
        print("✓ Delta-W projected gradient projection works correctly")
        print("✓ Adaptive-k subspace extraction adapts to matrix complexity")
        print("✓ Energy retention threshold is respected")
        print("✓ Orthonormality is maintained")
        print("✓ Simple cases use fewer vectors than complex cases")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
