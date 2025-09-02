#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.neuro_utils import (
    extract_subspace_from_BA, merge_cumulative_subspace, 
    project_grad_B, compute_plasticity_loss
)


def test_extract_subspace():
    """Test subspace extraction from LoRA matrices"""
    print("Testing extract_subspace_from_BA...")
    
    d, r = 64, 8
    B = torch.randn(d, r)
    A = torch.randn(r, d)
    
    S_new = extract_subspace_from_BA(B, A, k=4)
    assert S_new.shape == (d, 4), f"Expected shape (64, 4), got {S_new.shape}"
    
    # Check orthonormality
    S_norms = S_new.norm(dim=0)
    assert torch.allclose(S_norms, torch.ones_like(S_norms), atol=1e-6), "Columns should be normalized"
    
    # Check orthogonality
    S_ortho = torch.mm(S_new.T, S_new)
    expected = torch.eye(4)
    assert torch.allclose(S_ortho, expected, atol=1e-6), "Columns should be orthogonal"
    
    print("✓ extract_subspace_from_BA passed")


def test_merge_subspace():
    """Test merging subspaces"""
    print("Testing merge_cumulative_subspace...")
    
    d = 64
    S_prev = torch.randn(d, 8)
    S_new = torch.randn(d, 4)
    
    # Test merging
    S_cum = merge_cumulative_subspace(S_prev, S_new, K_max=10)
    assert S_cum.shape[0] == d, f"Expected first dimension {d}, got {S_cum.shape[0]}"
    assert S_cum.shape[1] <= 10, f"Expected max 10 columns, got {S_cum.shape[1]}"
    
    # Test with None previous subspace
    S_cum_none = merge_cumulative_subspace(None, S_new, K_max=10)
    assert S_cum_none.shape == (d, 4), f"Expected shape (64, 4), got {S_cum_none.shape}"
    
    # Test truncation
    S_cum_trunc = merge_cumulative_subspace(S_prev, S_new, K_max=5)
    assert S_cum_trunc.shape[1] == 5, f"Expected 5 columns, got {S_cum_trunc.shape[1]}"
    
    print("✓ merge_cumulative_subspace passed")


def test_gradient_projection():
    """Test gradient projection"""
    print("Testing project_grad_B...")
    
    d, r, K = 64, 8, 16
    gB = torch.randn(d, r)
    S = torch.randn(d, K)
    
    # Test projection
    gB_proj = project_grad_B(gB, S)
    assert gB_proj.shape == gB.shape, f"Expected shape {gB.shape}, got {gB_proj.shape}"
    
    # Test projection with None subspace
    gB_proj_none = project_grad_B(gB, None)
    assert torch.allclose(gB_proj_none, gB), "Should return original gradient when S is None"
    
    print("✓ project_grad_B passed")


def test_plasticity_loss():
    """Test plasticity loss computation"""
    print("Testing compute_plasticity_loss...")
    
    B, m = 16, 32
    h = torch.abs(torch.randn(B, m))  # Non-negative activations
    
    loss = compute_plasticity_loss(h)
    assert isinstance(loss, torch.Tensor), "Should return a tensor"
    assert loss.numel() == 1, "Should return a scalar"
    assert loss.item() > 0, "Loss should be positive"
    
    # Test with uniform activations (should have high entropy)
    h_uniform = torch.ones(B, m)
    loss_uniform = compute_plasticity_loss(h_uniform)
    
    # Test with sparse activations (should have lower entropy)
    h_sparse = torch.zeros(B, m)
    h_sparse[:, 0] = 1.0
    loss_sparse = compute_plasticity_loss(h_sparse)
    
    assert loss_uniform.item() > loss_sparse.item(), "Uniform should have higher entropy than sparse"
    
    print("✓ compute_plasticity_loss passed")


def test_integration():
    """Test integration of all components"""
    print("Testing integration...")
    
    # Simulate a complete workflow
    d, r = 64, 8
    B = torch.randn(d, r)
    A = torch.randn(r, d)
    
    # 1. Extract subspace
    S_new = extract_subspace_from_BA(B, A, k=4)
    
    # 2. Merge with previous
    S_prev = torch.randn(d, 6)
    S_cum = merge_cumulative_subspace(S_prev, S_new, K_max=8)
    
    # 3. Project gradient
    gB = torch.randn(d, r)
    gB_proj = project_grad_B(gB, S_cum)
    
    # 4. Compute plasticity loss
    h = torch.abs(torch.randn(16, 32))
    plast_loss = compute_plasticity_loss(h)
    
    # All should work without errors
    assert S_cum.shape[1] <= 8, "Subspace should respect K_max"
    assert gB_proj.shape == gB.shape, "Projected gradient should maintain shape"
    assert plast_loss.item() > 0, "Plasticity loss should be positive"
    
    print("✓ Integration test passed")


def smoke():
    """Run all smoke tests"""
    print("Running Neuro-LoRA smoke tests...")
    print("=" * 50)
    
    try:
        test_extract_subspace()
        test_merge_subspace()
        test_gradient_projection()
        test_plasticity_loss()
        test_integration()
        
        print("=" * 50)
        print("🎉 All smoke tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = smoke()
    sys.exit(0 if success else 1)
