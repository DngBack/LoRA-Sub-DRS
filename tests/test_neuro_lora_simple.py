#!/usr/bin/env python3
"""
Simple test for Neuro-LoRA core utilities
Tests the mathematical components without requiring full model dependencies.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import tempfile
import shutil

# Add parent directory to path for imports
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.neuro_utils import (
    extract_subspace_from_BA,
    merge_cumulative_subspace,
    project_grad_B,
    compute_plasticity_loss,
    save_subspace,
    load_subspace,
)


def test_subspace_operations():
    """Test subspace operations with realistic data"""
    print("Testing subspace operations...")

    # Create realistic LoRA matrices
    dim = 768
    r = 64
    k_per_task = 8

    # Simulate trained LoRA parameters
    # A should be (r, dim) and B should be (dim, r) for LoRA
    A = torch.randn(r, dim) * 0.1
    B = torch.randn(dim, r) * 0.1

    # Test subspace extraction
    S_new = extract_subspace_from_BA(B, A, k_per_task)
    assert S_new.shape == (dim, k_per_task), (
        f"Expected shape ({dim}, {k_per_task}), got {S_new.shape}"
    )

    # Test orthogonality
    S_ortho = torch.mm(S_new.T, S_new)
    identity = torch.eye(k_per_task)
    ortho_error = torch.norm(S_ortho - identity)
    assert ortho_error < 1e-4, f"Subspace should be orthonormal, error: {ortho_error}"

    # Test cumulative subspace merging
    S_prev = torch.randn(dim, k_per_task)
    S_prev = torch.mm(S_prev, S_prev.T)  # Make it symmetric
    S_prev = torch.mm(S_prev, torch.randn(dim, k_per_task))  # Make it full rank

    S_merged = merge_cumulative_subspace(S_prev, S_new, K_max=16)
    assert S_merged.shape[1] <= 16, f"Expected max 16 columns, got {S_merged.shape[1]}"

    # Test orthogonality of merged subspace
    S_ortho_merged = torch.mm(S_merged.T, S_merged)
    identity_merged = torch.eye(S_merged.shape[1])
    ortho_error_merged = torch.norm(S_ortho_merged - identity_merged)
    assert ortho_error_merged < 1e-4, (
        f"Merged subspace should be orthonormal, error: {ortho_error_merged}"
    )

    print("✓ Subspace operations successful")


def test_gradient_projection():
    """Test gradient projection with realistic data"""
    print("Testing gradient projection...")

    # Create realistic data
    dim = 768
    k = 8

    # Create subspace
    S = torch.randn(dim, k)
    S = torch.mm(S, S.T)  # Make it symmetric
    S = torch.mm(S, torch.randn(dim, k))  # Make it full rank
    Q, _ = torch.linalg.qr(S)
    S = Q[:, :k]  # Ensure orthonormal

    # Create gradient
    gB = torch.randn(dim, dim)

    # Project gradient
    gB_proj = project_grad_B(gB, S)

    # Check that projected gradient is orthogonal to subspace
    projection_check = torch.mm(S.T, gB_proj)
    projection_error = torch.norm(projection_check)
    assert projection_error < 1e-4, (
        f"Projected gradient should be orthogonal to subspace, error: {projection_error}"
    )

    print("✓ Gradient projection successful")


def test_plasticity_loss():
    """Test plasticity loss computation"""
    print("Testing plasticity loss...")

    # Create mock LoRA activations
    batch_size = 32
    r = 64

    # Test with random activations (make them non-negative)
    activations = torch.abs(torch.randn(batch_size, r))
    loss = compute_plasticity_loss(activations)

    print(f"Random activations loss: {loss}")
    assert loss > 0, "Plasticity loss should be positive"
    assert torch.isfinite(loss), "Plasticity loss should be finite"

    # Test with uniform activations (should give high entropy)
    uniform_activations = (
        torch.ones(batch_size, r) + 0.1
    )  # Add small offset to avoid log(0)
    uniform_loss = compute_plasticity_loss(uniform_activations)

    # Test with sparse activations (should give lower entropy)
    sparse_activations = torch.zeros(batch_size, r) + 0.1  # Add small offset
    sparse_activations[:, 0] = 1.0  # Only first neuron active
    sparse_loss = compute_plasticity_loss(sparse_activations)

    # Uniform should have higher loss (more diverse) than sparse
    print(f"Uniform loss: {uniform_loss}, Sparse loss: {sparse_loss}")
    assert uniform_loss > sparse_loss, (
        "Uniform activations should have higher plasticity loss"
    )

    print("✓ Plasticity loss computation successful")


def test_subspace_save_load():
    """Test subspace saving and loading"""
    print("Testing subspace save/load...")

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()

    try:
        # Create test subspace
        dim = 768
        k = 8
        S = torch.randn(dim, k)
        Q, _ = torch.linalg.qr(S)
        S = Q[:, :k]  # Ensure orthonormal

        # Save subspace
        save_path = os.path.join(temp_dir, "test_subspace.pt")
        save_subspace(S, save_path)

        # Check file exists
        assert os.path.exists(save_path), "Subspace file should be created"

        # Load subspace
        S_loaded = load_subspace(save_path, device="cpu")

        # Check if loaded correctly
        assert S_loaded is not None, "Subspace should be loaded"
        assert S_loaded.shape == S.shape, (
            f"Shape mismatch: {S_loaded.shape} vs {S.shape}"
        )

        # Check values are close
        diff = torch.norm(S - S_loaded)
        assert diff < 1e-4, f"Loaded subspace should match original, diff: {diff}"

        print("✓ Subspace save/load successful")

    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_mock_lora_module():
    """Test with a simple mock LoRA module"""
    print("Testing with mock LoRA module...")

    class MockLoRAModule(nn.Module):
        def __init__(self, dim=768, r=64):
            super().__init__()
            self.lora_A_k = nn.Linear(dim, r, bias=False)
            self.lora_B_k = nn.Linear(r, dim, bias=False)
            self.lora_A_v = nn.Linear(dim, r, bias=False)
            self.lora_B_v = nn.Linear(r, dim, bias=False)

            # Initialize weights
            nn.init.normal_(self.lora_A_k.weight, mean=0.0, std=0.1)
            nn.init.normal_(self.lora_B_k.weight, mean=0.0, std=0.1)
            nn.init.normal_(self.lora_A_v.weight, mean=0.0, std=0.1)
            nn.init.normal_(self.lora_B_v.weight, mean=0.0, std=0.1)

            self._current_task = 0

        def set_current_task(self, task_id):
            self._current_task = task_id

        def get_A_k(self):
            return self.lora_A_k.weight

        def get_B_k(self):
            return self.lora_B_k.weight

        def get_A_v(self):
            return self.lora_A_v.weight

        def get_B_v(self):
            return self.lora_B_v.weight

        def get_lora_activation_k(self, x):
            return self.lora_A_k(x)

        def get_lora_activation_v(self, x):
            return self.lora_A_v(x)

    # Create mock module
    module = MockLoRAModule()

    # Test subspace extraction
    A_k = module.get_A_k()
    B_k = module.get_B_k()
    A_v = module.get_A_v()
    B_v = module.get_B_v()

    S_k = extract_subspace_from_BA(B_k, A_k, k=4)
    S_v = extract_subspace_from_BA(B_v, A_v, k=4)

    assert S_k.shape == (768, 4), f"Expected shape (768, 4), got {S_k.shape}"
    assert S_v.shape == (768, 4), f"Expected shape (768, 4), got {S_v.shape}"

    # Test plasticity loss
    x = torch.randn(16, 768)
    act_k = module.get_lora_activation_k(x)
    act_v = module.get_lora_activation_v(x)

    # Make activations non-negative for plasticity loss
    act_k = torch.abs(act_k) + 0.1
    act_v = torch.abs(act_v) + 0.1

    plast_loss_k = compute_plasticity_loss(act_k)
    plast_loss_v = compute_plasticity_loss(act_v)

    assert plast_loss_k > 0, "Key plasticity loss should be positive"
    assert plast_loss_v > 0, "Value plasticity loss should be positive"

    print("✓ Mock LoRA module test successful")


if __name__ == "__main__":
    print("Running Neuro-LoRA simple tests...\n")

    try:
        test_subspace_operations()
        test_gradient_projection()
        test_plasticity_loss()
        test_subspace_save_load()
        test_mock_lora_module()

        print("\n🎉 All simple tests completed successfully!")
        print("\nNeuro-LoRA implementation summary:")
        print("✓ Core mathematical utilities implemented and tested")
        print("✓ Subspace extraction and merging working correctly")
        print("✓ Gradient projection orthogonal to protected subspaces")
        print("✓ Plasticity loss encouraging diverse neuron usage")
        print("✓ Subspace persistence (save/load) functional")
        print("✓ Mock LoRA module integration verified")

    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
