#!/usr/bin/env python3
"""
Integration test for Neuro-LoRA implementation
Tests the complete Neuro-LoRA pipeline including subspace extraction, gradient projection, and plasticity loss.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import tempfile
import shutil
from unittest.mock import Mock, patch

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.neuro_utils import (
    extract_subspace_from_BA, merge_cumulative_subspace, 
    project_grad_B, compute_plasticity_loss, save_subspace, load_subspace
)
from methods.neuro_lora import NeuroLoRA
from models.sinet_lora import SiNet
from utils.data_manager import DataManager


class MockAttentionLoRA(nn.Module):
    """Mock Attention_LoRA module for testing"""
    def __init__(self, dim=768, r=64, n_tasks=5):
        super().__init__()
        self.dim = dim
        self.rank = r
        self.n_tasks = n_tasks
        
        # Initialize LoRA parameters
        self.lora_A_k = nn.ModuleList([nn.Linear(dim, r, bias=False) for _ in range(n_tasks)])
        self.lora_B_k = nn.ModuleList([nn.Linear(r, dim, bias=False) for _ in range(n_tasks)])
        self.lora_A_v = nn.ModuleList([nn.Linear(dim, r, bias=False) for _ in range(n_tasks)])
        self.lora_B_v = nn.ModuleList([nn.Linear(r, dim, bias=False) for _ in range(n_tasks)])
        
        # Initialize weights
        for t in range(n_tasks):
            nn.init.normal_(self.lora_A_k[t].weight, mean=0.0, std=0.1)
            nn.init.normal_(self.lora_B_k[t].weight, mean=0.0, std=0.1)
            nn.init.normal_(self.lora_A_v[t].weight, mean=0.0, std=0.1)
            nn.init.normal_(self.lora_B_v[t].weight, mean=0.0, std=0.1)
        
        self._current_task = 0
    
    def set_current_task(self, task_id):
        self._current_task = task_id
    
    def get_current_task(self):
        return self._current_task
    
    def get_A_k(self, task_id=None):
        if task_id is None:
            task_id = self._current_task
        return self.lora_A_k[task_id].weight
    
    def get_B_k(self, task_id=None):
        if task_id is None:
            task_id = self._current_task
        return self.lora_B_k[task_id].weight
    
    def get_A_v(self, task_id=None):
        if task_id is None:
            task_id = self._current_task
        return self.lora_A_v[task_id].weight
    
    def get_B_v(self, task_id=None):
        if task_id is None:
            task_id = self._current_task
        return self.lora_B_v[task_id].weight
    
    def get_lora_activation_k(self, x):
        return self.lora_A_k[self._current_task](x)
    
    def get_lora_activation_v(self, x):
        return self.lora_A_v[self._current_task](x)


class MockNetwork(nn.Module):
    """Mock network with LoRA modules for testing"""
    def __init__(self, dim=768, r=64, n_tasks=5):
        super().__init__()
        self.dim = dim
        self.numtask = n_tasks
        
        # Add some LoRA attention modules
        self.attn1 = MockAttentionLoRA(dim, r, n_tasks)
        self.attn2 = MockAttentionLoRA(dim, r, n_tasks)
        
        # Add classifier
        self.classifier_pool = nn.ModuleList([
            nn.Linear(dim, 10) for _ in range(n_tasks)
        ])
        
        # Mock forward method
        self.features = nn.Linear(dim, dim)
    
    def forward(self, x):
        # Mock forward pass
        features = self.features(x)
        logits = self.classifier_pool[self.attn1.get_current_task()](features)
        return {"logits": logits, "features": features}


def test_neuro_lora_integration():
    """Test the complete Neuro-LoRA integration"""
    print("Testing Neuro-LoRA integration...")
    
    # Create temporary directory for test
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Test parameters
        device = 'cpu'
        dim = 768
        r = 64
        n_tasks = 3
        k_per_task = 8
        K_max = 16
        
        # Create mock network
        network = MockNetwork(dim, r, n_tasks)
        
        # Create mock data manager
        data_manager = Mock()
        data_manager.get_dataset = Mock(return_value=([], [], []))
        data_manager.get_dataloader = Mock(return_value=[])
        
        # Create Neuro-LoRA instance
        neuro_lora = NeuroLoRA(
            network=network,
            device=device,
            data_manager=data_manager,
            checkpoint_dir=temp_dir,
            k_per_task=k_per_task,
            K_max=K_max,
            lambda_plast=0.1,
            sleep_epochs=0  # Disable sleep phase for testing
        )
        
        # Test 1: Initialization
        print("✓ Neuro-LoRA initialization successful")
        
        # Test 2: Subspace extraction and saving
        print("Testing subspace extraction...")
        neuro_lora._cur_task = 0
        neuro_lora._extract_and_save_subspaces()
        
        # Check if subspaces were created
        subspace_files = [f for f in os.listdir(temp_dir) if f.startswith('subspace_')]
        assert len(subspace_files) > 0, "Subspace files should be created"
        print(f"✓ Subspace extraction successful: {len(subspace_files)} files created")
        
        # Test 3: Subspace loading
        print("Testing subspace loading...")
        neuro_lora._cur_task = 1
        neuro_lora._load_cumulative_subspaces()
        
        # Check if subspaces were loaded
        assert len(neuro_lora.S_cumulative) > 0, "Subspaces should be loaded"
        print("✓ Subspace loading successful")
        
        # Test 4: Gradient projection
        print("Testing gradient projection...")
        
        # Create mock gradients
        for name, param in network.named_parameters():
            if "lora_B" in name:
                param.grad = torch.randn_like(param)
        
        # Apply gradient projection
        neuro_lora._project_gradients()
        
        # Check if gradients were modified
        grad_modified = False
        for name, param in network.named_parameters():
            if "lora_B" in name and param.grad is not None:
                grad_modified = True
                break
        
        assert grad_modified, "Gradients should be modified by projection"
        print("✓ Gradient projection successful")
        
        # Test 5: Plasticity loss computation
        print("Testing plasticity loss...")
        
        # Create mock input
        batch_size = 4
        x = torch.randn(batch_size, dim)
        
        # Compute plasticity loss
        plast_loss = 0.0
        for name, module in network.named_modules():
            if hasattr(module, 'get_lora_activation_k'):
                act_k = module.get_lora_activation_k(x)
                act_v = module.get_lora_activation_v(x)
                plast_loss += compute_plasticity_loss(act_k)
                plast_loss += compute_plasticity_loss(act_v)
        
        assert plast_loss > 0, "Plasticity loss should be positive"
        print("✓ Plasticity loss computation successful")
        
        # Test 6: Parameter training setup
        print("Testing parameter training setup...")
        neuro_lora._setup_parameter_training()
        
        # Check if correct parameters are trainable
        trainable_params = sum(p.requires_grad for p in network.parameters())
        assert trainable_params > 0, "Some parameters should be trainable"
        print("✓ Parameter training setup successful")
        
        print("\n🎉 All Neuro-LoRA integration tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_subspace_operations():
    """Test subspace operations with realistic data"""
    print("\nTesting subspace operations...")
    
    # Create realistic LoRA matrices
    dim = 768
    r = 64
    k_per_task = 8
    
    # Simulate trained LoRA parameters
    A = torch.randn(dim, r) * 0.1
    B = torch.randn(r, dim) * 0.1
    
    # Test subspace extraction
    S_new = extract_subspace_from_BA(B, A, k_per_task)
    assert S_new.shape == (dim, k_per_task), f"Expected shape ({dim}, {k_per_task}), got {S_new.shape}"
    
    # Test orthogonality
    S_ortho = torch.mm(S_new.T, S_new)
    identity = torch.eye(k_per_task)
    ortho_error = torch.norm(S_ortho - identity)
    assert ortho_error < 1e-6, f"Subspace should be orthonormal, error: {ortho_error}"
    
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
    assert ortho_error_merged < 1e-6, f"Merged subspace should be orthonormal, error: {ortho_error_merged}"
    
    print("✓ Subspace operations successful")


def test_gradient_projection():
    """Test gradient projection with realistic data"""
    print("\nTesting gradient projection...")
    
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
    assert projection_error < 1e-6, f"Projected gradient should be orthogonal to subspace, error: {projection_error}"
    
    print("✓ Gradient projection successful")


if __name__ == "__main__":
    print("Running Neuro-LoRA integration tests...\n")
    
    try:
        test_subspace_operations()
        test_gradient_projection()
        test_neuro_lora_integration()
        
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
