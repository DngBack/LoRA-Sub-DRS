"""
Training script for Hyperspherical Drift-Resistant Space (HDRS)
This script demonstrates how to train the Riemannian LoRA approach
"""

import torch
import torch.nn as nn
import numpy as np
from models.riemannian_lora import (
    RiemannianLoRALayer, 
    RiemannianAttention, 
    ManualSphere, 
    ManualStiefel,
    compute_karcher_mean_sphere,
    compute_tangent_pca
)

def test_riemannian_components():
    """Test individual Riemannian components"""
    print("Testing Riemannian LoRA Components...")
    
    # Test manual sphere operations
    print("\n1. Testing Sphere Operations:")
    x = torch.randn(5, 10)
    x_norm = ManualSphere.normalize(x)
    print(f"Original norms: {x.norm(dim=-1)}")
    print(f"Normalized norms: {x_norm.norm(dim=-1)}")
    
    # Test log/exp maps
    mu = ManualSphere.normalize(torch.randn(10))
    x_sphere = ManualSphere.normalize(torch.randn(10))
    v = ManualSphere.log_map(mu, x_sphere)
    x_reconstructed = ManualSphere.exp_map(mu, v)
    print(f"Reconstruction error: {(x_sphere - x_reconstructed).norm():.6f}")
    
    # Test Stiefel operations
    print("\n2. Testing Stiefel Operations:")
    X = torch.randn(10, 3)
    X_orth = ManualStiefel.retraction_qr(torch.zeros_like(X), X)
    orthogonality_error = (X_orth.T @ X_orth - torch.eye(3)).norm()
    print(f"Orthogonality error: {orthogonality_error:.6f}")
    
    # Test Karcher mean
    print("\n3. Testing Karcher Mean:")
    points = ManualSphere.normalize(torch.randn(20, 10))
    mu_karcher = compute_karcher_mean_sphere(points)
    print(f"Karcher mean norm: {mu_karcher.norm():.6f}")
    
    # Test tangent PCA
    print("\n4. Testing Tangent PCA:")
    mu, P_t = compute_tangent_pca(points, k=5)
    print(f"Tangent basis shape: {P_t.shape}")
    print(f"Basis orthogonality: {(P_t.T @ P_t - torch.eye(P_t.shape[1])).norm():.6f}")
    
    print("\nAll tests completed!")


def test_riemannian_lora_layer():
    """Test the Riemannian LoRA layer"""
    print("\n" + "="*50)
    print("Testing Riemannian LoRA Layer")
    print("="*50)
    
    # Create layer
    layer = RiemannianLoRALayer(
        input_dim=512,
        output_dim=256, 
        rank=16,
        n_tasks=5,
        use_geoopt=False,  # Use manual implementation
        manifold_A="stiefel",
        manifold_B="sphere"
    )
    
    print(f"Layer created with {len(layer.lora_A)} tasks")
    
    # Test forward pass
    x = torch.randn(32, 512)  # batch_size=32, input_dim=512
    
    # Task 0 (no subtraction)
    out0 = layer(x, task_id=0, use_subtraction=False)
    print(f"Task 0 output shape: {out0.shape}")
    
    # Task 1 (with subtraction) 
    out1 = layer(x, task_id=1, use_subtraction=True)
    print(f"Task 1 output shape: {out1.shape}")
    
    # Test DRS projection
    P_t = torch.randn(512, 32)  # Example projection matrix
    P_t = torch.linalg.qr(P_t)[0]  # Make orthogonal
    
    # Simulate gradient
    layer.lora_A[1].grad = torch.randn_like(layer.lora_A[1])
    layer.apply_drs_projection(P_t, task_id=1)
    print("DRS projection applied successfully")
    
    # Test optimizer
    optimizer = layer.get_riemannian_optimizer(lr=1e-3)
    print(f"Optimizer type: {type(optimizer)}")


def test_riemannian_attention():
    """Test the Riemannian attention module"""
    print("\n" + "="*50)
    print("Testing Riemannian Attention")
    print("="*50)
    
    # Create attention module
    attn = RiemannianAttention(
        dim=768,
        num_heads=12,
        rank=16,
        n_tasks=5,
        use_geoopt=False
    )
    
    print("Attention module created")
    
    # Test forward pass
    batch_size, seq_len, dim = 8, 196, 768
    x = torch.randn(batch_size, seq_len, dim)
    
    # Forward without LoRA
    out_no_lora = attn(x, task_id=-1)
    print(f"Output without LoRA: {out_no_lora.shape}")
    
    # Forward with LoRA (task 0)
    out_task0 = attn(x, task_id=0, collect_features=True)
    print(f"Output task 0: {out_task0.shape}")
    
    # Forward with LoRA (task 1, with subtraction)
    out_task1 = attn(x, task_id=1, collect_features=True)
    print(f"Output task 1: {out_task1.shape}")
    
    # Test DRS computation
    P_t = attn.compute_drs_projection(k=32)
    print(f"DRS projection computed: {P_t.shape}")
    
    # Apply DRS projection
    attn.lora_k.lora_A[1].grad = torch.randn_like(attn.lora_k.lora_A[1])
    attn.lora_v.lora_A[1].grad = torch.randn_like(attn.lora_v.lora_A[1])
    attn.apply_drs_projection(P_t, task_id=1)
    print("DRS projection applied to attention")


def demonstrate_manifold_subtraction():
    """Demonstrate manifold-aware LoRA subtraction"""
    print("\n" + "="*50)
    print("Demonstrating Manifold Subtraction")
    print("="*50)
    
    # Create some example LoRA matrices
    dim = 100
    rank = 8
    
    # Task 0 LoRA
    A0 = torch.randn(rank, dim) * 0.1
    B0 = ManualSphere.normalize(torch.randn(dim, rank).T).T  # columns normalized
    
    # Task 1 LoRA  
    A1 = torch.randn(rank, dim) * 0.1
    B1 = ManualSphere.normalize(torch.randn(dim, rank).T).T
    
    print("Original matrices created")
    
    # Compute cumulative LoRA effect
    cumulative = B0 @ A0 + B1 @ A1
    current = B1 @ A1
    
    # Riemannian subtraction (simplified version)
    def riemannian_subtract(current_matrix, cumulative_matrix, eta=0.1):
        result = current_matrix.clone()
        for i in range(current_matrix.shape[0]):
            # Get current row
            w_curr = ManualSphere.normalize(current_matrix[i])
            v_cum = cumulative_matrix[i] 
            
            # Project to tangent
            tau = ManualSphere.project_to_tangent(w_curr, v_cum)
            
            # Apply exponential map with negative direction
            if tau.norm() > 1e-8:
                result[i] = ManualSphere.exp_map(w_curr, -eta * tau)
                # Restore magnitude
                result[i] = result[i] * current_matrix[i].norm()
        
        return result
    
    # Apply subtraction
    subtracted = riemannian_subtract(current, cumulative)
    
    print(f"Current norm: {current.norm():.4f}")
    print(f"Cumulative norm: {cumulative.norm():.4f}")  
    print(f"Subtracted norm: {subtracted.norm():.4f}")
    print(f"Difference from current: {(current - subtracted).norm():.4f}")


def create_training_example():
    """Create a simple training example"""
    print("\n" + "="*50)
    print("Simple Training Example")
    print("="*50)
    
    # Simple dataset
    torch.manual_seed(42)
    X = torch.randn(100, 64)  # 100 samples, 64 features
    y = torch.randint(0, 5, (100,))  # 5 classes
    
    # Create model
    model = RiemannianLoRALayer(
        input_dim=64,
        output_dim=5,  # 5 classes
        rank=8,
        n_tasks=2,
        use_geoopt=False
    )
    
    # Training setup
    optimizer = model.get_riemannian_optimizer(lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    print("Starting training...")
    
    # Task 0 training
    task_0_mask = y < 3  # First 3 classes for task 0
    X_task0 = X[task_0_mask]
    y_task0 = y[task_0_mask]
    
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(X_task0, task_id=0, use_subtraction=False)
        loss = criterion(outputs, y_task0)
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            print(f"Task 0, Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # Task 1 training (with subtraction)
    task_1_mask = y >= 3  # Last 2 classes for task 1
    X_task1 = X[task_1_mask]
    y_task1 = y[task_1_mask] - 3  # Relabel to 0,1
    
    # Create new optimizer for task 1 to avoid gradient conflicts
    optimizer = model.get_riemannian_optimizer(lr=1e-3)
    
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(X_task1, task_id=1, use_subtraction=True)
        # Only use first 2 outputs for task 1
        outputs_task1 = outputs[:, :2]  
        loss = criterion(outputs_task1, y_task1)
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            print(f"Task 1, Epoch {epoch}, Loss: {loss.item():.4f}")
    
    print("Training completed!")


if __name__ == "__main__":
    print("Hyperspherical Drift-Resistant Space (HDRS) - Component Tests")
    print("=" * 70)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        # Run all tests
        test_riemannian_components()
        test_riemannian_lora_layer()
        test_riemannian_attention()
        demonstrate_manifold_subtraction()
        create_training_example()
        
        print("\n" + "="*70)
        print("All tests completed successfully!")
        print("You can now try running with the full HDRS method:")
        print("python main.py --config configs/hdrs_cifar100.json")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        
        print("\nTroubleshooting:")
        print("1. Make sure all dependencies are installed")
        print("2. For full Riemannian optimization, install geoopt: pip install geoopt")
        print("3. Check that the models directory is in your Python path")
