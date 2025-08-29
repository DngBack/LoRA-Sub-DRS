#!/usr/bin/env python3
"""
Performance diagnostic script for HDRS training
This script runs a minimal version to identify bottlenecks
"""

import time
import torch
import torch.nn.functional as F
import numpy as np
from models.riemannian_lora import RiemannianAttention, compute_tangent_pca
import json
import sys
import os

def profile_attention():
    """Profile the Riemannian attention layer"""
    print("=== Profiling Riemannian Attention ===")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Create attention layer
    attention = RiemannianAttention(
        embed_dim=512,
        num_heads=8,
        rank=8,
        alpha=16,
        dropout=0.1,
        manifold_A="stiefel",
        manifold_B="sphere"
    ).to(device)
    
    # Create input
    batch_size, seq_len, embed_dim = 32, 49, 512  # Typical image patch size
    x = torch.randn(batch_size, seq_len, embed_dim, device=device)
    
    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = attention(x)
    
    # Time forward pass
    torch.cuda.synchronize()
    start_time = time.time()
    
    num_iters = 10
    for _ in range(num_iters):
        output = attention(x)
    
    torch.cuda.synchronize()
    forward_time = (time.time() - start_time) / num_iters
    
    print(f"Forward pass time: {forward_time:.4f}s")
    
    # Time backward pass
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(num_iters):
        output = attention(x)
        loss = output.sum()
        loss.backward()
        attention.zero_grad()
    
    torch.cuda.synchronize()
    backward_time = (time.time() - start_time) / num_iters
    
    print(f"Forward + backward time: {backward_time:.4f}s")
    print(f"Backward overhead: {backward_time - forward_time:.4f}s")

def profile_tangent_pca():
    """Profile the tangent PCA computation"""
    print("\n=== Profiling Tangent PCA ===")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Create some feature matrices to simulate DRS computation
    batch_size, feature_dim = 128, 512
    features = torch.randn(batch_size, feature_dim, device=device)
    
    # Time tangent PCA
    torch.cuda.synchronize()
    start_time = time.time()
    
    num_iters = 10
    for _ in range(num_iters):
        components, _ = compute_tangent_pca(features, energy_threshold=0.95, max_components=16)
    
    torch.cuda.synchronize()
    pca_time = (time.time() - start_time) / num_iters
    
    print(f"Tangent PCA time: {pca_time:.4f}s")
    print(f"PCA components shape: {components.shape if components is not None else 'None'}")

def profile_manifold_ops():
    """Profile manifold operations"""
    print("\n=== Profiling Manifold Operations ===")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Test Stiefel manifold operations
    rank, dim = 8, 512
    A = torch.randn(rank, dim, device=device)
    
    # Stiefel projection
    torch.cuda.synchronize()
    start_time = time.time()
    
    num_iters = 100
    for _ in range(num_iters):
        U, s, Vh = torch.linalg.svd(A, full_matrices=False)
        A_proj = U @ Vh
    
    torch.cuda.synchronize()
    stiefel_time = (time.time() - start_time) / num_iters
    
    print(f"Stiefel projection time: {stiefel_time:.4f}s")
    
    # Test Sphere manifold operations  
    B = torch.randn(dim, rank, device=device)
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(num_iters):
        B_proj = F.normalize(B, dim=0)
    
    torch.cuda.synchronize()
    sphere_time = (time.time() - start_time) / num_iters
    
    print(f"Sphere projection time: {sphere_time:.4f}s")

def check_pytorch_optimizations():
    """Check PyTorch optimization settings"""
    print("\n=== PyTorch Optimization Status ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
        print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Check for Flash Attention support
    try:
        # This will work if PyTorch 2.0+ is installed
        x = torch.randn(1, 8, 64, device="cuda:0")
        output = F.scaled_dot_product_attention(x, x, x)
        print("✅ Flash Attention (scaled_dot_product_attention) supported")
    except:
        print("❌ Flash Attention not supported")

def main():
    print("HDRS Performance Diagnostic Tool")
    print("=" * 50)
    
    check_pytorch_optimizations()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        print("\nRunning quick tests only...")
        profile_manifold_ops()
        return
    
    profile_attention()
    profile_tangent_pca() 
    profile_manifold_ops()
    
    print("\n=== Recommendations ===")
    print("If attention is slow: Enable Flash Attention in PyTorch 2.0+")
    print("If PCA is slow: Reduce max_components or energy_threshold")
    print("If manifold ops are slow: Consider reducing rank or using simpler projections")

if __name__ == "__main__":
    main()
