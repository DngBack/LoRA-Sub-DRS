#!/usr/bin/env python3
"""
Test script for Hyperspherical DRS implementation
This script tests the core hyperspherical operations and utilities
"""

import torch
import numpy as np
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.hyperspherical import (
    normalize_to_sphere,
    mobius_transform,
    sample_spcauchy,
    angular_distance,
    spherical_covariance,
    spherical_pca,
    HypersphericalProjector,
    kl_spcauchy,
)


def test_normalize_to_sphere():
    """Test sphere normalization"""
    print("Testing normalize_to_sphere...")

    # Test with random vectors
    x = torch.randn(10, 768)
    x_norm = normalize_to_sphere(x)

    # Check if normalized vectors have unit norm
    norms = torch.norm(x_norm, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6), (
        "Normalized vectors should have unit norm"
    )

    print("✓ normalize_to_sphere test passed")


def test_angular_distance():
    """Test angular distance computation"""
    print("Testing angular_distance...")

    # Test with known vectors
    x = torch.tensor([[1.0, 0.0, 0.0]])  # x-axis
    y = torch.tensor([[0.0, 1.0, 0.0]])  # y-axis

    dist = angular_distance(x, y)
    expected = torch.tensor([np.pi / 2])  # 90 degrees

    assert torch.allclose(dist, expected, atol=1e-6), f"Expected {expected}, got {dist}"

    # Test with identical vectors
    dist_same = angular_distance(x, x)
    assert torch.allclose(dist_same, torch.zeros_like(dist_same), atol=1e-6), (
        "Distance between identical vectors should be 0"
    )

    print("✓ angular_distance test passed")


def test_mobius_transform():
    """Test Möbius transformation"""
    print("Testing mobius_transform...")

    # Test with unit vectors
    x = normalize_to_sphere(torch.randn(5, 768))
    mu = normalize_to_sphere(torch.randn(768))
    rho = 0.5

    y = mobius_transform(x, mu, rho)

    # Check if output is still on unit sphere
    norms = torch.norm(y, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6), (
        "Möbius output should be on unit sphere"
    )

    # Test with rho=0 (should be identity-like)
    y_identity = mobius_transform(x, mu, 0.0)
    # With rho=0, should get back to uniform distribution

    print("✓ mobius_transform test passed")


def test_spcauchy_sampling():
    """Test spherical Cauchy sampling"""
    print("Testing sample_spcauchy...")

    device = "cpu"  # Use CPU for testing to avoid device mismatch
    mu = normalize_to_sphere(torch.randn(768)).to(device)
    rho = 0.3
    num_samples = 100

    samples = sample_spcauchy(mu, rho, num_samples, device=device)

    # Check dimensions and unit norm
    assert samples.shape == (num_samples, 768), (
        f"Expected shape ({num_samples}, 768), got {samples.shape}"
    )

    norms = torch.norm(samples, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6), (
        "Samples should be on unit sphere"
    )

    print("✓ sample_spcauchy test passed")


def test_spherical_pca():
    """Test spherical PCA"""
    print("Testing spherical_pca...")

    # Generate correlated data on sphere
    base = torch.randn(100, 768)
    features = normalize_to_sphere(base)

    U, k = spherical_pca(features, variance_threshold=0.9)

    assert U.shape[0] == 768, "PCA components should have correct dimension"
    assert k > 0, "Should select at least one component"
    assert k <= 768, "Should not select more components than dimensions"

    print(f"✓ spherical_pca test passed (selected {k} components)")


def test_hyperspherical_projector():
    """Test HypersphericalProjector class"""
    print("Testing HypersphericalProjector...")

    projector = HypersphericalProjector(sphere_dim=768, spcauchy_rho=0.4)

    # Test with random features
    features = torch.randn(50, 768)

    # Compute projection
    P, k = projector.compute_projection(features, use_spcauchy=True)

    assert P.shape[0] == 768, "Projection matrix should have correct input dimension"
    assert P.shape[1] == k, "Projection matrix should have correct output dimension"

    # Test gradient projection
    grad = torch.randn(100, 10)  # Some random gradient
    projected_grad = projector.project_gradients(grad)

    assert projected_grad.shape == grad.shape, (
        "Projected gradient should have same shape"
    )

    print(f"✓ HypersphericalProjector test passed (k={k})")


def test_kl_spcauchy():
    """Test KL divergence for spherical Cauchy"""
    print("Testing kl_spcauchy...")

    # Test with known values
    kl = kl_spcauchy(0.5, 0.0, d=768)

    assert isinstance(kl, torch.Tensor), "KL should return tensor"
    assert kl.item() >= 0, "KL divergence should be non-negative"

    # KL should be 0 when distributions are identical
    kl_zero = kl_spcauchy(0.3, 0.3, d=768)
    assert abs(kl_zero.item()) < 1e-6, "KL should be 0 for identical distributions"

    print("✓ kl_spcauchy test passed")


def test_integration():
    """Test integration of hyperspherical components"""
    print("Testing integration scenario...")

    # Simulate a mini continual learning scenario
    device = "cpu"  # Use CPU for consistent testing

    # Task 1: Generate some features and prototypes
    features_t1 = normalize_to_sphere(torch.randn(20, 768)).to(device)
    proto_t1 = normalize_to_sphere(torch.mean(features_t1, dim=0, keepdim=True))

    # Task 2: Generate different features
    features_t2 = normalize_to_sphere(torch.randn(20, 768)).to(device)
    proto_t2 = normalize_to_sphere(torch.mean(features_t2, dim=0, keepdim=True))

    # Measure angular distance between prototypes
    drift = angular_distance(proto_t1, proto_t2)

    # Test hyperspherical projector
    projector = HypersphericalProjector(sphere_dim=768, spcauchy_rho=0.5)
    P, k = projector.compute_projection(features_t1, use_spcauchy=True)

    print(f"✓ Integration test passed - drift: {drift.item():.4f}, components: {k}")


def main():
    """Run all tests"""
    print("Starting Hyperspherical DRS Tests...")
    print("=" * 50)

    try:
        test_normalize_to_sphere()
        test_angular_distance()
        test_mobius_transform()
        test_spcauchy_sampling()
        test_spherical_pca()
        test_hyperspherical_projector()
        test_kl_spcauchy()
        test_integration()

        print("=" * 50)
        print(
            "🎉 All tests passed! Hyperspherical DRS implementation is working correctly."
        )
        return True

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
