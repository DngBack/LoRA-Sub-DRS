#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for DRS-Hyperspherical implementation

This script provides basic tests to verify the mathematical operations
and core components are working correctly.
"""

import torch
import numpy as np
import logging
from utils.spherical_geometry import SphericalGeometry, TangentPCA, PrototypeManager
from utils.angular_losses import ArcFaceHead, AngularTripletLoss


def test_spherical_geometry():
    """Test spherical geometry operations"""
    print("Testing Spherical Geometry Operations...")

    geom = SphericalGeometry()

    # Test normalization
    x = torch.randn(5, 10)
    x_norm = geom.normalize_to_sphere(x)
    norms = torch.norm(x_norm, dim=1)
    assert torch.allclose(norms, torch.ones(5), atol=1e-6), "Normalization failed"
    print("✓ Sphere normalization works")

    # Test geodesic distance
    u = torch.tensor([[1.0, 0.0, 0.0]])
    v = torch.tensor([[0.0, 1.0, 0.0]])
    dist = geom.geodesic_distance(u, v)
    expected = torch.tensor([np.pi / 2])
    assert torch.allclose(dist, expected, atol=1e-4), "Geodesic distance incorrect"
    print("✓ Geodesic distance works")

    # Test log/exp maps
    mu = torch.tensor([1.0, 0.0, 0.0])
    x = torch.tensor([[0.8, 0.6, 0.0]])
    x_norm = geom.normalize_to_sphere(x)

    # Log map followed by exp map should recover original point
    v = geom.log_map(mu, x_norm)
    x_recovered = geom.exp_map(mu, v)
    assert torch.allclose(x_norm, x_recovered, atol=1e-4), "Log/Exp map inconsistent"
    print("✓ Log/Exp maps work")

    # Test tangent projection
    tangent_v = geom.project_to_tangent(mu, v)
    dot_product = torch.sum(mu * tangent_v)
    assert torch.abs(dot_product) < 1e-5, "Tangent projection failed"
    print("✓ Tangent projection works")


def test_tangent_pca():
    """Test PCA in tangent space"""
    print("\nTesting Tangent PCA...")

    # Generate synthetic tangent vectors
    torch.manual_seed(42)
    tangent_vectors = torch.randn(100, 20)

    pca = TangentPCA(energy_threshold=0.9, max_components=10)
    pca.fit(tangent_vectors)

    assert pca.U is not None, "PCA not fitted"
    assert pca.n_components <= 10, "Too many components selected"
    assert pca.n_components > 0, "No components selected"
    print(f"✓ PCA fitted with {pca.n_components} components")

    # Test projection
    projected = pca.project(tangent_vectors[:5])
    assert projected.shape == tangent_vectors[:5].shape, "Projection shape mismatch"
    print("✓ PCA projection works")


def test_prototype_manager():
    """Test prototype management"""
    print("\nTesting Prototype Manager...")

    # Generate synthetic embeddings
    torch.manual_seed(42)
    embeddings = torch.randn(50, 16)
    embeddings = embeddings / torch.norm(embeddings, dim=1, keepdim=True)
    labels = torch.randint(0, 5, (50,))

    manager = PrototypeManager(
        feature_dim=16, num_classes=5, momentum=0.9, per_class=True
    )

    manager.initialize_prototypes(embeddings, labels, torch.device("cpu"))
    assert manager.prototypes.shape == (5, 16), "Prototype shape incorrect"
    print("✓ Prototype initialization works")

    # Test EMA updates
    new_embeddings = torch.randn(10, 16)
    new_embeddings = new_embeddings / torch.norm(new_embeddings, dim=1, keepdim=True)
    new_labels = torch.randint(0, 5, (10,))

    old_prototypes = manager.prototypes.clone()
    manager.update_prototypes(new_embeddings, new_labels)

    # Prototypes should have changed
    assert not torch.allclose(old_prototypes, manager.prototypes), (
        "Prototypes not updated"
    )
    print("✓ EMA prototype updates work")


def test_arcface_head():
    """Test ArcFace head with annealing"""
    print("\nTesting ArcFace Head...")

    head = ArcFaceHead(
        in_features=16,
        out_features=5,
        s_start=10.0,
        s_end=30.0,
        m_start=0.0,
        m_end=0.2,
        total_epochs=10,
        warmup_epochs=2,
    )

    # Test parameter annealing
    head.set_epoch(0)  # Warmup
    s0, m0 = head.get_current_params()
    assert s0 == 10.0 and m0 == 0.0, "Warmup parameters incorrect"

    head.set_epoch(5)  # Mid training
    s5, m5 = head.get_current_params()
    assert 10.0 < s5 < 30.0 and 0.0 < m5 < 0.2, "Annealing not working"

    head.set_epoch(10)  # End training
    s10, m10 = head.get_current_params()
    assert s10 == 30.0 and m10 == 0.2, "Final parameters incorrect"
    print("✓ ArcFace annealing works")

    # Test forward pass
    embeddings = torch.randn(8, 16)
    embeddings = embeddings / torch.norm(embeddings, dim=1, keepdim=True)
    labels = torch.randint(0, 5, (8,))

    logits = head(embeddings, labels)
    assert logits.shape == (8, 5), "ArcFace output shape incorrect"
    print("✓ ArcFace forward pass works")


def test_angular_triplet():
    """Test Angular Triplet Loss"""
    print("\nTesting Angular Triplet Loss...")

    triplet_loss = AngularTripletLoss(
        margin=0.2, mining_strategy="hard", distance_type="geodesic"
    )

    # Generate synthetic embeddings
    embeddings = torch.randn(16, 32)
    embeddings = embeddings / torch.norm(embeddings, dim=1, keepdim=True)
    labels = torch.randint(0, 4, (16,))

    # Test loss computation
    loss = triplet_loss(embeddings, labels)
    assert isinstance(loss, torch.Tensor), "Loss not a tensor"
    assert loss.numel() == 1, "Loss should be scalar"
    assert loss.item() >= 0, "Loss should be non-negative"
    print("✓ Angular Triplet Loss works")


def test_integration():
    """Test integration of components"""
    print("\nTesting Component Integration...")

    # Simulate a small DRS-Hyperspherical training step
    batch_size, feature_dim, num_classes = 8, 16, 3

    # Generate data
    embeddings = torch.randn(batch_size, feature_dim)
    embeddings = embeddings / torch.norm(embeddings, dim=1, keepdim=True)
    labels = torch.randint(0, num_classes, (batch_size,))

    # Setup components
    geom = SphericalGeometry()
    prototype_manager = PrototypeManager(feature_dim, num_classes, per_class=True)
    arcface = ArcFaceHead(feature_dim, num_classes)
    triplet = AngularTripletLoss()

    # Initialize prototypes
    prototype_manager.initialize_prototypes(embeddings, labels, torch.device("cpu"))

    # Compute losses
    arcface_logits = arcface(embeddings, labels)
    triplet_loss = triplet(embeddings, labels)

    # Verify everything works
    assert arcface_logits.shape == (batch_size, num_classes), "ArcFace shape wrong"
    assert isinstance(triplet_loss, torch.Tensor), "Triplet loss wrong type"
    print("✓ Component integration works")


def main():
    """Run all tests"""
    print("=" * 60)
    print("DRS-Hyperspherical Implementation Tests")
    print("=" * 60)

    try:
        test_spherical_geometry()
        test_tangent_pca()
        test_prototype_manager()
        test_arcface_head()
        test_angular_triplet()
        test_integration()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("DRS-Hyperspherical implementation is ready to use.")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        print("=" * 60)


if __name__ == "__main__":
    main()
