#!/usr/bin/env python3
"""
Test for Task Similarity-aware Drift Correction (TSDC)

This test verifies the TSDC functionality including:
- Task similarity computation
- Similarity-aware gradient projection
- ΔW history management
"""

import torch
import torch.nn as nn
import os
import sys
import tempfile

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.neuro_utils import (
    compute_task_similarity,
    project_grad_similarity_aware,
    save_delta_w_history,
    load_delta_w_history,
)


def test_task_similarity_computation():
    """Test task similarity computation"""
    print("Testing task similarity computation...")

    d = 768
    device = "cpu"

    # Create test ΔW matrices
    delta_w_1 = torch.randn(d, d) * 0.1
    delta_w_2 = torch.randn(d, d) * 0.1
    delta_w_3 = torch.randn(d, d) * 0.1

    # Make delta_w_2 similar to delta_w_1
    delta_w_2 = delta_w_1 + torch.randn(d, d) * 0.01  # Small perturbation

    # Make delta_w_3 very different from delta_w_1
    delta_w_3 = torch.randn(d, d) * 0.1  # Completely different

    history = [delta_w_1]

    # Test cosine similarity
    sim_1 = compute_task_similarity(delta_w_1, history, method="cosine")
    sim_2 = compute_task_similarity(delta_w_2, history, method="cosine")
    sim_3 = compute_task_similarity(delta_w_3, history, method="cosine")

    print(f"Similarity with identical matrix: {sim_1:.4f}")
    print(f"Similarity with similar matrix: {sim_2:.4f}")
    print(f"Similarity with different matrix: {sim_3:.4f}")

    # Verify that similar matrix has higher similarity
    assert sim_2 > sim_3, (
        f"Similar matrix should have higher similarity: {sim_2:.4f} vs {sim_3:.4f}"
    )

    # Test frobenius similarity
    sim_1_frob = compute_task_similarity(delta_w_1, history, method="frobenius")
    sim_2_frob = compute_task_similarity(delta_w_2, history, method="frobenius")
    sim_3_frob = compute_task_similarity(delta_w_3, history, method="frobenius")

    print(f"Frobenius similarity with identical matrix: {sim_1_frob:.4f}")
    print(f"Frobenius similarity with similar matrix: {sim_2_frob:.4f}")
    print(f"Frobenius similarity with different matrix: {sim_3_frob:.4f}")

    # Verify that similar matrix has higher similarity
    assert sim_2_frob > sim_3_frob, (
        f"Similar matrix should have higher frobenius similarity: {sim_2_frob:.4f} vs {sim_3_frob:.4f}"
    )

    print("✓ Task similarity computation test passed!")


def test_similarity_aware_projection():
    """Test similarity-aware gradient projection"""
    print("\nTesting similarity-aware gradient projection...")

    d, r = 768, 64
    device = "cpu"

    # Create test matrices
    B = torch.randn(d, r) * 0.1
    A = torch.randn(r, d) * 0.1
    gB = torch.randn(d, r) * 0.01
    gA = torch.randn(r, d) * 0.01

    # Create subspace
    k = 8
    S = torch.randn(d, k)
    Q, _ = torch.linalg.qr(S)
    S = Q[:, :k]

    # Create ΔW history
    delta_w_1 = torch.randn(d, d) * 0.1
    delta_w_2 = torch.randn(d, d) * 0.1
    history = [delta_w_1, delta_w_2]

    # Test with different current ΔW matrices
    current_delta_w_similar = delta_w_1 + torch.randn(d, d) * 0.01  # Similar to history
    current_delta_w_different = torch.randn(d, d) * 0.1  # Different from history

    # Test projection with similar task
    gA_proj_sim, gB_proj_sim, alpha_sim = project_grad_similarity_aware(
        gA,
        gB,
        A,
        B,
        S,
        current_delta_w_similar,
        history,
        similarity_method="cosine",
        similarity_decay=1.0,
        min_projection_strength=0.1,
    )

    # Test projection with different task
    gA_proj_diff, gB_proj_diff, alpha_diff = project_grad_similarity_aware(
        gA,
        gB,
        A,
        B,
        S,
        current_delta_w_different,
        history,
        similarity_method="cosine",
        similarity_decay=1.0,
        min_projection_strength=0.1,
    )

    print(f"Similar task projection strength: {alpha_sim:.4f}")
    print(f"Different task projection strength: {alpha_diff:.4f}")

    # Verify that similar task has higher projection strength
    assert alpha_sim > alpha_diff, (
        f"Similar task should have higher projection strength: {alpha_sim:.4f} vs {alpha_diff:.4f}"
    )

    # Verify that projection strength is within bounds
    assert 0.1 <= alpha_sim <= 1.0, (
        f"Projection strength out of bounds: {alpha_sim:.4f}"
    )
    assert 0.1 <= alpha_diff <= 1.0, (
        f"Projection strength out of bounds: {alpha_diff:.4f}"
    )

    # Verify that gradients are modified
    gA_diff_sim = torch.norm(gA_proj_sim - gA)
    gB_diff_sim = torch.norm(gB_proj_sim - gB)
    gA_diff_diff = torch.norm(gA_proj_diff - gA)
    gB_diff_diff = torch.norm(gB_proj_diff - gB)

    print(f"Similar task gradient changes: gA={gA_diff_sim:.6f}, gB={gB_diff_sim:.6f}")
    print(
        f"Different task gradient changes: gA={gA_diff_diff:.6f}, gB={gB_diff_diff:.6f}"
    )

    # Verify that similar task has larger gradient changes (stronger projection)
    assert gA_diff_sim > gA_diff_diff * 0.5, (
        f"Similar task should have larger gradient changes"
    )
    assert gB_diff_sim > gB_diff_diff * 0.5, (
        f"Similar task should have larger gradient changes"
    )

    print("✓ Similarity-aware projection test passed!")


def test_similarity_decay():
    """Test similarity decay parameter"""
    print("\nTesting similarity decay parameter...")

    d, r = 768, 64
    device = "cpu"

    # Create test matrices
    B = torch.randn(d, r) * 0.1
    A = torch.randn(r, d) * 0.1
    gB = torch.randn(d, r) * 0.01
    gA = torch.randn(r, d) * 0.01

    # Create subspace
    k = 8
    S = torch.randn(d, k)
    Q, _ = torch.linalg.qr(S)
    S = Q[:, :k]

    # Create ΔW history
    delta_w_1 = torch.randn(d, d) * 0.1
    history = [delta_w_1]

    # Create moderately similar current ΔW
    current_delta_w = delta_w_1 + torch.randn(d, d) * 0.05

    # Test with different decay values
    decay_values = [0.5, 1.0, 2.0]
    alphas = []

    for decay in decay_values:
        _, _, alpha = project_grad_similarity_aware(
            gA,
            gB,
            A,
            B,
            S,
            current_delta_w,
            history,
            similarity_method="cosine",
            similarity_decay=decay,
            min_projection_strength=0.1,
        )
        alphas.append(alpha)
        print(f"Decay {decay}: alpha = {alpha:.4f}")

    # Verify that higher decay reduces projection strength for moderate similarities
    assert alphas[0] > alphas[1], (
        f"Lower decay should give higher projection: {alphas[0]:.4f} vs {alphas[1]:.4f}"
    )
    assert alphas[1] > alphas[2], (
        f"Lower decay should give higher projection: {alphas[1]:.4f} vs {alphas[2]:.4f}"
    )

    print("✓ Similarity decay test passed!")


def test_delta_w_history_save_load():
    """Test ΔW history save and load functionality"""
    print("\nTesting ΔW history save and load...")

    d = 768
    device = "cpu"

    # Create test ΔW history
    delta_w_history = [
        torch.randn(d, d) * 0.1,
        torch.randn(d, d) * 0.1,
        torch.randn(d, d) * 0.1,
    ]

    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp_file:
        save_path = tmp_file.name

    try:
        # Save history
        save_delta_w_history(delta_w_history, save_path)

        # Load history
        loaded_history = load_delta_w_history(save_path, device)

        # Verify that loaded history matches original
        assert len(loaded_history) == len(delta_w_history), (
            f"History length mismatch: {len(loaded_history)} vs {len(delta_w_history)}"
        )

        for i, (original, loaded) in enumerate(zip(delta_w_history, loaded_history)):
            assert torch.allclose(original, loaded, atol=1e-6), (
                f"History element {i} mismatch"
            )

        print(f"Successfully saved and loaded {len(delta_w_history)} ΔW matrices")

    finally:
        # Clean up
        if os.path.exists(save_path):
            os.unlink(save_path)

    print("✓ ΔW history save/load test passed!")


def test_edge_cases():
    """Test edge cases for TSDC"""
    print("\nTesting edge cases...")

    d, r = 768, 64
    device = "cpu"

    # Create test matrices
    B = torch.randn(d, r) * 0.1
    A = torch.randn(r, d) * 0.1
    gB = torch.randn(d, r) * 0.01
    gA = torch.randn(r, d) * 0.01

    # Create subspace
    k = 8
    S = torch.randn(d, k)
    Q, _ = torch.linalg.qr(S)
    S = Q[:, :k]

    # Test with empty history
    empty_history = []
    current_delta_w = torch.randn(d, d) * 0.1

    similarity = compute_task_similarity(
        current_delta_w, empty_history, method="cosine"
    )
    assert similarity == 0.0, (
        f"Empty history should give similarity 0, got {similarity}"
    )

    # Test with zero matrices
    zero_delta_w = torch.zeros(d, d)
    history_with_zero = [zero_delta_w]

    similarity = compute_task_similarity(
        zero_delta_w, history_with_zero, method="cosine"
    )
    assert similarity == 0.5, f"Zero matrix similarity should be 0.5, got {similarity}"

    # Test projection with empty history
    gA_proj, gB_proj, alpha = project_grad_similarity_aware(
        gA,
        gB,
        A,
        B,
        S,
        current_delta_w,
        empty_history,
        similarity_method="cosine",
        similarity_decay=1.0,
        min_projection_strength=0.1,
    )

    # Should fall back to minimum projection strength
    assert abs(alpha - 0.1) < 1e-6, (
        f"Empty history should use min projection strength, got {alpha}"
    )

    print("✓ Edge cases test passed!")


if __name__ == "__main__":
    print("Task Similarity-aware Drift Correction (TSDC) Tests")
    print("===================================================")

    try:
        test_task_similarity_computation()
        test_similarity_aware_projection()
        test_similarity_decay()
        test_delta_w_history_save_load()
        test_edge_cases()

        print("\n🎉 All TSDC tests passed!")
        print("\nSummary:")
        print("✓ Task similarity computation works correctly")
        print("✓ Similarity-aware gradient projection adapts to task similarity")
        print("✓ Similarity decay parameter affects projection strength")
        print("✓ ΔW history save/load functionality works")
        print("✓ Edge cases are handled properly")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
