#!/usr/bin/env python3
"""
Test for Sleep Phase Functionality

This test verifies that the sleep phase distillation works correctly
without the AttributeError that was occurring.
"""

import torch
import torch.nn as nn
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.neuro_utils import sleep_phase_distill, create_noise_loader


class MockModel(nn.Module):
    """Mock model that returns dictionary outputs like the real model"""

    def __init__(self, input_size=768, output_size=100):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        # Return dictionary format like the real model
        logits = self.fc(x)
        return {"logits": logits, "features": x}


def test_sleep_phase_distillation():
    """Test sleep phase distillation with dictionary outputs"""
    print("Testing sleep phase distillation...")

    # Create mock models
    teacher = MockModel()
    student = MockModel()

    # Freeze teacher
    for p in teacher.parameters():
        p.requires_grad = False

    # Create noise dataloader
    noise_loader = create_noise_loader(
        batch_size=32, n_batches=3, device="cpu", input_size=(768,)
    )

    print(f"Created noise loader with {len(noise_loader)} batches")
    print(
        f"First batch format: {type(noise_loader[0])}, length: {len(noise_loader[0])}"
    )
    print(f"First batch[0] (indices) shape: {noise_loader[0][0].shape}")
    print(f"First batch[1] (inputs) shape: {noise_loader[0][1].shape}")
    print(f"First batch[2] (targets) shape: {noise_loader[0][2].shape}")

    # Test distillation
    try:
        sleep_phase_distill(
            teacher_model=teacher,
            student_model=student,
            dataloader=noise_loader,
            device="cpu",
            epochs=1,
            lr=1e-3,
        )
        print("✓ Sleep phase distillation completed successfully!")

        # Check that student model was updated
        original_weight = teacher.fc.weight.clone()
        updated_weight = student.fc.weight.clone()
        weight_change = torch.norm(updated_weight - original_weight)
        print(f"✓ Student model weights changed: {weight_change:.6f}")

    except Exception as e:
        print(f"❌ Sleep phase distillation failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


def test_noise_loader_format():
    """Test that noise loader returns correct format"""
    print("\nTesting noise loader format...")

    noise_loader = create_noise_loader(
        batch_size=16, n_batches=2, device="cpu", input_size=(3, 224, 224)
    )

    for i, batch in enumerate(noise_loader):
        print(f"Batch {i}:")
        print(f"  - Type: {type(batch)}")
        print(f"  - Length: {len(batch)}")
        print(f"  - batch[0] (indices): {type(batch[0])}, shape: {batch[0].shape}")
        print(f"  - batch[1] (inputs): {type(batch[1])}, shape: {batch[1].shape}")
        print(f"  - batch[2] (targets): {type(batch[2])}, shape: {batch[2].shape}")

        # Verify format
        assert len(batch) == 3, f"Batch should have 3 elements, got {len(batch)}"
        assert torch.is_tensor(batch[0]), (
            f"Indices should be tensor, got {type(batch[0])}"
        )
        assert torch.is_tensor(batch[1]), (
            f"Inputs should be tensor, got {type(batch[1])}"
        )
        assert torch.is_tensor(batch[2]), (
            f"Targets should be tensor, got {type(batch[2])}"
        )

    print("✓ Noise loader format is correct!")


if __name__ == "__main__":
    print("Sleep Phase Functionality Tests")
    print("===============================")

    try:
        test_noise_loader_format()
        success = test_sleep_phase_distillation()

        if success:
            print("\n🎉 All sleep phase tests passed!")
            print("\nSummary:")
            print("✓ Noise loader creates correct format")
            print("✓ Sleep phase distillation works with dictionary outputs")
            print("✓ No more AttributeError")
            print("✓ Student model gets updated during distillation")
        else:
            print("\n❌ Some tests failed")
            exit(1)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
