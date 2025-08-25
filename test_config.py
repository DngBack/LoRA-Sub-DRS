#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script để kiểm tra việc load config JSON và mapping các biến
"""

import json
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_config_loading():
    """Test việc load và map config"""

    # Import hàm convert
    from train_hyperspherical import load_config, convert_config_for_trainer

    print("=== Testing Config Loading ===")

    try:
        # Load config JSON
        config = load_config("configs/drs_hyperspherical_cifar100.json")
        print("✓ JSON config loaded successfully")

        # Convert sang trainer format
        trainer_config = convert_config_for_trainer(config)
        print("✓ Config converted to trainer format")

        # Kiểm tra các biến quan trọng
        important_vars = [
            "seed",
            "model_name",
            "dataset",
            "batch_size",
            "lrate",
            "epochs",
            "device",
            "embd_dim",
            "epochs_warm",
            "epochs_main",
            "pca_energy",
            "ema_momentum",
            "s_start",
            "s_end",
            "m_start",
            "m_end",
            "triplet_lambda",
            "triplet_margin",
            "lambada",
            "margin_inter",
        ]

        print("\n=== Checking Important Variables ===")
        for var in important_vars:
            if var in trainer_config:
                print(f"✓ {var}: {trainer_config[var]}")
            else:
                print(f"✗ MISSING: {var}")

        print(f"\nTotal config keys: {len(trainer_config)}")
        print("All config keys:", list(trainer_config.keys()))

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_json_structure():
    """Test cấu trúc JSON"""
    print("\n=== Testing JSON Structure ===")

    try:
        with open(
            "configs/drs_hyperspherical_cifar100.json", "r", encoding="utf-8"
        ) as f:
            config = json.load(f)

        print("JSON structure:")
        for section, content in config.items():
            if isinstance(content, dict):
                print(f"  {section}:")
                for key, value in content.items():
                    print(f"    {key}: {value}")
            else:
                print(f"  {section}: {content}")

        return True
    except Exception as e:
        print(f"✗ Error reading JSON: {e}")
        return False


if __name__ == "__main__":
    print("DRS-Hyperspherical Config Test")
    print("=" * 50)

    # Test JSON structure
    success1 = test_json_structure()

    # Test config loading and conversion
    success2 = test_config_loading()

    if success1 and success2:
        print("\n" + "=" * 50)
        print("✅ ALL TESTS PASSED!")
        print("Config JSON is ready to use.")
    else:
        print("\n" + "=" * 50)
        print("❌ SOME TESTS FAILED!")
        sys.exit(1)
