#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script to verify the serialization fix for AD-DRS
"""

import torch
import json
import tempfile
import os
from utils.analysis import ADDRSAnalyzer


def test_serialization():
    """Test that device objects can be properly serialized."""

    # Create test config with problematic objects
    test_config = {
        "dataset": "cifar100",
        "model_name": "ad_drs",
        "device": [torch.device("cuda:0"), torch.device("cpu")],
        "dtype": torch.float32,
        "batch_size": 128,
        "epochs": 20,
        "lr": 0.001,
        "nested_dict": {
            "another_device": torch.device("cuda:1"),
            "normal_param": "test",
        },
        "device_list": [torch.device("cpu"), torch.device("cuda:0")],
    }

    print("Testing serialization fix...")

    try:
        # Create temporary analyzer
        with tempfile.TemporaryDirectory() as temp_dir:
            analyzer = ADDRSAnalyzer("test_experiment", save_dir=temp_dir)

            # This should not raise an error now
            analyzer.log_config(test_config)

            # Verify the config was saved
            config_file = os.path.join(analyzer.experiment_dir, "config.json")
            assert os.path.exists(config_file), "Config file was not created"

            # Verify it can be read back
            with open(config_file, "r") as f:
                loaded_config = json.load(f)

            print("✅ Serialization test passed!")
            print("Original device:", test_config["device"][0])
            print("Serialized device:", loaded_config["device"][0])
            print("Config saved successfully to:", config_file)

            return True

    except Exception as e:
        print(f"❌ Serialization test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_serialization()
    if success:
        print("\n🎉 The serialization fix is working correctly!")
        print(
            "You can now run: python train_addrs.py --config configs/addrs_cifar100_full.json"
        )
    else:
        print("\n⚠️  There are still issues with serialization.")
