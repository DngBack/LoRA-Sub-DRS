#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced training script for AD-DRS (Adaptive Merging in Drift-Resistant Space)

This script provides comprehensive training and analysis capabilities for AD-DRS,
including ablation studies and detailed performance analysis.
"""

import json
import argparse
import time
import os
import sys
import logging
from trainer import train
from utils.analysis import run_ablation_study


def main():
    args = setup_parser().parse_args()
    
    # Load configuration
    param = load_json(args.config)
    args_dict = vars(args)
    args_dict.update(param)
    
    # Set up logging
    setup_logging(args_dict)
    
    # Log experiment start
    logging.info("="*70)
    logging.info(f"Starting AD-DRS Training Session")
    logging.info(f"Configuration: {args.config}")
    logging.info(f"Model: {args_dict.get('model_name', 'unknown')}")
    logging.info(f"Dataset: {args_dict.get('dataset', 'unknown')}")
    logging.info("="*70)
    
    start_time = time.time()
    
    try:
        # Run training
        train(args_dict)
        
        # Calculate total time
        total_time = time.time() - start_time
        logging.info(f"Training completed successfully in {total_time:.2f} seconds")
        
    except Exception as e:
        logging.error(f"Training failed with error: {e}")
        raise
    
    logging.info("="*70)
    logging.info("AD-DRS Training Session Completed")
    logging.info("="*70)


def setup_logging(args_dict):
    """Set up comprehensive logging."""
    # Create logs directory if it doesn't exist
    log_dir = f"logs/{args_dict.get('dataset', 'unknown')}/{args_dict.get('model_name', 'unknown')}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log filename with timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"addrs_training_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.info(f"Logging to: {log_file}")


def load_json(settings_path):
    """Load JSON configuration file."""
    try:
        with open(settings_path) as data_file:
            param = json.load(data_file)
        logging.info(f"Loaded configuration from {settings_path}")
        return param
    except Exception as e:
        logging.error(f"Failed to load configuration from {settings_path}: {e}")
        raise


def setup_parser():
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description='AD-DRS: Adaptive Merging in Drift-Resistant Space for Continual Learning'
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/addrs_cifar100.json',
        help='Path to JSON configuration file'
    )
    
    parser.add_argument(
        '--device', 
        type=str, 
        default='0',
        help='GPU device ID'
    )
    
    parser.add_argument(
        '--eval', 
        action='store_true', 
        help='Perform evaluation only'
    )
    
    parser.add_argument(
        '--ablation', 
        action='store_true',
        help='Generate ablation study configurations'
    )
    
    parser.add_argument(
        '--ablation-dir',
        type=str,
        default='ablation_configs',
        help='Directory to save ablation study configurations'
    )
    
    return parser


def run_ablation_command(args):
    """Run ablation study configuration generation."""
    print("="*70)
    print("AD-DRS Ablation Study Configuration Generator")
    print("="*70)
    
    # Generate ablation configurations
    run_ablation_study(args.config, args.ablation_dir)
    
    print("\nAblation study configurations generated!")
    print("\nTo run the ablation studies:")
    print("1. Baseline LoRA-Sub:")
    print(f"   python train_addrs.py --config {args.ablation_dir}/baseline_lorasub.json")
    print("2. AD-DRS without refinement:")
    print(f"   python train_addrs.py --config {args.ablation_dir}/addrs_no_refinement.json")
    print("3. AD-DRS with fixed lambda:")
    print(f"   python train_addrs.py --config {args.ablation_dir}/addrs_fixed_lambda_05.json")
    print("4. Full AD-DRS:")
    print(f"   python train_addrs.py --config {args.ablation_dir}/addrs_full.json")
    print("\nCompare results to analyze the contribution of each component!")


if __name__ == '__main__':
    # Parse arguments
    args = setup_parser().parse_args()
    
    # Handle ablation study generation
    if args.ablation:
        run_ablation_command(args)
    else:
        # Run normal training
        main()
