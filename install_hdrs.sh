#!/bin/bash
# Installation script for HDRS dependencies

echo "Installing Hyperspherical Drift-Resistant Space (HDRS) dependencies..."

# Core dependencies
echo "Installing core PyTorch dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Additional dependencies
echo "Installing additional dependencies..."
pip install numpy scipy matplotlib tqdm

# Optional: Geoopt for advanced Riemannian optimization
echo "Installing Geoopt (optional, for advanced Riemannian optimization)..."
pip install geoopt

# Timm for Vision Transformer components
echo "Installing timm..."
pip install timm

echo "Installation completed!"
echo ""
echo "Test the installation by running:"
echo "python train_hyperspherical.py"
echo ""
echo "Run full HDRS training with:"
echo "python main.py --config configs/hdrs_cifar100.json"
