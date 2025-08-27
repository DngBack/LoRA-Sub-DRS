# Installation script for HDRS dependencies (Windows)

Write-Host "Installing Hyperspherical Drift-Resistant Space (HDRS) dependencies..."

# Core dependencies
Write-Host "Installing core PyTorch dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Additional dependencies  
Write-Host "Installing additional dependencies..."
pip install numpy scipy matplotlib tqdm

# Optional: Geoopt for advanced Riemannian optimization
Write-Host "Installing Geoopt (optional, for advanced Riemannian optimization)..."
pip install geoopt

# Timm for Vision Transformer components
Write-Host "Installing timm..."
pip install timm

Write-Host "Installation completed!"
Write-Host ""
Write-Host "Test the installation by running:"
Write-Host "python train_hyperspherical.py"
Write-Host ""
Write-Host "Run full HDRS training with:"
Write-Host "python main.py --config configs/hdrs_cifar100.json"
