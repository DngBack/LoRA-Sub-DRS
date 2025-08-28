# HDRS Environment Setup Guide

This guide provides step-by-step instructions to set up the environment for HDRS (Hyperspherical Drift-Resistant Space) continual learning experiments.

## Prerequisites

- NVIDIA GPU with CUDA 11.7 support
- Anaconda or Miniconda installed
- At least 24GB GPU memory recommended for full experiments

## Quick Setup (Recommended)

### 1. Create Conda Environment

```bash
# Create a new conda environment with Python 3.11
conda create -n hdrs python=3.11.4 -y

# Activate the environment
conda activate hdrs
```

### 2. Install PyTorch with CUDA Support

```bash
# Install PyTorch 2.0.1 with CUDA 11.7 support
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --index-url https://download.pytorch.org/whl/cu117
```

### 3. Install Other Dependencies

```bash
# Install remaining packages from requirements.txt
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
# Test PyTorch CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Test geoopt installation
python -c "import geoopt; print(f'Geoopt version: {geoopt.__version__}')"

# Test timm installation
python -c "import timm; print(f'Timm version: {timm.__version__}')"
```

## Alternative Setup Methods

### Method 1: Using pip only

```bash
# Create virtual environment
python -m venv hdrs_env
source hdrs_env/bin/activate  # On Windows: hdrs_env\Scripts\activate

# Install PyTorch with CUDA
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --index-url https://download.pytorch.org/whl/cu117

# Install other dependencies
pip install -r requirements.txt
```

### Method 2: Using conda-forge

```bash
# Create conda environment
conda create -n hdrs python=3.11 -y
conda activate hdrs

# Install PyTorch from conda-forge
conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.7 -c pytorch -c nvidia

# Install remaining packages
pip install geoopt==0.5.1 timm==0.6.7 tqdm PyYAML requests Pillow ipython ipdb
```

## Package Versions Explained

### Core Components

- **torch==2.0.1+cu117**: PyTorch with CUDA 11.7 support for GPU acceleration
- **torchvision==0.15.2+cu117**: Computer vision utilities and datasets
- **timm==0.6.7**: Pre-trained vision transformer models
- **geoopt==0.5.1**: Riemannian optimization library for manifold-constrained learning

### Scientific Computing (Compatibility Critical)

- **numpy==1.24.3**: NumPy 2.x causes compatibility issues with other packages
- **scipy==1.11.4**: Compatible with NumPy 1.24.3
- **scikit-learn==1.3.0**: Compatible with NumPy 1.24.3 and SciPy 1.11.4

### Why These Specific Versions?

1. **NumPy 1.24.3**: NumPy 2.0+ breaks compatibility with many ML packages
2. **PyTorch 2.0.1**: Stable version with good CUDA 11.7 support
3. **geoopt 0.5.1**: Latest stable version with Stiefel/Sphere manifold support
4. **timm 0.6.7**: Contains required Vision Transformer architectures

## Common Issues and Solutions

### Issue 1: CUDA Out of Memory
```bash
# Solution: Reduce batch size in config files
# Edit configs/hdrs_cifar100.json: "batch_size": 32 (instead of 128)
```

### Issue 2: NumPy Version Conflicts
```bash
# Solution: Downgrade NumPy if needed
pip install numpy==1.24.3 --force-reinstall
```

### Issue 3: Missing CUDA Libraries
```bash
# Solution: Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --index-url https://download.pytorch.org/whl/cu117
```

### Issue 4: geoopt Installation Fails
```bash
# Solution: Install dependencies first
pip install torch numpy
pip install geoopt==0.5.1
```

## Running Experiments

### Basic CIFAR-100 Experiment
```bash
# Activate environment
conda activate hdrs

# Run HDRS on CIFAR-100 (reduced batch size for memory)
python main.py --config configs/hdrs_cifar100_small.json
```

### Full Experiments (Requires >20GB GPU memory)
```bash
# Original configuration
python main.py --config configs/hdrs_cifar100.json
```

## Memory Requirements

- **Minimum**: 8GB GPU memory (batch_size=16)
- **Recommended**: 12GB GPU memory (batch_size=32)
- **Full experiments**: 24GB GPU memory (batch_size=128)

## Environment Verification Script

Create and run this verification script:

```python
# verify_setup.py
import sys
import torch
import torchvision
import timm
import geoopt
import numpy as np
import scipy
import sklearn

print("=== HDRS Environment Verification ===")
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

print(f"NumPy version: {np.__version__}")
print(f"SciPy version: {scipy.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")
print(f"Timm version: {timm.__version__}")
print(f"Geoopt version: {geoopt.__version__}")

# Test critical functionality
try:
    # Test manifold operations
    from geoopt import Stiefel, Sphere
    print("✅ Geoopt manifolds working")
    
    # Test vision transformer
    model = timm.create_model('vit_base_patch16_224', pretrained=False)
    print("✅ Vision Transformer creation working")
    
    # Test CUDA tensor operations
    if torch.cuda.is_available():
        x = torch.randn(2, 3, 224, 224).cuda()
        y = model.cuda()(x)
        print("✅ CUDA tensor operations working")
    
    print("\n🎉 All checks passed! Environment is ready for HDRS experiments.")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("Please check your installation.")
```

Run with:
```bash
python verify_setup.py
```

## Troubleshooting

If you encounter issues, please:

1. Check GPU memory usage: `nvidia-smi`
2. Verify Python environment: `which python`
3. Check package versions: `pip list | grep -E "(torch|numpy|geoopt)"`
4. Run the verification script above

For additional help, see the original README.md or create an issue in the repository.
