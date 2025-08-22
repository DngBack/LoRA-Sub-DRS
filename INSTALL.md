# Installation Guide for AD-DRS

This guide provides multiple installation methods for the AD-DRS (Adaptive Merging in Drift-Resistant Space) implementation.

## Quick Installation

### Method 1: Using pip (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/LoRA-Sub-DRS.git
cd LoRA-Sub-DRS

# Install minimal requirements
pip install -r requirements-minimal.txt
```

### Method 2: Full installation with all features

```bash
# Install all dependencies including visualization and analysis tools
pip install -r requirements.txt
```

### Method 3: Using conda/mamba (Alternative)

```bash
# Create conda environment
conda env create -f environment.yml
conda activate addrs
```

### Method 4: Development installation

```bash
# For development with editable installation
pip install -e .

# Or with development dependencies
pip install -e ".[dev,viz,notebooks]"
```

## System Requirements

### Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA support (recommended)
  - Minimum: 8GB VRAM (for CIFAR-100)
  - Recommended: 16GB+ VRAM (for ImageNet-R)
- **RAM**: Minimum 16GB system RAM
- **Storage**: At least 10GB free space for datasets and experiments

### Software Requirements

- **Python**: 3.8 - 3.11 (3.11.4 recommended)
- **CUDA**: 11.8+ (if using GPU)
- **Operating System**: Linux, macOS, or Windows

## Detailed Installation Steps

### Step 1: Environment Setup

#### Option A: Using conda (Recommended for beginners)

```bash
# Install conda/miniconda if not already installed
# Then create a new environment
conda create -n addrs python=3.11.4
conda activate addrs
```

#### Option B: Using venv

```bash
# Create virtual environment
python -m venv addrs_env

# Activate environment
# On Linux/macOS:
source addrs_env/bin/activate
# On Windows:
addrs_env\Scripts\activate
```

### Step 2: Install PyTorch

Choose the appropriate PyTorch installation command from [pytorch.org](https://pytorch.org/get-started/locally/) based on your system:

```bash
# Example for CUDA 11.8 (most common)
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# For CPU-only installation
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
```

### Step 3: Install AD-DRS Dependencies

```bash
# Install remaining dependencies
pip install timm==0.6.7 scipy matplotlib tqdm pillow numpy
```

### Step 4: Verify Installation

```bash
# Test the installation
python -c "
import torch
import timm
import numpy as np
import matplotlib.pyplot as plt
from methods.addrs import AD_DRS
print('✅ All dependencies installed successfully!')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
"
```

## Dataset Setup

### CIFAR-100

The dataset will be automatically downloaded when you first run the training.

### ImageNet-R

```bash
# Create data directory
mkdir -p data

# Download ImageNet-R from the official source
# https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar
wget https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar -O data/imagenet-r.tar

# Extract the dataset
cd data
tar -xf imagenet-r.tar
cd ..
```

## Troubleshooting

### Common Issues

#### 1. CUDA Version Mismatch

```bash
# Check your CUDA version
nvidia-smi

# Install compatible PyTorch version
# Visit https://pytorch.org/get-started/previous-versions/ for other versions
```

#### 2. timm Version Error

```bash
# If you see "PretrainedCfg object is not subscriptable"
pip install timm==0.6.7 --force-reinstall
```

#### 3. Import Errors

```bash
# Make sure you're in the correct directory
cd LoRA-Sub-DRS

# Add current directory to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### 4. Memory Issues

If you encounter out-of-memory errors:

- Reduce batch size in configuration files
- Use gradient checkpointing (if available)
- Use mixed precision training

### Environment Variables

Add these to your shell profile for convenience:

```bash
# Add to ~/.bashrc or ~/.zshrc
export ADDRS_ROOT="/path/to/LoRA-Sub-DRS"
export PYTHONPATH="${PYTHONPATH}:${ADDRS_ROOT}"
export CUDA_VISIBLE_DEVICES="0"  # Specify GPU to use
```

## Quick Test Run

After installation, test with a small experiment:

```bash
# Quick test with minimal epochs
python train_addrs.py --config configs/addrs_cifar100.json
```

## Docker Installation (Advanced)

For containerized deployment:

```dockerfile
# Dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

WORKDIR /workspace
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "train_addrs.py", "--config", "configs/addrs_cifar100.json"]
```

```bash
# Build and run
docker build -t addrs .
docker run --gpus all -v $(pwd)/data:/workspace/data addrs
```

## Performance Optimization

### For maximum performance:

1. **Use conda-forge packages** when possible for optimized builds
2. **Install MKL** for Intel CPUs: `conda install mkl`
3. **Use NVIDIA's PyTorch builds** for GPU optimization
4. **Set environment variables**:
   ```bash
   export OMP_NUM_THREADS=4
   export MKL_NUM_THREADS=4
   export CUDA_LAUNCH_BLOCKING=0
   ```

## Getting Help

If you encounter issues:

1. **Check the logs** in the generated experiment directories
2. **Enable debug mode** in configuration files
3. **Verify GPU availability** with `torch.cuda.is_available()`
4. **Check CUDA compatibility** between PyTorch and your GPU driver

For additional support, please refer to the troubleshooting section in `README_ADDRS.md` or open an issue in the repository.
