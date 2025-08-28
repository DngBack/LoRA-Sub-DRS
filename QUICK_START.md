# Environment Setup - Quick Start

## 1. Create Environment
```bash
conda create -n hdrs python=3.11.4 -y
conda activate hdrs
```

## 2. Install Dependencies
```bash
# Install PyTorch with CUDA 11.7
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --index-url https://download.pytorch.org/whl/cu117

# Install other packages
pip install -r requirements.txt
```

## 3. Verify Setup
```bash
python verify_setup.py
```

## 4. Run Experiments
```bash
# Small batch size (recommended for <24GB GPU)
python main.py --config configs/hdrs_cifar100_small.json

# Full experiment (requires 24GB+ GPU)
python main.py --config configs/hdrs_cifar100.json
```

## Troubleshooting

- **CUDA out of memory**: Use `hdrs_cifar100_small.json` config
- **NumPy errors**: Run `pip install numpy==1.24.3 --force-reinstall`
- **Import errors**: Check `python verify_setup.py` output

See `SETUP_GUIDE.md` for detailed instructions.
