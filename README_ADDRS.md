# AD-DRS: Adaptive Merging in Drift-Resistant Space

## Overview

**AD-DRS (Adaptive Merging in Drift-Resistant Space)** is an advanced continual learning method that enhances the LoRA-Sub-DRS foundation with adaptive parameter merging using Bayesian optimization. This implementation provides a complete solution to the stability-plasticity dilemma in exemplar-free continual learning.

## Key Innovations

### 1. Two-Step Sequential Process

- **Step 1: Plasticity-Search Training in DRS** - Inherits the powerful drift-resistant space from LoRA-Sub
- **Step 2: Adaptive Merging** - Uses Bayesian optimization with Fisher Information Matrix to find optimal balance

### 2. Adaptive Coefficient Calculation

```
λ* = (Δθ^T F_t Δθ) / (Δθ^T F_t Δθ + Δθ^T Λ_{t-1} Δθ)
θ_t* = (1 - λ*) θ_{t-1}* + λ* θ_t^{cand}
```

### 3. Optional Refinement Techniques

- **Classifier Alignment**: Retrain classifiers on synthetic features from stored statistics
- **Self-Distillation**: Enforce consistency between class token and patch tokens

## Installation

The implementation is built on top of the LoRA-Sub-DRS codebase. Ensure you have the required dependencies:

```bash
pip install torch torchvision timm scipy matplotlib
```

## Usage

### Quick Start

1. **Basic AD-DRS Training**:

```bash
python train_addrs.py --config configs/addrs_cifar100.json
```

2. **AD-DRS with Refinements**:

```bash
python train_addrs.py --config configs/addrs_cifar100_full.json
```

3. **Generate Ablation Study Configs**:

```bash
python train_addrs.py --config configs/addrs_cifar100.json --ablation
```

### Configuration Options

#### Core AD-DRS Parameters

```json
{
  "model_name": "ad_drs", // Use AD-DRS model
  "use_classifier_alignment": true, // Enable classifier alignment
  "use_self_distillation": false, // Enable self-distillation
  "distill_temperature": 4.0, // Temperature for distillation
  "distill_alpha": 0.5, // Weight for distillation loss
  "fixed_lambda": null // For ablation: use fixed λ instead of adaptive
}
```

#### Example Configurations

**Standard AD-DRS** (`configs/addrs_cifar100.json`):

- Basic adaptive merging without refinements
- Good baseline for most experiments

**Full AD-DRS** (`configs/addrs_cifar100_full.json`):

- Includes classifier alignment
- Maximum performance configuration

## Ablation Studies

The implementation includes comprehensive ablation study support:

### Generating Ablation Configurations

```bash
python train_addrs.py --config configs/addrs_cifar100.json --ablation --ablation-dir my_ablation
```

This creates:

- `baseline_lorasub.json` - Original LoRA-Sub without adaptive merging
- `addrs_no_refinement.json` - AD-DRS without refinement techniques
- `addrs_fixed_lambda_05.json` - AD-DRS with fixed λ = 0.5
- `addrs_full.json` - Complete AD-DRS with all features

### Running Ablation Studies

```bash
# Run each configuration
python train_addrs.py --config my_ablation/baseline_lorasub.json
python train_addrs.py --config my_ablation/addrs_no_refinement.json
python train_addrs.py --config my_ablation/addrs_fixed_lambda_05.json
python train_addrs.py --config my_ablation/addrs_full.json
```

## Comprehensive Analysis

The implementation includes automatic analysis and visualization:

### Generated Outputs

- **Lambda Evolution Plot**: Shows how λ\* adapts over tasks
- **Accuracy Evolution**: Tracks average accuracy over time
- **Forgetting Analysis**: Measures catastrophic forgetting
- **Fisher Information Evolution**: Shows information accumulation

### Output Structure

```
experiments/
├── AD-DRS_cifar100_exp_20240101_123456/
│   ├── experiment.log              # Detailed training log
│   ├── config.json                 # Experiment configuration
│   ├── results.json                # Numerical results
│   ├── lambda_evolution.png        # λ* adaptation plot
│   ├── accuracy_evolution.png      # Performance over time
│   ├── forgetting_analysis.png     # Forgetting measurement
│   └── fisher_evolution.png        # Fisher information growth
```

## Key Implementation Files

### Core Components

- `methods/addrs.py` - Main AD-DRS implementation
- `methods/lorasub_drs.py` - Enhanced LoRA-Sub with adaptive merging
- `utils/fisher_utils.py` - Fisher Information Matrix utilities
- `utils/refinement.py` - Classifier alignment and self-distillation
- `utils/analysis.py` - Comprehensive experiment analysis

### Training Scripts

- `train_addrs.py` - Enhanced training script with ablation support
- `main.py` - Original training script (still compatible)

## Datasets

### CIFAR-100 (20 tasks, 5 classes each)

```bash
python train_addrs.py --config configs/addrs_cifar100.json
```

### ImageNet-R (20 tasks, 10 classes each)

```bash
python train_addrs.py --config configs/addrs_imagenetr.json
```

## Results Analysis

### Lambda Values Interpretation

- **λ\* ≈ 1.0**: Model favors plasticity (new task learning)
- **λ\* ≈ 0.0**: Model favors stability (old task preservation)
- **λ\* ≈ 0.5**: Balanced plasticity-stability trade-off

### Expected Performance

AD-DRS typically shows:

- **Adaptive λ values** that decrease over tasks as forgetting risk increases
- **Improved stability** compared to baseline LoRA-Sub
- **Maintained plasticity** for new task learning
- **Reduced forgetting** across all previous tasks

## Customization

### Adding New Refinement Techniques

1. Implement in `utils/refinement.py`
2. Add configuration parameters
3. Integrate in `methods/addrs.py`

### Custom Analysis Metrics

1. Extend `utils/analysis.py`
2. Add logging in `methods/addrs.py`
3. Update visualization functions

## Troubleshooting

### Common Issues

1. **CUDA Memory Error**:

   - Reduce batch size in configuration
   - Use gradient checkpointing if available

2. **Fisher Matrix Computation Error**:

   - Check that model parameters require gradients
   - Ensure dataloader provides valid data

3. **Lambda NaN Values**:
   - Increase epsilon in lambda calculation
   - Check Fisher matrix conditioning

### Debug Mode

Enable detailed logging:

```json
{
  "debug": true,
  "verbose_logging": true
}
```

## Citation

If you use this implementation, please cite:

```bibtex
@inproceedings{liu2025lora,
  title={LoRA Subtraction for Drift-Resistant Space in Exemplar-Free Continual Learning},
  author={Liu, Xuan and Chang, Xiaobin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}

@misc{addrs2024,
  title={AD-DRS: Adaptive Merging in Drift-Resistant Space},
  author={[Your Name]},
  year={2024},
  note={Enhancement of LoRA-Sub-DRS with Bayesian adaptive merging}
}
```

## License

This project builds upon the LoRA-Sub-DRS codebase and follows the same licensing terms.
