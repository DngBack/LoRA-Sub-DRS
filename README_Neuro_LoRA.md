# Neuro-LoRA: Biologically-Inspired Continual Learning

This repository contains the implementation of **Neuro-LoRA**, a novel continual learning method that combines the parameter-efficient advantages of Low-Rank Adaptation (LoRA) with synaptic consolidation principles from neuroscience.

## 🧠 Core Innovation

Neuro-LoRA introduces four key components inspired by biological neural systems:

1. **Synaptic Importance Projection (SIP)**: Extracts important directions from LoRA updates using SVD
2. **Synaptic Gradient Projection (SGP)**: Projects gradients to orthogonal complement to prevent interference
3. **Homeostatic Plasticity Regularization**: Encourages diverse neuron usage through entropy-based regularization
4. **Sleep-Phase Consolidation**: Self-distillation to stabilize learned representations

## 📁 Implementation Structure

### Core Files

- **`utils/neuro_utils.py`**: Core mathematical functions for Neuro-LoRA
- **`methods/neuro_lora.py`**: Main Neuro-LoRA method implementation
- **`models/vit_lora.py`**: Enhanced Vision Transformer with LoRA support
- **`configs/cifar100_neuro_lora.json`**: Configuration for CIFAR-100 experiments

### Key Functions

#### Subspace Management
```python
# Extract important directions from LoRA updates
S_new = extract_subspace_from_BA(B, A, k=4)

# Merge with cumulative subspace
S_cum = merge_cumulative_subspace(S_prev, S_new, K_max=64)

# Project gradients to orthogonal complement
gB_proj = project_grad_B(gB, S_cum)
```

#### Plasticity Regularization
```python
# Compute homeostatic plasticity loss
plast_loss = compute_plasticity_loss(lora_activations)
```

#### Sleep Phase
```python
# Self-distillation for representation stabilization
sleep_phase_distill(teacher_model, student_model, noise_loader)
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install torch torchvision timm
```

### 2. Run Smoke Tests
```bash
python tests/test_neuro_lora_smoke.py
```

### 3. Train on CIFAR-100
```bash
python main.py --config configs/cifar100_neuro_lora.json
```

## ⚙️ Configuration

### Neuro-LoRA Parameters

```json
{
    "neuro_lora": {
        "enabled": true,
        "k_per_task": 4,        // Directions extracted per task
        "K_max": 64,            // Maximum cumulative directions
        "lambda_plast": 0.1,    // Plasticity loss weight
        "sleep_epochs": 0,      // Sleep phase epochs (0 = disabled)
        "sleep_bs": 64,         // Sleep phase batch size
        "sleep_batches": 10,    // Number of sleep batches
        "sleep_lr": 1e-4        // Sleep phase learning rate
    }
}
```

### Key Hyperparameters

- **`k_per_task`**: Number of important directions to extract from each task's LoRA updates
- **`K_max`**: Maximum number of cumulative directions to maintain (memory vs. performance trade-off)
- **`lambda_plast`**: Weight for homeostatic plasticity regularization
- **`sleep_epochs`**: Number of epochs for sleep-phase consolidation (0 to disable)

## 🔬 Mathematical Foundation

### Synaptic Importance Projection (SIP)

For LoRA matrices A ∈ ℝ^(r×d) and B ∈ ℝ^(d×r), we compute:

1. Small matrix: M = A @ B ∈ ℝ^(r×r)
2. SVD: M = ŨΣṼ^T
3. Important directions: S_new = B @ Ũ[:, :k] ∈ ℝ^(d×k)

### Synaptic Gradient Projection (SGP)

Project gradient g_B to orthogonal complement of cumulative subspace S:

g_B^proj = g_B - S(S^T g_B)

### Homeostatic Plasticity

Compute entropy-based loss on LoRA activations h:

L_plasticity = -∑_i p̃_i log(p̃_i)

where p̃_i = (h̄_i + ε) / ∑_j(h̄_j + ε)

## 📊 Expected Benefits

1. **Better Catastrophic Forgetting Prevention**: Explicit subspace protection
2. **More Stable Training**: Gradient projection prevents interference
3. **Biological Plausibility**: Synaptic consolidation principles
4. **Memory Efficiency**: Low-rank representation with controlled growth

## 🧪 Experimental Setup

### Datasets
- **CIFAR-100**: 20 tasks, 5 classes per task
- **ImageNet-R**: 20 tasks, 10 classes per task

### Baselines
- LoRA-Sub-DRS (original method)
- CoDA
- L2P
- Dual-Prompt

### Metrics
- Top-1 accuracy per task
- Average accuracy across all tasks
- Forgetting measure

## 🔧 Debugging Tips

### Common Issues

1. **Memory Issues**: Reduce `K_max` or `k_per_task`
2. **Training Instability**: Increase `lambda_plast` or reduce learning rate
3. **Poor Performance**: Enable sleep phase with `sleep_epochs > 0`

### Logging

The implementation includes comprehensive logging:
- Subspace shapes and sizes
- Gradient norms before/after projection
- Plasticity loss values
- Sleep phase metrics

## 📈 Performance Optimization

### Memory Optimization
- Subspaces stored on CPU, transferred to GPU during computation
- Automatic truncation when K_max is reached
- Efficient SVD on small r×r matrices

### Computational Optimization
- Gradient projection only for tasks > 0
- Optional sleep phase (can be disabled)
- Batch processing for subspace extraction

## 🤝 Contributing

To extend Neuro-LoRA:

1. **New Subspace Extraction Methods**: Modify `extract_subspace_from_BA()`
2. **Alternative Plasticity Measures**: Extend `compute_plasticity_loss()`
3. **Different Sleep Mechanisms**: Implement new `sleep_phase_*()` functions

## 📚 References

- Original LoRA-Sub-DRS paper
- Synaptic consolidation literature
- Continual learning benchmarks

## 📄 License

This implementation follows the same license as the original LoRA-Sub-DRS repository.
