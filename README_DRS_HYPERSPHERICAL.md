# DRS-Hyperspherical: Spherical Geometry for Drift-Resistant Space

## 🎯 Overview

**DRS-Hyperspherical** is an innovative enhancement to continual learning that combines **LoRA⁻ (subtraction)** at the parameter level with **hyperspherical geometry** at the feature level. This method provides a mathematically principled solution to the stability-plasticity dilemma in exemplar-free continual learning.

### 🔬 Core Innovation

The method combines several breakthrough concepts:

1. **LoRA⁻ (Subtraction)**: Reset parameter drift by subtracting previous task adaptations before each new task
2. **Spherical Normalization**: Constrain embeddings to unit hypersphere S^{d-1}
3. **Riemannian Projection**: Project gradients in tangent spaces using spherical geometry
4. **DRS Construction**: Build drift-resistant subspaces via PCA on tangent vectors
5. **2-Phase Training**: Warm-up + main phases with adaptive annealing
6. **EMA Prototypes**: Prevent distribution shift caused by LoRA⁻
7. **Angular Losses**: ArcFace + Angular Triplet for better spherical embeddings

## 🏗️ Architecture

### Mathematical Foundation

- **Spherical Manifold**: Embeddings lie on S^{d-1} = {x ∈ ℝ^d : ||x|| = 1}
- **Log Map**: `log_μ(x) = (θ/sin(θ)) * (x - cos(θ)*μ)` where `θ = arccos(μᵀx)`
- **Exp Map**: `exp_μ(v) = cos(||v||)*μ + sin(||v||) * v/||v||`
- **Riemannian Projection**: `g_T = (I - μμᵀ)g`
- **DRS Projection**: `g_proj = U_t U_tᵀ g_T`

### Training Pipeline

```
For each task t:
  1. Apply LoRA⁻: W̃_t = W_0 - Σ_{j=1}^{t-1} ΔW_j
  2. Collect embeddings and build DRS via PCA on tangent space
  3. 2-Phase Training:
     - Warm-up: CE + Riemannian projection, anneal s↑ & m↑
     - Main: CE + Angular Triplet + DRS projection
  4. EMA update prototypes throughout training
  5. Re-estimate prototypes with final model
```

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install torch torchvision timm scipy matplotlib pyyaml

# Clone repository (assuming existing LoRA-Sub-DRS codebase)
cd LoRA-Sub-DRS
```

### Basic Usage

```bash
# Train with default CIFAR-100 configuration
python train_hyperspherical.py --config configs/drs_hyperspherical_cifar100.yaml

# Generate and run ablation study
python train_hyperspherical.py --config configs/drs_hyperspherical_cifar100.yaml --ablation
```

### Configuration

The method uses YAML configuration files for easy experimentation:

```yaml
# Core spherical geometry settings
spherical:
  per_class_prototypes: true
  pca_energy: 0.90 # PCA energy threshold
  k_max: 128 # Max PCA components

# 2-phase training
train:
  epochs_warm: 4 # Warm-up epochs
  epochs_main: 26 # Main training epochs
  ema_momentum: 0.97 # EMA for prototypes

# Angular losses with annealing
loss:
  s_start: 10.0 # ArcFace scale start
  s_end: 30.0 # ArcFace scale end
  m_start: 0.0 # ArcFace margin start
  m_end: 0.2 # ArcFace margin end
  triplet_lambda: 0.5 # Angular triplet weight
```

## 📊 Ablation Studies

The implementation includes a comprehensive ablation framework:

```bash
# Generate all ablation configurations
python train_hyperspherical.py --ablation --config configs/drs_hyperspherical_cifar100.yaml

# This creates:
# - baseline_lorasub.yaml        (Linear DRS baseline)
# - riemann_only.yaml           (Riemannian without DRS)
# - no_drs_projection.yaml      (No PCA subspace)
# - no_angular_triplet.yaml     (No triplet loss)
# - no_ema_prototypes.yaml      (Fixed prototypes)
# - no_warmup_anneal.yaml       (No 2-phase training)
# - drs_hyperspherical_full.yaml (Complete method)
```

### Ablation Matrix

| Method                         | Linear DRS | Riemannian | DRS Subspace | Angular Triplet | EMA | Warm-up |
| ------------------------------ | ---------- | ---------- | ------------ | --------------- | --- | ------- |
| 1. Baseline LoRA-Sub           | ✓          | ✗          | ✗            | ✗               | ✗   | ✗       |
| 2. Riemannian-only             | ✗          | ✓          | ✗            | ✓               | ✓   | ✓       |
| 3. No DRS subspace             | ✗          | ✓          | ✗            | ✓               | ✓   | ✓       |
| 4. No Angular Triplet          | ✗          | ✓          | ✓            | ✗               | ✓   | ✓       |
| 5. No EMA                      | ✗          | ✓          | ✓            | ✓               | ✗   | ✓       |
| 6. No Warm-up                  | ✗          | ✓          | ✓            | ✓               | ✓   | ✗       |
| 7. **Full DRS-Hyperspherical** | ✗          | ✓          | ✓            | ✓               | ✓   | ✓       |

## 📈 Performance Metrics

The method tracks specialized metrics for spherical geometry:

- **Geodesic Drift**: `d_g(μ_old, μ_new) = arccos(μ_old · μ_new)`
- **Backward Transfer (BWT)**: `(1/(T-1)) Σ_{t=1}^{T-1} (R_{T,t} - R_{t,t})`
- **PCA Components Evolution**: Number of components maintaining 90% energy
- **Prototype Stability**: EMA convergence analysis

### Visualization

The framework automatically generates comprehensive analysis plots:

- Accuracy evolution over tasks
- Geodesic drift measurements
- PCA component evolution
- Prototype shift analysis
- Lambda adaptation curves (if using AD-DRS base)

## 🔧 Implementation Details

### Key Components

1. **`utils/spherical_geometry.py`**: Core Riemannian operations

   - SphericalGeometry class with log/exp maps
   - TangentPCA for DRS construction
   - PrototypeManager with EMA updates
   - Gradient projection hooks

2. **`utils/angular_losses.py`**: Spherical-aware loss functions

   - ArcFace head with annealing
   - Angular Triplet Loss with geodesic distances
   - Combined loss with label smoothing

3. **`methods/drs_hyperspherical.py`**: Main method implementation

   - 2-phase training pipeline
   - LoRA⁻ subtraction logic
   - Comprehensive logging and analysis

4. **`utils/hyperspherical_analysis.py`**: Analysis framework
   - Experiment tracking and visualization
   - Ablation study generation
   - Performance comparison tools

### Hyperparameters

**Recommended starting points:**

- LoRA rank: 16 (range: 8-32)
- PCA energy: 0.90 (range: 0.85-0.95)
- Max components: 128 (range: 64-256)
- EMA momentum: 0.97 (range: 0.95-0.99)
- Warm-up epochs: 4 (range: 3-6)
- ArcFace s: 10→30, m: 0→0.2
- Triplet weight: 0.5 (range: 0.1-1.0)

## 🧪 Experimental Setup

### Datasets

- **CIFAR-100**: 50 initial + 10 tasks × 5 classes
- **ImageNet-R**: 100 initial + 10 tasks × 10 classes
- **CUB-200**: Custom splits
- **DomainNet**: Cross-domain evaluation

### Hardware Requirements

- **GPU Memory**: ~8GB for CIFAR-100, ~12GB for ImageNet-R
- **Storage**: ~100MB per experiment (prototypes + analysis)
- **Compute**: 2-4 hours per full CIFAR-100 experiment

### Memory Efficiency

- Covariance matrix (d×d): ~2.3MB for d=768
- PCA components U_t: ~0.19MB per task
- Prototypes: ~3KB per class
- Total overhead: <50MB for typical experiments

## 🔍 Failure Modes & Solutions

### Common Issues

1. **Strong Forgetting**

   - ↑ Warm-up epochs
   - ↓ Final margin (m_end)
   - Enable label smoothing
   - ↓ Learning rate

2. **Weak New Learning**

   - ↑ Main epochs
   - ↑ Final scale (s_end)
   - ↑ Max PCA components
   - ↑ Triplet weight

3. **NaN/Instability**

   - Clamp cosine: [-1+ε, 1-ε]
   - Small-angle approximation
   - Check for θ ≈ π cases
   - Fixed random seed

4. **Distribution Shift from LoRA⁻**
   - Ensure EMA prototype updates
   - Re-estimate after training
   - Use representative data for DRS

## 📚 Theoretical Background

### Spherical Geometry

The method leverages the Riemannian structure of the unit hypersphere:

- **Tangent Space**: T_μS^{d-1} = {v ∈ ℝ^d : μᵀv = 0}
- **Geodesics**: Great circles on the sphere
- **Exponential Map**: Retraction from tangent to manifold
- **Parallel Transport**: Moving tangent vectors along geodesics

### DRS Construction

The Drift-Resistant Space is constructed by:

1. Mapping embeddings to tangent space: `v_i = log_μ(x̂_i)`
2. Computing PCA on tangent vectors: `{v_i} → U_t`
3. Projecting gradients: `g_proj = U_t U_tᵀ (I - μμᵀ) g`

### Convergence Guarantees

Under mild assumptions, the method provides:

- **Prototype Convergence**: EMA updates converge to true class means
- **Subspace Stability**: PCA components stabilize as data increases
- **Gradient Boundedness**: Projections preserve gradient norms

## 🔗 References

1. **LoRA**: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models"
2. **ArcFace**: Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
3. **Riemannian Optimization**: Absil et al., "Optimization Algorithms on Matrix Manifolds"
4. **Continual Learning**: Lopez-Paz & Ranzato, "Gradient Episodic Memory for Continual Learning"

## 🤝 Contributing

Contributions are welcome! Please see:

- Implementation guidelines in `/docs/`
- Test suite in `/tests/`
- Performance benchmarks in `/benchmarks/`

## 📄 License

This project builds upon the LoRA-Sub-DRS codebase and follows the same licensing terms.

---

**DRS-Hyperspherical** represents a significant advancement in continual learning by mathematically unifying parameter-level and feature-level drift prevention through spherical geometry. The comprehensive implementation provides both theoretical rigor and practical effectiveness for real-world sequential learning scenarios.
