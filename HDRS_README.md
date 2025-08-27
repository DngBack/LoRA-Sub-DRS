# Hyperspherical Drift-Resistant Space (HDRS) Implementation

This implementation extends the LoRA-Sub-DRS method with **Riemannian geometry** on **hyperspherical manifolds** for improved continual learning. Instead of operating in Euclidean space, we constrain LoRA parameters to Riemannian manifolds and use geodesic-aware operations.

## Key Innovations

### 1. **Riemannian LoRA Parameters**
- **A matrices**: Constrained to **Stiefel manifold** (orthonormal columns) for stable basis learning
- **B matrices**: Constrained to **Sphere manifold** (unit norm columns) for directional consistency
- **Manifold-aware optimization**: Uses Riemannian gradients with proper retraction

### 2. **Hyperspherical Subtraction**
Instead of Euclidean subtraction `W = W₀ - Σ BⱼAⱼ`, we perform:
- **Manifold subtraction**: Use exponential map with negative tangent vectors
- **Direction-preserving**: Maintains geometric structure while "unlearning" previous tasks
- **Controllable magnitude**: Hyperparameter `η` controls subtraction strength

### 3. **Tangent Space PCA for DRS**
- Compute **Karcher mean** (Riemannian center) of normalized features
- Map features to **tangent space** via logarithmic map
- Perform **PCA in tangent space** → more principled than Euclidean PCA on sphere
- **Geodesic-aware projection**: Preserves spherical geometry

### 4. **Cosine/Geodesic Metrics**
- **Cosine similarity** for feature comparison
- **Geodesic distance** for prototype-based classification
- **Angular loss functions**: Augmented Triplet Loss with cosine/geodesic distances

## File Structure

```
models/
├── riemannian_lora.py     # Core Riemannian LoRA components
    ├── ManualSphere       # Sphere manifold operations (log/exp maps)
    ├── ManualStiefel      # Stiefel manifold operations (QR retraction)
    ├── RiemannianLoRALayer# LoRA layer with manifold constraints
    ├── RiemannianAttention# Attention with Riemannian LoRA
    └── compute_tangent_pca# Tangent space PCA implementation

methods/
├── hdrs.py               # Main HDRS method
    ├── HypersphericalDRS # Main class inheriting BaseLearner
    ├── Manifold training # Riemannian optimization setup
    ├── DRS computation   # Tangent space projection
    └── Prototype building# Spherical prototype classification

configs/
├── hdrs_cifar100.json   # Configuration for HDRS method

train_hyperspherical.py  # Test and demonstration script
```

## Usage

### 1. **Basic Testing**
```bash
python train_hyperspherical.py
```
This runs component tests and demonstrations including:
- Sphere/Stiefel manifold operations
- Riemannian LoRA layers
- Tangent space PCA
- Manifold subtraction
- Simple training example

### 2. **Full Training (CIFAR-100)**
```bash
python main.py --config configs/hdrs_cifar100.json
```

### 3. **Custom Configuration**
Create your own config file with HDRS-specific parameters:

```json
{
    "model_name": "hdrs",
    "manifold_A": "stiefel",         // "stiefel" or "sphere" for A matrices
    "manifold_B": "sphere",          // "sphere" or "stiefel" for B matrices  
    "use_geoopt": false,             // Use geoopt library (requires installation)
    "eta_subtraction": 0.1,          // LoRA subtraction strength
    "use_tangent_pca": true,         // Use tangent space PCA vs standard PCA
    "drs_energy_threshold": 0.99,    // Energy threshold for PCA components
    "drs_max_components": 64         // Max number of DRS components
}
```

## Implementation Details

### **Manifold Operations**

#### Sphere Operations (`ManualSphere`)
- **Exponential map**: `exp_μ(v) = cos(‖v‖)μ + sin(‖v‖)(v/‖v‖)`
- **Logarithmic map**: `log_μ(x) = (θ/sin θ)(x - cos θ · μ)` where `θ = arccos(μ·x)`
- **Tangent projection**: `proj_μ(v) = v - (μ·v)μ`

#### Stiefel Operations (`ManualStiefel`)
- **QR retraction**: Ensures orthonormal columns via QR decomposition
- **Tangent projection**: `proj_X(V) = V - X((X^T V + V^T X)/2)`

### **Riemannian Subtraction**
```python
def riemannian_subtraction(current, cumulative, eta=0.1):
    for each row i:
        w_curr = normalize(current[i])           # Current direction
        v_cum = cumulative[i]                    # Cumulative effect
        tau = project_to_tangent(w_curr, v_cum)  # Project to tangent
        result[i] = exp_map(w_curr, -eta * tau)  # Exponential map with negative direction
    return result
```

### **Tangent Space PCA**
```python
def compute_tangent_pca(points, k=64):
    mu = karcher_mean(points)                    # Riemannian center
    tangent_vecs = [log_map(mu, p) for p in points]  # Map to tangent space
    cov_matrix = cov(tangent_vecs)              # Covariance in tangent space
    eigenvals, eigenvecs = eigh(cov_matrix)     # Standard PCA in tangent space
    return mu, eigenvecs[:, :k]                 # Return center and basis
```

## Dependencies

### **Required**
- PyTorch ≥ 1.9.0
- NumPy
- SciPy (for some distance computations)

### **Optional** 
- **geoopt**: For advanced Riemannian optimization
  ```bash
  pip install geoopt
  ```
  Set `"use_geoopt": true` in config to enable.

### **Fallback**
If geoopt is not available, the implementation falls back to manual Riemannian operations with standard PyTorch optimizers.

## Mathematical Background

### **Why Hyperspherical?**
1. **Directional Learning**: Many features are naturally directional (e.g., attention patterns)
2. **Scale Invariance**: Sphere constrains focus to directions, not magnitudes
3. **Geodesic Distances**: More meaningful for angular similarity
4. **Drift Resistance**: Tangent space PCA preserves spherical geometry

### **Manifold Constraints**
- **Stiefel St(n,p)**: `{X ∈ ℝⁿˣᵖ : X^T X = I_p}` (orthonormal columns)
- **Sphere S^(n-1)**: `{x ∈ ℝⁿ : ‖x‖ = 1}` (unit norm)

### **Riemannian Optimization**
- **Riemannian gradient**: Project Euclidean gradient to manifold tangent space  
- **Retraction**: Map tangent vector back to manifold (e.g., normalization, QR)
- **Parallel transport**: Move tangent vectors along geodesics (advanced)

## Experimental Comparisons

### **Baselines to Compare**
1. **Original LoRA-DRS**: Standard Euclidean approach
2. **Norm-SVD-DRS**: Normalize features before SVD (simple baseline)
3. **HDRS (This work)**: Full Riemannian approach

### **Metrics**
- **Accuracy**: Standard continual learning metrics
- **Drift**: Geodesic distance between prototypes over time
- **Computational cost**: Training time and memory usage

### **Expected Results**
- **Better drift resistance** on long task sequences
- **Improved angular stability** of learned representations  
- **More robust to feature scale variations**
- **Moderate computational overhead** (manageable with optimizations)

## Troubleshooting

### **Common Issues**
1. **Numerical instability**: Use `eps=1e-8` in log/exp maps near poles
2. **Slow convergence**: Reduce Riemannian learning rates (use 0.5× Euclidean rates)
3. **Memory usage**: Use randomized SVD for large feature matrices
4. **Geoopt compatibility**: Ensure PyTorch version compatibility

### **Performance Tuning**
- **Batch size**: May need smaller batches for Riemannian operations
- **Learning rates**: Start with lower LR for manifold parameters
- **Subtraction strength** `η`: Start with 0.1, tune in range [0.05, 0.5]
- **DRS components**: Balance between performance and computational cost

## Citation

If you use this implementation, please cite the original LoRA-Sub-DRS paper and consider acknowledging this Riemannian extension:

```bibtex
@inproceedings{lorasub_drs_2025,
  title={LoRA Subtraction for Drift-Resistant Space in Exemplar-Free Continual Learning},
  author={[Original Authors]},
  booktitle={CVPR},
  year={2025}
}
```

## Contributing

This implementation provides a foundation for Riemannian continual learning research. Potential extensions:
- **Advanced manifolds**: Grassmann, hyperbolic spaces
- **Learned metrics**: Trainable Riemannian metrics
- **Efficient implementations**: GPU-optimized manifold operations
- **Integration with other CL methods**: Apply to rehearsal-based methods
