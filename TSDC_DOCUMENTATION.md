# Task Similarity-aware Drift Correction (TSDC)

## Overview

**Task Similarity-aware Drift Correction (TSDC)** is an advanced feature in Neuro-LoRA that dynamically adjusts gradient projection strength based on the similarity between current and previous tasks. This prevents over-regularization when learning new, dissimilar tasks while maintaining strong protection for similar tasks.

## Motivation

Traditional continual learning methods protect all previously learned subspaces equally, which can be suboptimal:

- **Similar tasks**: Require strong protection to prevent interference
- **Dissimilar tasks**: Can learn more freely without affecting previous knowledge

TSDC addresses this by quantifying task similarity and adjusting projection strength accordingly.

## Mathematical Foundation

### Task Similarity Computation

TSDC computes similarity between current task's ΔW and previous tasks' ΔW matrices:

#### Cosine Similarity

```python
# For matrices ΔW_current and ΔW_prev
curr_flat = ΔW_current.flatten()
prev_flat = ΔW_prev.flatten()

curr_norm = torch.norm(curr_flat)
prev_norm = torch.norm(prev_flat)

sim = torch.dot(curr_flat, prev_flat) / (curr_norm * prev_norm)
sim_normalized = (sim + 1) / 2  # Convert from [-1, 1] to [0, 1]
```

#### Frobenius Similarity

```python
# Normalize matrices by Frobenius norm
curr_normalized = ΔW_current / torch.norm(ΔW_current, p='fro')
prev_normalized = ΔW_prev / torch.norm(ΔW_prev, p='fro')

# Compute similarity as dot product
sim = torch.dot(curr_normalized.flatten(), prev_normalized.flatten())
sim_normalized = (sim + 1) / 2
```

### Similarity-Aware Projection

The projection strength α is computed as:

```python
similarity = average_similarity_with_previous_tasks()
alpha = (similarity ** decay) ** clamp(min=min_strength, max=1.0)

# Apply similarity-aware projection
g_proj = g - α * (S @ (S.T @ g))
```

Where:

- `decay`: Controls sensitivity to similarity differences
- `min_strength`: Minimum projection strength (default: 0.1)
- `S`: Cumulative subspace from previous tasks

## Implementation Details

### Core Functions

#### `compute_task_similarity(delta_w_current, delta_w_history, method)`

- **Purpose**: Compute similarity between current and previous tasks
- **Parameters**:
  - `delta_w_current`: Current task's ΔW matrix (d, d)
  - `delta_w_history`: List of previous tasks' ΔW matrices
  - `method`: "cosine" or "frobenius"
- **Returns**: Average similarity score [0, 1]

#### `project_grad_similarity_aware(gA, gB, A, B, S, delta_w_current, delta_w_history, ...)`

- **Purpose**: Apply similarity-aware gradient projection
- **Parameters**:
  - `gA, gB`: Gradients of A and B matrices
  - `A, B`: Current A and B matrices
  - `S`: Cumulative subspace
  - `delta_w_current`: Current task's ΔW matrix
  - `delta_w_history`: List of previous tasks' ΔW matrices
  - `similarity_method`: "cosine" or "frobenius"
  - `similarity_decay`: Power to apply to similarity (β)
  - `min_projection_strength`: Minimum projection strength
- **Returns**: `(gA_proj, gB_proj, alpha)`

### ΔW History Management

#### `save_delta_w_history(delta_w_history, save_path)`

- Saves ΔW matrices from previous tasks to disk
- Converts tensors to numpy arrays for compatibility

#### `load_delta_w_history(load_path, device)`

- Loads ΔW history from disk
- Handles PyTorch version compatibility

## Configuration

### Basic Configuration

```json
{
  "neuro_lora": {
    "tsdc_enabled": true,
    "similarity_method": "cosine",
    "similarity_decay": 1.5,
    "min_projection_strength": 0.1
  }
}
```

### Advanced Configuration

```json
{
  "neuro_lora": {
    "enabled": true,
    "k_per_task": 12,
    "K_max": 160,
    "lambda_plast": 0.2,
    "sleep_epochs": 2,
    "sleep_bs": 64,
    "sleep_batches": 20,
    "sleep_lr": 1e-4,
    "use_bi_directional": true,
    "bi_directional_method": "delta_w_projected",
    "adaptive_k_enabled": true,
    "energy_threshold": 0.95,
    "tsdc_enabled": true,
    "similarity_method": "cosine",
    "similarity_decay": 1.5,
    "min_projection_strength": 0.1
  }
}
```

## Parameter Tuning

### `similarity_method`

- **"cosine"**: Better for capturing directional similarity
- **"frobenius"**: Better for capturing magnitude similarity
- **Recommendation**: Use "cosine" for most cases

### `similarity_decay`

- **β = 0.5**: Less sensitive to similarity differences
- **β = 1.0**: Linear relationship
- **β = 1.5**: Emphasizes differences (recommended)
- **β = 2.0**: Very sensitive to differences

### `min_projection_strength`

- **0.0**: No minimum protection (risky)
- **0.1**: Always protect at least 10% (recommended)
- **0.2**: Conservative protection

## Usage Examples

### Basic Usage

```python
# Enable TSDC in configuration
config = {
    "neuro_lora": {
        "tsdc_enabled": True,
        "similarity_method": "cosine",
        "similarity_decay": 1.5,
        "min_projection_strength": 0.1
    }
}

# Run training
python main.py --config configs/cifar100_neuro_lora_advanced.json
```

### Custom Similarity Computation

```python
from utils.neuro_utils import compute_task_similarity

# Compute similarity manually
current_delta_w = B @ A
similarity = compute_task_similarity(
    current_delta_w,
    delta_w_history,
    method="cosine"
)
print(f"Task similarity: {similarity:.3f}")
```

## Performance Characteristics

### Computational Overhead

- **Similarity computation**: O(d² × num_previous_tasks)
- **ΔW history storage**: O(d² × num_tasks)
- **Overall**: Minimal overhead compared to training time

### Memory Usage

- **ΔW history**: ~d² × num_tasks × 4 bytes (float32)
- **Example**: 768² × 20 tasks = ~47MB for 20 tasks

### Expected Benefits

- **Similar tasks**: 20-50% stronger protection
- **Dissimilar tasks**: 30-70% weaker protection
- **Overall**: Better balance between learning and forgetting

## Integration with Other Features

### With ΔW-Projected Bi-directional Gradient Projection

TSDC works seamlessly with the ΔW-projected method:

```python
# TSDC automatically uses ΔW-projected method when enabled
if tsdc_enabled:
    gA_proj, gB_proj, alpha = project_grad_similarity_aware(...)
else:
    gA_proj, gB_proj = project_grad_delta_w(...)
```

### With Adaptive-k Subspace Building

TSDC complements adaptive-k by:

- Using adaptive subspace sizes for similarity computation
- Maintaining consistent protection regardless of subspace size

### With Sleep-Phase Consolidation

TSDC enhances sleep-phase consolidation by:

- Preserving task-specific learning patterns
- Allowing selective forgetting during consolidation

## Testing

### Unit Tests

```bash
# Test TSDC functionality
python tests/test_tsdc.py

# Test integration with other features
python tests/test_delta_w_projection.py
python tests/test_neuro_lora_simple.py
```

### Expected Test Results

- ✅ Task similarity computation works correctly
- ✅ Similarity-aware gradient projection adapts to task similarity
- ✅ Similarity decay parameter affects projection strength
- ✅ ΔW history save/load functionality works
- ✅ Edge cases are handled properly

## Troubleshooting

### Common Issues

#### Low Similarity Scores

- **Cause**: Tasks are genuinely different
- **Solution**: Lower `similarity_decay` or `min_projection_strength`

#### High Memory Usage

- **Cause**: Too many tasks in history
- **Solution**: Implement history truncation or compression

#### Numerical Instability

- **Cause**: Very small ΔW matrices
- **Solution**: Add normalization or minimum magnitude threshold

### Debugging

Enable logging to monitor TSDC behavior:

```python
# Similarity scores are logged every 100 batches
logging.info(f"Task similarity: {alpha:.3f} (task {cur_task})")
```

## Research Applications

### Ablation Studies

Compare TSDC variants:

- Fixed projection strength (baseline)
- Cosine similarity vs Frobenius similarity
- Different decay values
- Different minimum projection strengths

### Task Similarity Analysis

Analyze task relationships:

- Compute similarity matrices between all tasks
- Identify task clusters
- Understand forgetting patterns

### Performance Evaluation

Measure TSDC impact:

- Accuracy on previous tasks
- Learning speed on new tasks
- Overall continual learning performance

## Future Enhancements

### Potential Improvements

1. **Dynamic similarity thresholds**: Adapt based on task difficulty
2. **Multi-scale similarity**: Consider different granularities
3. **Task clustering**: Group similar tasks for better protection
4. **Online similarity learning**: Learn similarity metrics from data

### Research Directions

1. **Theoretical analysis**: Convergence guarantees with TSDC
2. **Empirical evaluation**: Large-scale experiments
3. **Comparison studies**: Against other adaptive methods
4. **Neuroscience validation**: Biological plausibility

## Conclusion

TSDC provides a principled approach to adaptive continual learning by:

- **Quantifying task relationships** through ΔW similarity
- **Adapting protection strength** based on task similarity
- **Maintaining learning flexibility** for dissimilar tasks
- **Preserving knowledge** for similar tasks

This makes Neuro-LoRA more robust and efficient for real-world continual learning scenarios with varying task relationships.
