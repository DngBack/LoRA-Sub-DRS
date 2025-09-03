#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import os
import math
from copy import deepcopy

EPS = 1e-12


def extract_subspace_from_BA(B: torch.Tensor, A: torch.Tensor, k: int):
    """
    Efficiently extract top-k left singular vectors of Delta = B @ A
    using small SVD on r x r matrix M = A @ B.

    Args:
        B: (d, r) - LoRA B matrix
        A: (r, d) - LoRA A matrix
        k: number of top directions to extract

    Returns:
        S_new: (d, k) with orthonormal columns
    """
    assert B.ndim == 2 and A.ndim == 2
    d, r = B.shape

    # Small r x r matrix for efficient SVD
    M = A @ B  # (r, r)

    try:
        U_small, Svals, Vt = torch.linalg.svd(M)  # U_small: (r, r)
    except Exception:  # fallback for older torch versions
        U_small, Svals, Vt = torch.svd(M)

    k = min(k, r)
    S_new = B @ U_small[:, :k]  # (d, k)

    # Orthonormalize using QR decomposition
    Q, _ = torch.linalg.qr(S_new)
    S_new = Q[:, :k]

    return S_new


def merge_cumulative_subspace(S_prev: torch.Tensor, S_new: torch.Tensor, K_max: int):
    """
    Merge S_prev (d, Kprev) and S_new (d, k) into orthonormal S_cum (d, K_keep<=K_max)

    Args:
        S_prev: previous cumulative subspace or None
        S_new: new subspace from current task
        K_max: maximum allowed dimensions

    Returns:
        S_cum: (d, K_keep) orthonormal cumulative subspace
    """
    if S_prev is None or S_prev.numel() == 0:
        S_merge = S_new
    else:
        S_merge = torch.cat([S_prev, S_new], dim=1)  # (d, Kprev + k)

    # QR orthogonalization
    Q, _ = torch.linalg.qr(S_merge)
    K_keep = min(Q.shape[1], K_max)
    S_cum = Q[:, :K_keep].contiguous()

    return S_cum


def project_grad_B(gB: torch.Tensor, S: torch.Tensor):
    """
    Project gradient of B (d, r) to orthogonal complement of subspace S (d, K)

    Args:
        gB: gradient of B matrix (d, r)
        S: cumulative subspace (d, K)

    Returns:
        gB_proj: projected gradient of same shape
    """
    if S is None or S.numel() == 0:
        return gB

    # Project onto orthogonal complement: gB - S(S^T gB)
    coef = torch.matmul(S.T, gB)  # (K, r)
    gB_proj = gB - torch.matmul(S, coef)  # (d, r)

    return gB_proj


def project_grad_bi_directional(
    gA: torch.Tensor,
    gB: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    S: torch.Tensor,
):
    """
    Bi-directional gradient projection for LoRA matrices A and B

    Instead of just projecting gB, we project the total gradient of ΔW = BA
    to better protect the learned subspace.

    Args:
        gA: gradient of A matrix (r, d)
        gB: gradient of B matrix (d, r)
        A: current A matrix (r, d)
        B: current B matrix (d, r)
        S: cumulative subspace (d, K)

    Returns:
        gA_proj: projected gradient of A
        gB_proj: projected gradient of B
    """
    if S is None or S.numel() == 0:
        return gA, gB

    # Compute approximate total gradient of ΔW = BA
    # ∇(ΔW) ≈ B⋅∇A + ∇B⋅A
    # This gives us the total change in the learned direction

    # For now, use the simple method as it's more stable
    # The full method requires more sophisticated numerical analysis
    return project_grad_bi_directional_simple(gA, gB, A, B, S)

    return gA_proj, gB_proj


def project_grad_bi_directional_simple(
    gA: torch.Tensor,
    gB: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    S: torch.Tensor,
):
    """
    Simplified bi-directional gradient projection

    Projects gradients of both A and B separately to orthogonal complement of S
    This is simpler and more stable than the full bi-directional approach

    Args:
        gA: gradient of A matrix (r, d)
        gB: gradient of B matrix (d, r)
        A: current A matrix (r, d)
        B: current B matrix (d, r)
        S: cumulative subspace (d, K)

    Returns:
        gA_proj: projected gradient of A
        gB_proj: projected gradient of B
    """
    if S is None or S.numel() == 0:
        return gA, gB

    # Project gB to orthogonal complement of S (as before)
    coef_B = torch.matmul(S.T, gB)  # (K, r)
    gB_proj = gB - torch.matmul(S, coef_B)  # (d, r)

    # Project gA to orthogonal complement of S in the d-dimensional space
    # The effect of gA in d-space is B⋅gA, so we need to ensure this is orthogonal to S
    # We can't directly project gA since it's (r, d), but we can ensure B⋅gA_proj is orthogonal

    # Method 1: Direct projection of the effect
    # Compute the effect of gA in d-space: B⋅gA
    gA_effect = torch.matmul(B, gA)  # (d, d)

    # Project this effect to orthogonal complement of S
    coef_A_effect = torch.matmul(S.T, gA_effect)  # (K, d)
    gA_effect_proj = gA_effect - torch.matmul(S, coef_A_effect)  # (d, d)

    # Now we need to find gA_proj such that B⋅gA_proj = gA_effect_proj
    # Since B is (d, r), we can use pseudo-inverse: gA_proj = B^+ ⋅ gA_effect_proj
    B_pinv = torch.pinverse(B)  # (r, d)
    gA_proj = torch.matmul(B_pinv, gA_effect_proj)  # (r, d)

    return gA_proj, gB_proj


def project_grad_delta_w(
    gA: torch.Tensor,
    gB: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    S: torch.Tensor,
):
    """
    ΔW-Projected Bi-directional Gradient Projection

    Instead of projecting A and B gradients separately, we:
    1. Compute the total gradient of ΔW = BA
    2. Project this total gradient out of the protected subspace
    3. Backsolve to get projected A and B gradients

    Args:
        gA: gradient of A matrix (r, d)
        gB: gradient of B matrix (d, r)
        A: current A matrix (r, d)
        B: current B matrix (d, r)
        S: cumulative subspace (d, K)

    Returns:
        gA_proj: projected gradient of A
        gB_proj: projected gradient of B
    """
    if S is None or S.numel() == 0:
        return gA, gB

    # Step 1: Compute total gradient of ΔW = BA
    # ∇(ΔW) = B⋅∇A + ∇B⋅A
    grad_delta_w = torch.matmul(B, gA) + torch.matmul(gB, A)  # (d, d)

    # Step 2: Project ∇(ΔW) out of the protected subspace
    # proj_grad = grad_delta_w - S(S^T grad_delta_w)
    coef = torch.matmul(S.T, grad_delta_w)  # (K, d)
    proj_grad_delta_w = grad_delta_w - torch.matmul(S, coef)  # (d, d)

    # Step 3: Backsolve for A and B gradients
    # We solve: B⋅∇A + ∇B⋅A ≈ proj_grad_delta_w
    # Using pseudo-inverse approach

    # Method 1: Solve for ∇A first (freeze B)
    try:
        B_pinv = torch.linalg.pinv(B)  # (r, d)
        gA_proj = torch.matmul(B_pinv, proj_grad_delta_w)  # (r, d)
    except:
        # Fallback if pseudo-inverse fails
        gA_proj = gA

    # Method 2: Solve for ∇B (freeze A)
    try:
        A_pinv = torch.linalg.pinv(A)  # (d, r)
        gB_proj = torch.matmul(proj_grad_delta_w, A_pinv)  # (d, r)
    except:
        # Fallback if pseudo-inverse fails
        gB_proj = gB

    return gA_proj, gB_proj


def extract_subspace_adaptive_k(
    B: torch.Tensor, A: torch.Tensor, energy_threshold: float = 0.95, k_max: int = None
):
    """
    Extract subspace with adaptive k based on energy retention

    Args:
        B: (d, r) - LoRA B matrix
        A: (r, d) - LoRA A matrix
        energy_threshold: minimum energy retention ratio (default: 0.95)
        k_max: maximum number of vectors to extract (default: None)

    Returns:
        S_new: (d, k) with orthonormal columns, where k is adaptively chosen
    """
    assert B.ndim == 2 and A.ndim == 2
    d, r = B.shape

    # Compute ΔW = BA
    delta_w = torch.matmul(B, A)  # (d, d)

    # Perform SVD on ΔW
    try:
        U, S, Vt = torch.linalg.svd(delta_w)  # U: (d, d), S: (d,)
    except Exception:  # fallback for older torch versions
        U, S, Vt = torch.svd(delta_w)

    # Calculate total energy
    total_energy = S.sum()

    # Calculate cumulative energy ratio
    cumulative_energy = torch.cumsum(S, dim=0)
    energy_ratio = cumulative_energy / total_energy

    # Find minimum k such that energy_ratio >= threshold
    k_indices = (energy_ratio >= energy_threshold).nonzero(as_tuple=False)
    if k_indices.numel() > 0:
        adaptive_k = k_indices[0].item() + 1  # +1 because indices are 0-based
    else:
        adaptive_k = min(r, d)  # fallback to full rank

    # Apply k_max constraint if specified
    if k_max is not None:
        adaptive_k = min(adaptive_k, k_max)

    # Ensure at least 1 vector
    adaptive_k = max(adaptive_k, 1)

    # Extract top-k left singular vectors
    S_new = U[:, :adaptive_k]  # (d, adaptive_k)

    # Orthonormalize using QR decomposition
    Q, _ = torch.linalg.qr(S_new)
    S_new = Q[:, :adaptive_k]

    return S_new


def compute_plasticity_loss(lora_activation: torch.Tensor, eps=1e-8):
    """
    Compute homeostatic plasticity loss based on activation distribution

    Args:
        lora_activation: (B, m) activations after LoRA module (non-negative preferred)

    Returns:
        loss: scalar entropy-like loss (lower when activation distribution is diverse)
    """
    hmean = lora_activation.mean(dim=0)  # (m,)
    hmean = hmean + eps
    p = hmean / hmean.sum()
    loss = -(p * torch.log(p)).sum()

    return loss


def save_subspace(S: torch.Tensor, path: str):
    """Save subspace tensor to file"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(S.cpu(), path)


def load_subspace(path: str, device="cpu"):
    """Load subspace tensor from file"""
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location=device)


def sleep_phase_distill(
    teacher_model, student_model, dataloader, device="cuda", epochs=1, lr=1e-4
):
    """
    Simple self-distillation: MSE between teacher logits and student logits

    Args:
        teacher_model: frozen model (with previous LoRAs applied)
        student_model: model to update
        dataloader: noise/unlabeled data loader
        device: device to run on
        epochs: number of distillation epochs
        lr: learning rate for distillation
    """
    # Check if there are trainable parameters
    trainable_params = [p for p in student_model.parameters() if p.requires_grad]
    if not trainable_params:
        print("Warning: No trainable parameters found for sleep phase distillation")
        return

    opt = torch.optim.Adam(trainable_params, lr=lr)
    mse = nn.MSELoss()

    teacher_model.eval()
    student_model.train()

    total_loss = 0.0
    batch_count = 0

    for ep in range(epochs):
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                xb = batch[1].to(device)  # batch[1] is inputs, batch[0] is indices
            else:
                xb = batch.to(device)

            with torch.no_grad():
                t_out = teacher_model(xb)

            s_out = student_model(xb)

            # Handle both dictionary and tensor outputs
            if isinstance(t_out, dict) and isinstance(s_out, dict):
                # Extract logits from both outputs
                t_logits = t_out.get("logits", t_out.get("output", None))
                s_logits = s_out.get("logits", s_out.get("output", None))

                if t_logits is not None and s_logits is not None:
                    loss = mse(s_logits, t_logits.detach())
                else:
                    # Fallback: use the first tensor value from dictionaries
                    t_tensor = next(
                        (v for v in t_out.values() if torch.is_tensor(v)), None
                    )
                    s_tensor = next(
                        (v for v in s_out.values() if torch.is_tensor(v)), None
                    )

                    if t_tensor is not None and s_tensor is not None:
                        loss = mse(s_tensor, t_tensor.detach())
                    else:
                        print(
                            "Warning: Could not find suitable tensors for distillation loss"
                        )
                        continue
            else:
                # Direct tensor outputs
                if torch.is_tensor(t_out) and torch.is_tensor(s_out):
                    loss = mse(s_out, t_out.detach())
                else:
                    print("Warning: Unexpected output format for distillation")
                    continue

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            batch_count += 1

    if batch_count > 0:
        avg_loss = total_loss / batch_count
        print(f"Sleep phase distillation completed. Average loss: {avg_loss:.6f}")
    else:
        print("Warning: No batches processed during sleep phase distillation")


def get_lora_modules(model):
    """
    Get all LoRA modules from model

    Args:
        model: PyTorch model

    Returns:
        items: list of (name, module) tuples for LoRA modules
    """
    items = []
    for name, module in model.named_modules():
        # Check if module has LoRA methods
        if (
            hasattr(module, "get_delta")
            and hasattr(module, "get_A")
            and hasattr(module, "get_B")
        ):
            items.append((name, module))
    return items


def create_noise_loader(
    batch_size=64, n_batches=10, device="cuda", input_size=(3, 224, 224)
):
    """
    Create a simple noise dataloader for sleep phase

    Args:
        batch_size: batch size
        n_batches: number of batches
        device: device to create tensors on
        input_size: input tensor size

    Returns:
        noise_loader: list of noise batches in format (idx, inputs, targets)
    """
    noise_loader = []
    for i in range(n_batches):
        # Create Gaussian noise
        noise = torch.randn(batch_size, *input_size, device=device)
        # Create dummy targets (not used in distillation)
        dummy_targets = torch.zeros(batch_size, dtype=torch.long, device=device)
        # Create dummy indices
        dummy_indices = torch.arange(batch_size, device=device)

        # Format: (idx, inputs, targets) to match expected dataloader format
        batch = (dummy_indices, noise, dummy_targets)
        noise_loader.append(batch)
    return noise_loader


def compute_task_similarity(
    delta_w_current: torch.Tensor, delta_w_history: list, method: str = "cosine"
):
    """
    Compute similarity between current task's ΔW and previous tasks' ΔW

    Args:
        delta_w_current: current task's ΔW matrix (d, d)
        delta_w_history: list of previous tasks' ΔW matrices [(d, d), ...]
        method: similarity method ("cosine" or "frobenius")

    Returns:
        similarity_score: average similarity score [0, 1]
    """
    if not delta_w_history:
        return torch.tensor(0.0, device=delta_w_current.device)

    similarities = []

    for delta_w_prev in delta_w_history:
        if method == "cosine":
            # Flatten matrices and compute cosine similarity
            curr_flat = delta_w_current.flatten()
            prev_flat = delta_w_prev.flatten()

            # Normalize to unit vectors
            curr_norm = torch.norm(curr_flat)
            prev_norm = torch.norm(prev_flat)

            if curr_norm > 1e-8 and prev_norm > 1e-8:
                sim = torch.dot(curr_flat, prev_flat) / (curr_norm * prev_norm)
                # Convert from [-1, 1] to [0, 1] range
                sim = (sim + 1) / 2
            else:
                sim = torch.tensor(
                    0.5, device=delta_w_current.device
                )  # neutral similarity

        elif method == "frobenius":
            # Frobenius norm-based similarity
            curr_norm = torch.norm(delta_w_current, p="fro")
            prev_norm = torch.norm(delta_w_prev, p="fro")

            if curr_norm > 1e-8 and prev_norm > 1e-8:
                # Normalize both matrices
                curr_normalized = delta_w_current / curr_norm
                prev_normalized = delta_w_prev / prev_norm

                # Compute similarity as dot product of flattened normalized matrices
                sim = torch.dot(curr_normalized.flatten(), prev_normalized.flatten())
                sim = (sim + 1) / 2  # Convert to [0, 1]
            else:
                sim = torch.tensor(0.5, device=delta_w_current.device)
        else:
            raise ValueError(f"Unknown similarity method: {method}")

        similarities.append(sim)

    # Return average similarity
    return torch.stack(similarities).mean()


def project_grad_similarity_aware(
    gA: torch.Tensor,
    gB: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    S: torch.Tensor,
    delta_w_current: torch.Tensor,
    delta_w_history: list,
    similarity_method: str = "cosine",
    similarity_decay: float = 1.0,
    min_projection_strength: float = 0.1,
):
    """
    Similarity-aware gradient projection

    Args:
        gA, gB: gradients of A and B matrices
        A, B: current A and B matrices
        S: cumulative subspace
        delta_w_current: current task's ΔW matrix
        delta_w_history: list of previous tasks' ΔW matrices
        similarity_method: "cosine" or "frobenius"
        similarity_decay: power to apply to similarity (β in α = sim^β)
        min_projection_strength: minimum projection strength (0.1 = always project at least 10%)

    Returns:
        gA_proj, gB_proj: similarity-aware projected gradients
    """
    if S is None or S.numel() == 0:
        return gA, gB

    # Compute task similarity
    similarity = compute_task_similarity(
        delta_w_current, delta_w_history, method=similarity_method
    )

    # Apply decay and clamp to get projection strength
    alpha = (similarity**similarity_decay).clamp(min=min_projection_strength, max=1.0)

    # Compute total gradient of ΔW = BA
    grad_delta_w = torch.matmul(B, gA) + torch.matmul(gB, A)  # (d, d)

    # Apply similarity-aware projection
    # g_proj = g - α * (S @ (S.T @ g))
    coef = torch.matmul(S.T, grad_delta_w)  # (K, d)
    proj_grad_delta_w = grad_delta_w - alpha * torch.matmul(S, coef)  # (d, d)

    # Backsolve for A and B gradients
    try:
        B_pinv = torch.linalg.pinv(B)
        gA_proj = torch.matmul(B_pinv, proj_grad_delta_w)
    except:
        gA_proj = gA

    try:
        A_pinv = torch.linalg.pinv(A)
        gB_proj = torch.matmul(proj_grad_delta_w, A_pinv)
    except:
        gB_proj = gB

    return gA_proj, gB_proj, alpha.item()


def save_delta_w_history(delta_w_history: list, save_path: str):
    """
    Save ΔW history to disk

    Args:
        delta_w_history: list of ΔW tensors
        save_path: path to save the history
    """
    import os

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Convert to list of numpy arrays for saving
    delta_w_numpy = [dw.cpu().numpy() for dw in delta_w_history]
    torch.save(delta_w_numpy, save_path)


def load_delta_w_history(load_path: str, device: str = "cpu"):
    """
    Load ΔW history from disk

    Args:
        load_path: path to load the history from
        device: device to load tensors to

    Returns:
        delta_w_history: list of ΔW tensors
    """
    if not os.path.exists(load_path):
        return []

    try:
        # Try with weights_only=False for PyTorch 2.6 compatibility
        try:
            delta_w_numpy = torch.load(
                load_path, map_location=device, weights_only=False
            )
        except TypeError:
            # Fallback for older PyTorch versions
            delta_w_numpy = torch.load(load_path, map_location=device)
        delta_w_history = [torch.tensor(dw, device=device) for dw in delta_w_numpy]
        return delta_w_history
    except Exception as e:
        print(f"Warning: Could not load delta_w_history from {load_path}: {e}")
        return []
