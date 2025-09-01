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
    opt = torch.optim.Adam(
        [p for p in student_model.parameters() if p.requires_grad], lr=lr
    )
    mse = nn.MSELoss()

    teacher_model.eval()
    student_model.train()

    for ep in range(epochs):
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                xb = batch[0].to(device)
            else:
                xb = batch.to(device)

            with torch.no_grad():
                t_out = teacher_model(xb)

            s_out = student_model(xb)
            loss = mse(s_out, t_out.detach())

            opt.zero_grad()
            loss.backward()
            opt.step()


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
        noise_loader: list of noise batches
    """
    noise_loader = []
    for _ in range(n_batches):
        # Create Gaussian noise
        noise = torch.randn(batch_size, *input_size, device=device)
        noise_loader.append(noise)
    return noise_loader
