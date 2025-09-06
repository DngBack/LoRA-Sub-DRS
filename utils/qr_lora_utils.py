#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
from copy import deepcopy

EPS = 1e-12


def extract_subspace_qr_from_BA(
    B: torch.Tensor,
    A: torch.Tensor,
    k: int,
    use_pivoting: bool = True,
    energy_threshold: float = 0.95,
):
    """
    Extract top-k orthogonal directions from LoRA matrices using QR decomposition
    with column pivoting for better numerical stability.

    Args:
        B: (d, r) - LoRA B matrix
        A: (r, d) - LoRA A matrix
        k: number of top directions to extract
        use_pivoting: whether to use column pivoting for importance selection
        energy_threshold: threshold for adaptive k selection

    Returns:
        S_new: (d, k) with orthonormal columns
        importance_scores: (k,) importance scores from R matrix
    """
    assert B.ndim == 2 and A.ndim == 2
    d, r = B.shape

    # Compute ΔW = BA for QR decomposition
    delta_w = torch.matmul(B, A)  # (d, d)

    if use_pivoting:
        # QR decomposition with column pivoting
        try:
            Q, R, P = torch.linalg.qr(
                delta_w, mode="reduced"
            )  # Q: (d, d), R: (d, d), P: (d, d)
        except Exception:
            # Fallback to regular QR if pivoting fails
            Q, R = torch.linalg.qr(delta_w, mode="reduced")
            P = torch.eye(d, device=delta_w.device)
    else:
        Q, R = torch.linalg.qr(delta_w, mode="reduced")
        P = torch.eye(d, device=delta_w.device)

    # Extract importance scores from diagonal of R
    importance_scores = torch.abs(torch.diag(R))

    # Adaptive k selection based on energy threshold
    if energy_threshold < 1.0:
        total_energy = importance_scores.sum()
        cumulative_energy = torch.cumsum(importance_scores, dim=0)
        energy_ratio = cumulative_energy / total_energy

        k_indices = (energy_ratio >= energy_threshold).nonzero(as_tuple=False)
        if k_indices.numel() > 0:
            adaptive_k = k_indices[0].item() + 1
        else:
            adaptive_k = min(r, d)

        k = min(k, adaptive_k)

    # Select top-k columns based on importance
    k = min(k, Q.shape[1])
    S_new = Q[:, :k]  # (d, k)
    importance_scores = importance_scores[:k]  # (k,)

    return S_new, importance_scores


def gated_fusion_subspaces(
    S_old: torch.Tensor,
    S_new: torch.Tensor,
    gate_temperature: float = 1.0,
    fusion_strength: float = 0.5,
):
    """
    Gated fusion mechanism to selectively merge old and new subspaces.

    Args:
        S_old: (d, K_old) previous cumulative subspace
        S_new: (d, k) new subspace from current task
        gate_temperature: temperature for attention computation
        fusion_strength: learnable parameter controlling fusion strength

    Returns:
        S_fused: (d, K_fused) fused subspace
        gate_weights: attention weights for fusion
    """
    if S_old is None or S_old.numel() == 0:
        return S_new, torch.ones(S_new.shape[1], device=S_new.device)

    d, K_old = S_old.shape
    _, k = S_new.shape

    # Compute attention between old and new subspaces
    # Attention(Q_old, Q_new) = softmax(Q_old^T Q_new / sqrt(d) / temperature)
    attention_matrix = (
        torch.matmul(S_old.T, S_new) / math.sqrt(d) / gate_temperature
    )  # (K_old, k)
    attention_weights = F.softmax(attention_matrix, dim=0)  # (K_old, k)

    # Compute gate weights based on similarity
    similarity_scores = torch.norm(attention_matrix, dim=0)  # (k,)
    gate_weights = torch.sigmoid(fusion_strength * similarity_scores)  # (k,)

    # Fused subspace: weighted combination
    S_fused_list = []

    for i in range(k):
        # For each new direction, compute weighted combination with old directions
        weights = attention_weights[:, i]  # (K_old,)
        gate_weight = gate_weights[i]  # scalar

        # Weighted combination of old directions
        weighted_old = torch.matmul(S_old, weights)  # (d,)

        # Gated fusion: gate_weight * weighted_old + (1 - gate_weight) * S_new[:, i]
        fused_direction = gate_weight * weighted_old + (1 - gate_weight) * S_new[:, i]
        S_fused_list.append(fused_direction)

    S_fused = torch.stack(S_fused_list, dim=1)  # (d, k)

    # Orthonormalize the fused subspace
    Q, _ = torch.linalg.qr(S_fused)
    S_fused = Q[:, :k]

    return S_fused, gate_weights


def merge_cumulative_subspace_qr(
    S_prev: torch.Tensor,
    S_new: torch.Tensor,
    K_max: int,
    use_gated_fusion: bool = True,
    fusion_strength: float = 0.5,
):
    """
    Merge previous and new subspaces using QR decomposition with optional gated fusion.

    Args:
        S_prev: previous cumulative subspace or None
        S_new: new subspace from current task
        K_max: maximum allowed dimensions
        use_gated_fusion: whether to use gated fusion mechanism
        fusion_strength: strength of gated fusion

    Returns:
        S_cum: (d, K_keep) orthonormal cumulative subspace
        fusion_info: dictionary with fusion statistics
    """
    fusion_info = {}

    if S_prev is None or S_prev.numel() == 0:
        S_merge = S_new
        fusion_info["method"] = "new_only"
    else:
        if use_gated_fusion:
            # Use gated fusion to merge subspaces
            S_fused, gate_weights = gated_fusion_subspaces(
                S_prev, S_new, fusion_strength=fusion_strength
            )

            # Combine fused subspace with remaining old subspace
            S_merge = torch.cat([S_prev, S_fused], dim=1)  # (d, K_old + k)
            fusion_info["method"] = "gated_fusion"
            fusion_info["gate_weights"] = gate_weights
            fusion_info["avg_gate_weight"] = gate_weights.mean().item()
        else:
            # Traditional concatenation
            S_merge = torch.cat([S_prev, S_new], dim=1)  # (d, K_prev + k)
            fusion_info["method"] = "concatenation"

    # QR orthogonalization
    Q, _ = torch.linalg.qr(S_merge)
    K_keep = min(Q.shape[1], K_max)
    S_cum = Q[:, :K_keep].contiguous()

    fusion_info["final_dim"] = K_keep
    fusion_info["input_dim"] = S_merge.shape[1]

    return S_cum, fusion_info


def project_grad_qr_gated(
    gA: torch.Tensor,
    gB: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    S: torch.Tensor,
    gate_weights: torch.Tensor = None,
):
    """
    Project gradients using QR-based subspace with optional gating.

    Args:
        gA: gradient of A matrix (r, d)
        gB: gradient of B matrix (d, r)
        A: current A matrix (r, d)
        B: current B matrix (d, r)
        S: cumulative subspace (d, K)
        gate_weights: optional gate weights for selective projection

    Returns:
        gA_proj: projected gradient of A
        gB_proj: projected gradient of B
    """
    if S is None or S.numel() == 0:
        return gA, gB

    # Project gB to orthogonal complement of S
    coef_B = torch.matmul(S.T, gB)  # (K, r)
    gB_proj = gB - torch.matmul(S, coef_B)  # (d, r)

    # Apply gating if provided
    if gate_weights is not None:
        # Gate the projection based on fusion weights
        gate_factor = gate_weights.mean()  # Use average gate weight
        gB_proj = gate_factor * gB_proj + (1 - gate_factor) * gB

    # Project gA effect in d-dimensional space
    gA_effect = torch.matmul(B, gA)  # (d, d)
    coef_A_effect = torch.matmul(S.T, gA_effect)  # (K, d)
    gA_effect_proj = gA_effect - torch.matmul(S, coef_A_effect)  # (d, d)

    # Apply gating to A effect
    if gate_weights is not None:
        gate_factor = gate_weights.mean()
        gA_effect_proj = gate_factor * gA_effect_proj + (1 - gate_factor) * gA_effect

    # Backsolve for gA_proj
    try:
        B_pinv = torch.linalg.pinv(B)  # (r, d)
        gA_proj = torch.matmul(B_pinv, gA_effect_proj)  # (r, d)
    except:
        # Fallback if pseudo-inverse fails
        gA_proj = gA

    return gA_proj, gB_proj


def compute_qr_regularization_loss(
    gate_weights: torch.Tensor,
    target_gate_value: float = 0.5,
    regularization_weight: float = 0.01,
):
    """
    Compute regularization loss for gate weights to encourage selective fusion.

    Args:
        gate_weights: gate weights from fusion
        target_gate_value: target value for gate weights
        regularization_weight: weight of regularization term

    Returns:
        reg_loss: regularization loss
    """
    if gate_weights is None:
        return torch.tensor(0.0, device="cpu")

    # Encourage gate weights to be around target value
    gate_loss = torch.mean((gate_weights - target_gate_value) ** 2)

    # Encourage diversity in gate weights (not all 0 or 1)
    diversity_loss = -torch.mean(
        gate_weights * torch.log(gate_weights + EPS)
        + (1 - gate_weights) * torch.log(1 - gate_weights + EPS)
    )

    reg_loss = regularization_weight * (gate_loss + 0.1 * diversity_loss)

    return reg_loss


def save_qr_subspace(
    S: torch.Tensor, importance_scores: torch.Tensor, fusion_info: dict, path: str
):
    """Save QR-based subspace with additional information"""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    save_dict = {
        "subspace": S.cpu(),
        "importance_scores": importance_scores.cpu()
        if importance_scores is not None
        else None,
        "fusion_info": fusion_info,
    }

    torch.save(save_dict, path)


def load_qr_subspace(path: str, device="cpu"):
    """Load QR-based subspace with additional information"""
    if not os.path.exists(path):
        return None, None, None

    save_dict = torch.load(path, map_location=device)

    subspace = save_dict["subspace"]
    importance_scores = save_dict.get("importance_scores", None)
    fusion_info = save_dict.get("fusion_info", {})

    return subspace, importance_scores, fusion_info


def create_gated_fusion_module(input_dim: int, hidden_dim: int = 64):
    """
    Create a learnable gated fusion module.

    Args:
        input_dim: input dimension
        hidden_dim: hidden dimension for fusion network

    Returns:
        fusion_module: PyTorch module for gated fusion
    """

    class GatedFusionModule(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim

            # Learnable fusion parameters
            self.fusion_strength = nn.Parameter(torch.tensor(0.5))
            self.gate_temperature = nn.Parameter(torch.tensor(1.0))

            # Attention network for fusion
            self.attention_net = nn.Sequential(
                nn.Linear(input_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )

        def forward(self, S_old, S_new):
            """
            Forward pass for gated fusion.

            Args:
                S_old: (d, K_old) old subspace
                S_new: (d, k) new subspace

            Returns:
                S_fused: (d, k) fused subspace
                gate_weights: (k,) gate weights
            """
            if S_old is None or S_old.numel() == 0:
                return S_new, torch.ones(S_new.shape[1], device=S_new.device)

            d, k = S_new.shape

            # Compute attention between subspaces
            attention_matrix = (
                torch.matmul(S_old.T, S_new) / math.sqrt(d) / self.gate_temperature
            )
            attention_weights = F.softmax(attention_matrix, dim=0)

            # Compute gate weights using attention network
            gate_weights_list = []
            S_fused_list = []

            for i in range(k):
                # Compute attention-weighted old subspace
                weights = attention_weights[:, i]
                weighted_old = torch.matmul(S_old, weights)

                # Compute gate weight using attention network
                concat_features = torch.cat([weighted_old, S_new[:, i]], dim=0)
                gate_weight = self.attention_net(concat_features).squeeze()

                # Apply learnable fusion strength
                gate_weight = torch.sigmoid(self.fusion_strength * gate_weight)

                # Fused direction
                fused_direction = (
                    gate_weight * weighted_old + (1 - gate_weight) * S_new[:, i]
                )

                gate_weights_list.append(gate_weight)
                S_fused_list.append(fused_direction)

            gate_weights = torch.stack(gate_weights_list)
            S_fused = torch.stack(S_fused_list, dim=1)

            # Orthonormalize
            Q, _ = torch.linalg.qr(S_fused)
            S_fused = Q[:, :k]

            return S_fused, gate_weights

    return GatedFusionModule(input_dim, hidden_dim)
