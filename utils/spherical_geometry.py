#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Spherical Geometry Utilities for DRS-Hyperspherical

Implements Riemannian operations on hyperspheres including:
- Log/Exp maps
- Geodesic distances
- Tangent space projections
- PCA on tangent spaces
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Union
import logging


class SphericalGeometry:
    """Core spherical geometry operations on S^{d-1}"""

    @staticmethod
    def normalize_to_sphere(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Normalize vectors to unit hypersphere S^{d-1}

        Args:
            x: Input tensor [..., d]
            eps: Small epsilon for numerical stability

        Returns:
            Normalized tensor on sphere
        """
        norm = torch.norm(x, dim=-1, keepdim=True)
        return x / (norm + eps)

    @staticmethod
    def geodesic_distance(
        u: torch.Tensor, v: torch.Tensor, eps: float = 1e-7
    ) -> torch.Tensor:
        """
        Compute geodesic distance on sphere: d_g(u,v) = arccos(u^T v)

        Args:
            u, v: Points on sphere [..., d]
            eps: Clamping epsilon

        Returns:
            Geodesic distances [...]
        """
        # Clamp cosine to valid range to prevent NaN
        cos_theta = torch.sum(u * v, dim=-1)
        cos_theta = torch.clamp(cos_theta, -1 + eps, 1 - eps)
        return torch.arccos(cos_theta)

    @staticmethod
    def log_map(mu: torch.Tensor, x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        """
        Logarithmic map from sphere to tangent space at mu
        log_μ(x) = (θ/sin(θ)) * (x - cos(θ)*μ)

        Args:
            mu: Base point on sphere [d] or [..., d]
            x: Points on sphere [..., d]
            eps: Small epsilon for numerical stability

        Returns:
            Tangent vectors [..., d]
        """
        # Ensure mu and x are on sphere
        mu = SphericalGeometry.normalize_to_sphere(mu, eps)
        x = SphericalGeometry.normalize_to_sphere(x, eps)

        # Compute angle θ = arccos(μ^T x)
        cos_theta = torch.sum(mu * x, dim=-1, keepdim=True)
        cos_theta = torch.clamp(cos_theta, -1 + eps, 1 - eps)
        theta = torch.arccos(cos_theta)

        # Handle small angles (θ ≈ 0) with Taylor expansion
        sin_theta = torch.sin(theta)
        small_angle_mask = theta < 1e-3

        # For small angles: log_map ≈ x - μ (first order)
        # For normal angles: log_map = (θ/sin(θ)) * (x - cos(θ)*μ)
        coefficient = torch.where(
            small_angle_mask,
            torch.ones_like(theta),  # First order approximation
            theta / (sin_theta + eps),
        )

        tangent_component = x - cos_theta * mu
        return coefficient * tangent_component

    @staticmethod
    def exp_map(mu: torch.Tensor, v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Exponential map from tangent space to sphere (retraction)
        exp_μ(v) = cos(||v||)*μ + sin(||v||) * v/||v||

        Args:
            mu: Base point on sphere [d] or [..., d]
            v: Tangent vectors [..., d]
            eps: Small epsilon for numerical stability

        Returns:
            Points on sphere [..., d]
        """
        mu = SphericalGeometry.normalize_to_sphere(mu, eps)

        v_norm = torch.norm(v, dim=-1, keepdim=True)
        v_normalized = v / (v_norm + eps)

        # Handle zero tangent vectors
        zero_mask = v_norm.squeeze(-1) < eps
        cos_norm = torch.cos(v_norm)
        sin_norm = torch.sin(v_norm)

        result = cos_norm * mu + sin_norm * v_normalized

        # For zero tangent vectors, return mu
        if zero_mask.any():
            result = torch.where(zero_mask.unsqueeze(-1), mu, result)

        return SphericalGeometry.normalize_to_sphere(result, eps)

    @staticmethod
    def project_to_tangent(mu: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Project vector v to tangent space at mu: v_T = (I - μμ^T)v

        Args:
            mu: Base point on sphere [d] or [..., d]
            v: Vector to project [..., d]

        Returns:
            Projected vector in tangent space [..., d]
        """
        mu = SphericalGeometry.normalize_to_sphere(mu)

        # Project: v_T = v - (μ^T v) * μ
        mu_dot_v = torch.sum(mu * v, dim=-1, keepdim=True)
        return v - mu_dot_v * mu


class TangentPCA:
    """PCA operations in tangent space for DRS construction"""

    def __init__(self, energy_threshold: float = 0.90, max_components: int = 128):
        self.energy_threshold = energy_threshold
        self.max_components = max_components
        self.U = None  # Principal components
        self.explained_variance_ratio = None
        self.n_components = None

    def fit(self, tangent_vectors: torch.Tensor, center: bool = True) -> "TangentPCA":
        """
        Fit PCA on tangent vectors

        Args:
            tangent_vectors: Tensor [N, d] of tangent vectors
            center: Whether to center the data

        Returns:
            self
        """
        if center:
            mean = torch.mean(tangent_vectors, dim=0, keepdim=True)
            centered_vectors = tangent_vectors - mean
        else:
            centered_vectors = tangent_vectors

        # Compute covariance matrix
        N = centered_vectors.shape[0]
        cov_matrix = torch.mm(centered_vectors.t(), centered_vectors) / (N - 1)

        # Eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)

        # Sort in descending order
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Compute explained variance ratio
        total_variance = torch.sum(eigenvalues)
        explained_variance_ratio = eigenvalues / total_variance

        # Select components based on energy threshold
        cumulative_variance = torch.cumsum(explained_variance_ratio, dim=0)
        n_components = (
            torch.sum(cumulative_variance <= self.energy_threshold).item() + 1
        )
        n_components = min(n_components, self.max_components)

        self.U = eigenvectors[:, :n_components]
        self.explained_variance_ratio = explained_variance_ratio[:n_components]
        self.n_components = n_components

        logging.info(
            f"TangentPCA: Selected {n_components} components, "
            f"explained variance: {torch.sum(self.explained_variance_ratio):.3f}"
        )

        return self

    def project(self, tangent_vectors: torch.Tensor) -> torch.Tensor:
        """
        Project tangent vectors to DRS subspace: U_t U_t^T v

        Args:
            tangent_vectors: Tangent vectors [..., d]

        Returns:
            Projected vectors [..., d]
        """
        if self.U is None:
            raise RuntimeError("TangentPCA not fitted yet")

        # Project to subspace and back to full space
        projected = torch.mm(tangent_vectors, self.U)  # [..., k]
        return torch.mm(projected, self.U.t())  # [..., d]

    def get_components(self) -> torch.Tensor:
        """Get the principal components matrix U_t"""
        return self.U


class PrototypeManager:
    """Manages spherical prototypes with EMA updates"""

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        momentum: float = 0.97,
        per_class: bool = True,
        multi_anchor: bool = False,
        num_anchors: int = 3,
    ):
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.momentum = momentum
        self.per_class = per_class
        self.multi_anchor = multi_anchor
        self.num_anchors = num_anchors

        if per_class and not multi_anchor:
            # Single prototype per class
            self.prototypes = torch.zeros(num_classes, feature_dim)
            self.initialized = torch.zeros(num_classes, dtype=torch.bool)
        elif multi_anchor:
            # Multiple anchors per class
            self.prototypes = torch.zeros(num_classes, num_anchors, feature_dim)
            self.initialized = torch.zeros(num_classes, num_anchors, dtype=torch.bool)
        else:
            # Single global prototype
            self.prototypes = torch.zeros(1, feature_dim)
            self.initialized = torch.zeros(1, dtype=torch.bool)

    def initialize_prototypes(
        self, embeddings: torch.Tensor, labels: torch.Tensor, device: torch.device
    ):
        """
        Initialize prototypes from initial embeddings

        Args:
            embeddings: Normalized embeddings [N, d]
            labels: Class labels [N]
            device: Device to move tensors to
        """
        self.prototypes = self.prototypes.to(device)
        self.initialized = self.initialized.to(device)

        if self.per_class:
            for class_idx in range(self.num_classes):
                class_mask = labels == class_idx
                if class_mask.sum() > 0:
                    class_embeddings = embeddings[class_mask]

                    if self.multi_anchor:
                        # K-means on sphere for multiple anchors
                        anchors = self._spherical_kmeans(
                            class_embeddings, self.num_anchors
                        )
                        self.prototypes[class_idx] = anchors
                        self.initialized[class_idx] = True
                    else:
                        # Mean direction (Fréchet mean on sphere)
                        mean_direction = torch.mean(class_embeddings, dim=0)
                        self.prototypes[class_idx] = (
                            SphericalGeometry.normalize_to_sphere(mean_direction)
                        )
                        self.initialized[class_idx] = True
        else:
            # Global prototype
            mean_direction = torch.mean(embeddings, dim=0)
            self.prototypes[0] = SphericalGeometry.normalize_to_sphere(mean_direction)
            self.initialized[0] = True

    def update_prototypes(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """
        Update prototypes using EMA

        Args:
            embeddings: Normalized embeddings [N, d]
            labels: Class labels [N]
        """
        if self.per_class:
            for class_idx in range(self.num_classes):
                class_mask = labels == class_idx
                if class_mask.sum() > 0:
                    class_embeddings = embeddings[class_mask]
                    mean_embedding = torch.mean(class_embeddings, dim=0)
                    mean_embedding = SphericalGeometry.normalize_to_sphere(
                        mean_embedding
                    )

                    if not self.multi_anchor:
                        if self.initialized[class_idx]:
                            # EMA update
                            self.prototypes[class_idx] = (
                                self.momentum * self.prototypes[class_idx]
                                + (1 - self.momentum) * mean_embedding
                            )
                            self.prototypes[class_idx] = (
                                SphericalGeometry.normalize_to_sphere(
                                    self.prototypes[class_idx]
                                )
                            )
                        else:
                            self.prototypes[class_idx] = mean_embedding
                            self.initialized[class_idx] = True

    def get_prototype(self, class_idx: Optional[int] = None) -> torch.Tensor:
        """Get prototype(s) for a class or global"""
        if self.per_class and class_idx is not None:
            return self.prototypes[class_idx]
        elif not self.per_class:
            return self.prototypes[0]
        else:
            return self.prototypes

    def get_closest_anchor(
        self, embeddings: torch.Tensor, class_idx: int
    ) -> torch.Tensor:
        """Get closest anchor for multi-anchor setup"""
        if not self.multi_anchor:
            return self.get_prototype(class_idx)

        anchors = self.prototypes[class_idx]  # [num_anchors, d]

        # Compute distances to all anchors
        distances = SphericalGeometry.geodesic_distance(
            embeddings.unsqueeze(1),  # [N, 1, d]
            anchors.unsqueeze(0),  # [1, num_anchors, d]
        )  # [N, num_anchors]

        # Find closest anchor for each embedding
        closest_anchor_idx = torch.argmin(distances, dim=1)  # [N]
        return anchors[closest_anchor_idx]  # [N, d]

    def _spherical_kmeans(
        self, embeddings: torch.Tensor, k: int, max_iter: int = 50
    ) -> torch.Tensor:
        """Simple spherical K-means clustering"""
        N, d = embeddings.shape

        if N <= k:
            # Not enough points, use available points + random
            centers = torch.zeros(k, d, device=embeddings.device)
            centers[:N] = embeddings
            for i in range(N, k):
                centers[i] = SphericalGeometry.normalize_to_sphere(
                    torch.randn(d, device=embeddings.device)
                )
            return centers

        # Initialize centers randomly
        idx = torch.randperm(N)[:k]
        centers = embeddings[idx].clone()

        for _ in range(max_iter):
            # Assign points to closest center
            distances = SphericalGeometry.geodesic_distance(
                embeddings.unsqueeze(1),  # [N, 1, d]
                centers.unsqueeze(0),  # [1, k, d]
            )  # [N, k]

            assignments = torch.argmin(distances, dim=1)  # [N]

            # Update centers
            new_centers = torch.zeros_like(centers)
            for i in range(k):
                mask = assignments == i
                if mask.sum() > 0:
                    cluster_mean = torch.mean(embeddings[mask], dim=0)
                    new_centers[i] = SphericalGeometry.normalize_to_sphere(cluster_mean)
                else:
                    new_centers[i] = centers[i]  # Keep old center if no assignments

            # Check convergence
            if torch.allclose(centers, new_centers, atol=1e-4):
                break

            centers = new_centers

        return centers


def gradient_projection_hook(
    spherical_geom: SphericalGeometry,
    prototype_manager: PrototypeManager,
    tangent_pca: Optional[TangentPCA] = None,
    class_labels: Optional[torch.Tensor] = None,
):
    """
    Create a gradient projection hook for spherical DRS

    Args:
        spherical_geom: SphericalGeometry instance
        prototype_manager: PrototypeManager instance
        tangent_pca: Optional TangentPCA for subspace projection
        class_labels: Class labels for current batch

    Returns:
        Hook function to be registered on embedding tensor
    """

    def hook_fn(grad):
        """
        Project gradients to Riemannian tangent space and optionally to DRS subspace

        Args:
            grad: Gradient tensor [..., d]

        Returns:
            Projected gradient
        """
        if grad is None:
            return grad

        # Get appropriate prototypes for each sample
        if class_labels is not None and prototype_manager.per_class:
            if prototype_manager.multi_anchor:
                # Use closest anchors
                batch_prototypes = []
                for i, label in enumerate(class_labels):
                    embedding = grad[i : i + 1]  # Current embedding (proxy)
                    closest_mu = prototype_manager.get_closest_anchor(
                        embedding, label.item()
                    )
                    batch_prototypes.append(closest_mu)
                mu_batch = torch.stack(batch_prototypes, dim=0)
            else:
                # Use class prototypes
                mu_batch = prototype_manager.get_prototype()[class_labels]
        else:
            # Use global prototype
            mu_global = prototype_manager.get_prototype()
            mu_batch = mu_global.expand_as(grad)

        # Project to tangent space: g_T = (I - μμ^T)g
        tangent_grad = spherical_geom.project_to_tangent(mu_batch, grad)

        # Optionally project to DRS subspace: g_proj = U_t U_t^T g_T
        if tangent_pca is not None and tangent_pca.U is not None:
            # Reshape for matrix operations if needed
            original_shape = tangent_grad.shape
            if len(original_shape) > 2:
                tangent_grad_flat = tangent_grad.view(-1, original_shape[-1])
                projected_grad = tangent_pca.project(tangent_grad_flat)
                projected_grad = projected_grad.view(original_shape)
            else:
                projected_grad = tangent_pca.project(tangent_grad)
        else:
            projected_grad = tangent_grad

        # Ensure projection doesn't increase gradient norm (safety)
        original_norm = torch.norm(grad, dim=-1, keepdim=True)
        projected_norm = torch.norm(projected_grad, dim=-1, keepdim=True)

        # Scale down if projection increased norm
        scale = torch.where(
            projected_norm > original_norm,
            original_norm / (projected_norm + 1e-8),
            torch.ones_like(projected_norm),
        )

        return projected_grad * scale

    return hook_fn
