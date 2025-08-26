"""
Hyperspherical utilities for H-DRS implementation in LoRA-Sub-DRS
Implements spherical operations, Möbius transformations, and spCauchy distributions
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy.special import digamma
import math


def normalize_to_sphere(features, dim=-1, eps=1e-8):
    """
    Normalize features to unit hypersphere (S^{d-1})

    Args:
        features: Input features tensor
        dim: Dimension to normalize along
        eps: Small epsilon for numerical stability

    Returns:
        Normalized features on unit sphere
    """
    return F.normalize(features, p=2, dim=dim, eps=eps)


def mobius_transform(x, mu, rho):
    """
    Möbius transformation for spherical Cauchy distribution
    Based on Hyperspherical VAE paper implementation

    Args:
        x: Points on unit sphere (batch_size, d)
        mu: Center point on unit sphere (d,) or (1, d)
        rho: Concentration parameter in [0, 1)

    Returns:
        Transformed points on unit sphere
    """
    if mu.dim() == 1:
        mu = mu.unsqueeze(0)

    # Ensure inputs are on unit sphere
    x = normalize_to_sphere(x)
    mu = normalize_to_sphere(mu)

    # Compute dot product
    dot = torch.sum(x * mu, dim=-1, keepdim=True)  # (batch, 1)

    # Möbius transformation formula
    numerator = (1 - rho**2) * x + rho * mu * (1 + rho * dot)
    denominator = 1 + 2 * rho * dot + rho**2

    y = numerator / (denominator + 1e-8)

    # Project back to sphere for numerical stability
    return normalize_to_sphere(y)


def sample_spcauchy(mu, rho, num_samples=1, device="cuda"):
    """
    Sample from spherical Cauchy distribution

    Args:
        mu: Center point on unit sphere (d,)
        rho: Concentration parameter in [0, 1)
        num_samples: Number of samples to generate
        device: Device to generate samples on

    Returns:
        Samples from spherical Cauchy (num_samples, d)
    """
    d = mu.size(-1)

    # Sample uniformly on sphere (Gaussian then normalize)
    eps = torch.randn(num_samples, d, device=device)
    x_uniform = normalize_to_sphere(eps)

    # Apply Möbius transformation
    if mu.dim() == 1:
        mu = mu.unsqueeze(0)

    return mobius_transform(x_uniform, mu, rho)


def kl_spcauchy(rho_q, rho_p=0.0, d=768, K=5):
    """
    Approximate KL divergence between spherical Cauchy distributions
    Using simplified approximation to avoid overflow

    Args:
        rho_q: Concentration of posterior
        rho_p: Concentration of prior (0 for uniform)
        d: Dimension of sphere
        K: Number of terms in series (reduced to avoid overflow)

    Returns:
        KL divergence value
    """
    if isinstance(rho_q, torch.Tensor):
        rho_q = rho_q.item()

    # Simplified approximation to avoid gamma overflow
    if abs(rho_q - rho_p) < 1e-6:
        return torch.tensor(0.0)

    # Use log-space computation to avoid overflow
    try:
        term1 = (d - 1) * np.log((1 - rho_q) / (1 - rho_p + 1e-8))

        # Simplified series approximation
        z = 4 * rho_q / (1 + rho_q) ** 2
        series = 0

        # Use only first few terms to avoid overflow
        for k in range(min(K, 5)):
            if k == 0:
                poch_log = 0  # log(1) = 0
            else:
                # Use log-gamma for stability
                try:
                    poch_log = (
                        math.lgamma(d - 0.5 + k)
                        - math.lgamma(d - 0.5)
                        - math.lgamma(k + 1)
                    )
                    poch = np.exp(poch_log)
                except:
                    break

            term = poch * (z**k) if k == 0 else poch * (z**k) / math.factorial(k)
            series += term

            # Stop if term becomes too small
            if term < 1e-10:
                break

        term2 = (d - 1) * ((1 - rho_q) / (1 + rho_q)) ** (d - 1) * series

        result = term1 + term2

        # Clamp result to reasonable range
        result = np.clip(result, 0, 100)

        return torch.tensor(result)

    except:
        # Fallback: simple approximation
        return torch.tensor((rho_q - rho_p) ** 2 * d)


def angular_distance(x, y, eps=1e-8):
    """
    Compute angular distance between points on sphere

    Args:
        x, y: Points on unit sphere
        eps: Small epsilon for numerical stability

    Returns:
        Angular distances
    """
    # Ensure inputs are normalized
    x = normalize_to_sphere(x)
    y = normalize_to_sphere(y)

    # Compute cosine similarity
    cos_sim = F.cosine_similarity(x, y, dim=-1)

    # Clamp for numerical stability
    cos_sim = torch.clamp(cos_sim, -1 + eps, 1 - eps)

    # Return angular distance
    return torch.acos(cos_sim)


def spherical_covariance(features):
    """
    Compute spherical covariance matrix for features on unit sphere

    Args:
        features: Features on unit sphere (batch_size, d)

    Returns:
        Spherical covariance matrix (d, d)
    """
    # Ensure features are normalized
    features = normalize_to_sphere(features)

    # Spherical covariance using dot products
    cov = torch.mm(features.t(), features) / features.size(0)

    return cov


def spherical_pca(features, variance_threshold=0.95):
    """
    Perform spherical PCA on features

    Args:
        features: Features on unit sphere (batch_size, d)
        variance_threshold: Cumulative variance threshold for component selection

    Returns:
        Principal components and selected dimension
    """
    # Compute spherical covariance
    cov = spherical_covariance(features)

    # SVD decomposition
    U, S, V = torch.svd(cov)

    # Select components based on variance threshold
    cum_var = torch.cumsum(S / S.sum(), dim=0)
    k = (cum_var >= variance_threshold).nonzero(as_tuple=False)[0].item() + 1

    # Return top-k components
    return U[:, :k], k


class HypersphericalProjector:
    """
    Projector for Hyperspherical DRS operations
    """

    def __init__(self, sphere_dim=768, spcauchy_rho=0.5, variance_threshold=0.95):
        self.sphere_dim = sphere_dim
        self.spcauchy_rho = spcauchy_rho
        self.variance_threshold = variance_threshold
        self.projection_matrix = None
        self.mean_direction = None

    def compute_projection(self, features, use_spcauchy=True):
        """
        Compute H-DRS projection matrix

        Args:
            features: Input features (batch_size, d)
            use_spcauchy: Whether to enhance with spCauchy sampling

        Returns:
            Projection matrix and number of components
        """
        # Normalize features to sphere
        features_norm = normalize_to_sphere(features)

        # Compute mean direction
        self.mean_direction = normalize_to_sphere(
            torch.mean(features_norm, dim=0, keepdim=True)
        )

        if use_spcauchy:
            # Use spCauchy sampling for projection basis
            k = min(features.size(1) // 4, 256)  # Adaptive number of components
            sp_samples = sample_spcauchy(
                self.mean_direction.squeeze(),
                self.spcauchy_rho,
                num_samples=k,
                device=features.device,
            )
            self.projection_matrix = sp_samples.t()  # (d, k)
        else:
            # Use spherical PCA
            U, k = spherical_pca(features_norm, self.variance_threshold)
            self.projection_matrix = U

        return self.projection_matrix, self.projection_matrix.size(1)

    def project_gradients(self, gradients):
        """
        Project gradients to H-DRS subspace

        Args:
            gradients: Gradients to project

        Returns:
            Projected gradients
        """
        if self.projection_matrix is None:
            return gradients

        # Flatten gradients
        original_shape = gradients.shape
        grad_flat = gradients.view(-1)

        # Check dimension compatibility
        if grad_flat.size(0) != self.projection_matrix.size(0):
            # If dimensions don't match, skip projection
            return gradients

        # Project to subspace: P @ P^T @ g
        projected = torch.mm(
            self.projection_matrix,
            torch.mm(self.projection_matrix.t(), grad_flat.unsqueeze(1)),
        ).squeeze()

        # Apply Möbius transformation for directional update
        if (
            self.mean_direction is not None
            and projected.numel() == self.mean_direction.numel()
        ):
            projected_norm = normalize_to_sphere(projected.unsqueeze(0))
            projected = mobius_transform(
                projected_norm, self.mean_direction, self.spcauchy_rho
            ).squeeze()

        return projected.view(original_shape)


def save_spherical_prototypes(prototypes, task_id, save_dir="./prototypes"):
    """
    Save spherical prototypes for a task

    Args:
        prototypes: Dictionary of class prototypes
        task_id: Current task ID
        save_dir: Directory to save prototypes
    """
    import os

    os.makedirs(save_dir, exist_ok=True)

    # Normalize all prototypes to sphere
    spherical_prototypes = {}
    for class_id, proto in prototypes.items():
        spherical_prototypes[class_id] = normalize_to_sphere(proto)

    torch.save(
        spherical_prototypes, f"{save_dir}/spherical_prototypes_task_{task_id}.pt"
    )


def load_spherical_prototypes(task_id, save_dir="./prototypes"):
    """
    Load spherical prototypes for a task

    Args:
        task_id: Task ID to load
        save_dir: Directory containing prototypes

    Returns:
        Dictionary of spherical prototypes or None if not found
    """
    import os

    filepath = f"{save_dir}/spherical_prototypes_task_{task_id}.pt"

    if os.path.exists(filepath):
        return torch.load(filepath)
    return None
