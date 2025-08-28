"""
Full Riemannian LoRA Implementation for Hyperspherical Drift-Resistant Space
This module implements LoRA parameters on Riemannian manifolds (Sphere and Stiefel)
with manifold-aware optimization and subtraction operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import warnings

# Try to import geoopt, fallback to manual implementation if not available
try:
    import geoopt
    from geoopt import ManifoldParameter
    GEOOPT_AVAILABLE = True
except ImportError:
    GEOOPT_AVAILABLE = False
    warnings.warn("Geoopt not available. Using manual Riemannian operations.")


class ManualSphere:
    """Manual implementation of sphere manifold operations when geoopt is not available"""
    
    @staticmethod
    def exp_map(mu: torch.Tensor, v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Exponential map on sphere: mu + v -> sphere"""
        norm_v = v.norm(dim=-1, keepdim=True)
        norm_v = torch.clamp(norm_v, min=eps)
        return torch.cos(norm_v) * mu + torch.sin(norm_v) * (v / norm_v)
    
    @staticmethod
    def log_map(mu: torch.Tensor, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Logarithmic map on sphere: project x to tangent space at mu"""
        cos_theta = torch.clamp((mu * x).sum(dim=-1, keepdim=True), -1 + eps, 1 - eps)
        theta = torch.acos(cos_theta)
        sin_theta = torch.sin(theta)
        sin_theta = torch.clamp(sin_theta, min=eps)
        v = x - cos_theta * mu
        return (theta / sin_theta) * v
    
    @staticmethod
    def project_to_tangent(mu: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Project vector v to tangent space at mu on sphere"""
        return v - (mu * v).sum(dim=-1, keepdim=True) * mu
    
    @staticmethod
    def normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Normalize to unit sphere"""
        return x / torch.clamp(x.norm(dim=-1, keepdim=True), min=eps)


class ManualStiefel:
    """Manual implementation of Stiefel manifold operations"""
    
    @staticmethod
    def project_to_tangent(X: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Project V to tangent space of Stiefel manifold at X"""
        # For Stiefel(n,p): tangent space is {V : X^T V + V^T X = 0}
        # Projection: V - X(X^T V + V^T X)/2
        XTV = X.transpose(-2, -1) @ V
        return V - X @ ((XTV + XTV.transpose(-2, -1)) / 2)
    
    @staticmethod
    def retraction_qr(X: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """QR retraction for Stiefel manifold"""
        Y = X + V
        m, n = Y.shape[-2:]
        
        # For wide matrices (m < n), we need to use SVD to preserve the full shape
        # The Stiefel manifold St(m, n) consists of orthonormal m-frames in R^n
        if m < n:
            # For wide matrices, use SVD to get orthonormal rows
            U, S, Vh = torch.linalg.svd(Y, full_matrices=False)
            return U @ Vh
        else:
            # For tall or square matrices, use standard QR
            Q, R = torch.linalg.qr(Y)
            # Ensure positive diagonal
            signs = torch.sign(torch.diagonal(R, dim1=-2, dim2=-1))
            signs = torch.where(signs == 0, torch.ones_like(signs), signs)
            Q = Q * signs.unsqueeze(-2)
            return Q


class RiemannianLoRALayer(nn.Module):
    """
    Riemannian LoRA layer with manifold constraints
    A matrix on Stiefel manifold (orthonormal columns)
    B matrix with columns on Sphere manifold
    """
    
    def __init__(self, 
                 input_dim: int,
                 output_dim: int, 
                 rank: int,
                 n_tasks: int = 10,
                 use_geoopt: bool = True,
                 manifold_A: str = "stiefel",  # "stiefel" or "euclidean"
                 manifold_B: str = "sphere",   # "sphere" or "euclidean"
                 scale_init: float = 10.0):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank
        self.n_tasks = n_tasks
        self.use_geoopt = use_geoopt and GEOOPT_AVAILABLE
        self.manifold_A = manifold_A
        self.manifold_B = manifold_B
        
        # Learnable logit scale for normalized features
        self.logit_scale = nn.Parameter(torch.tensor(scale_init))
        
        # Initialize LoRA parameters for each task
        # We use a simple list and register parameters manually
        self.lora_A = []
        self.lora_B = []
        
        # Manifold objects if using geoopt
        if self.use_geoopt:
            if manifold_A == "stiefel":
                self.manifold_A_obj = geoopt.Stiefel(canonical=False)
            elif manifold_A == "sphere":
                self.manifold_A_obj = geoopt.Sphere()
            
            if manifold_B == "sphere":
                self.manifold_B_obj = geoopt.Sphere()
            elif manifold_B == "stiefel":
                self.manifold_B_obj = geoopt.Stiefel(canonical=False)
        
        # Initialize tasks
        for _ in range(n_tasks):
            self._add_task()
    
    def _add_task(self):
        """Add a new task with manifold-constrained parameters"""
        task_idx = len(self.lora_A)
        
        # Initialize A matrix
        if self.manifold_A == "stiefel" and self.use_geoopt:
            # For geoopt Stiefel manifold, we need shape[-1] <= shape[-2]
            # So we use shape (input_dim, rank) instead of (rank, input_dim)
            A_init = torch.randn(self.input_dim, self.rank)
            A_init = ManualStiefel.retraction_qr(torch.zeros(self.input_dim, self.rank), A_init)
            A = ManifoldParameter(A_init, manifold=self.manifold_A_obj)
        elif self.manifold_A == "sphere" and self.use_geoopt:
            # Each row of A on sphere
            A_init = torch.randn(self.rank, self.input_dim)
            A_init = ManualSphere.normalize(A_init)
            A = ManifoldParameter(A_init, manifold=self.manifold_A_obj)
        else:
            # Euclidean fallback
            A = nn.Parameter(torch.randn(self.rank, self.input_dim) * 0.01)
        
        # Initialize B matrix  
        if self.manifold_B == "sphere" and self.use_geoopt:
            # Each column of B on sphere
            B_init = torch.randn(self.output_dim, self.rank)
            B_init = ManualSphere.normalize(B_init.T).T  # normalize columns
            B = ManifoldParameter(B_init, manifold=self.manifold_B_obj)
        elif self.manifold_B == "stiefel" and self.use_geoopt:
            B_init = torch.randn(self.output_dim, self.rank)
            B_init = ManualStiefel.retraction_qr(torch.zeros_like(B_init), B_init)
            B = ManifoldParameter(B_init, manifold=self.manifold_B_obj)
        else:
            # Euclidean fallback
            B = nn.Parameter(torch.randn(self.output_dim, self.rank) * 0.01)
        
        # Register parameters with the module
        self.register_parameter(f'lora_A_{task_idx}', A)
        self.register_parameter(f'lora_B_{task_idx}', B)
        
        self.lora_A.append(A)
        self.lora_B.append(B)
    
    def forward(self, x: torch.Tensor, task_id: int, use_subtraction: bool = False) -> torch.Tensor:
        """
        Forward pass with optional LoRA subtraction for drift resistance
        
        Args:
            x: Input tensor
            task_id: Current task ID
            use_subtraction: Whether to apply LoRA subtraction from previous tasks
        """
        if task_id >= len(self.lora_A):
            raise ValueError(f"Task {task_id} not initialized. Available tasks: {len(self.lora_A)}")
        
        # Compute current task LoRA
        A_curr = self.lora_A[task_id]
        B_curr = self.lora_B[task_id]
        
        if use_subtraction and task_id > 0:
            # Compute cumulative effect from previous tasks
            cumulative_delta = torch.zeros(self.output_dim, self.input_dim, device=x.device)
            for t in range(task_id):
                A_t = self.lora_A[t]
                B_t = self.lora_B[t]
                # Handle different A shapes based on manifold
                if self.manifold_A == "stiefel" and self.use_geoopt:
                    # A is (input_dim, rank), so we need A.T for the multiplication
                    cumulative_delta += B_t @ A_t.T
                else:
                    # A is (rank, input_dim)
                    cumulative_delta += B_t @ A_t
            
            # Apply Riemannian subtraction (simplified version)
            # This subtracts the cumulative effect in a direction-preserving way
            if self.manifold_A == "stiefel" and self.use_geoopt:
                current_delta = B_curr @ A_curr.T
            else:
                current_delta = B_curr @ A_curr
            subtracted_delta = self._riemannian_subtraction(current_delta, cumulative_delta)
            output = F.linear(x, subtracted_delta)
        else:
            # Standard LoRA forward
            if self.manifold_A == "stiefel" and self.use_geoopt:
                # A is (input_dim, rank), so we need A.T for the multiplication
                weight_delta = B_curr @ A_curr.T
            else:
                # A is (rank, input_dim)
                weight_delta = B_curr @ A_curr
            output = F.linear(x, weight_delta)
        
        return output * self.logit_scale
    
    def _riemannian_subtraction(self, current: torch.Tensor, cumulative: torch.Tensor, 
                              eta: float = 0.1) -> torch.Tensor:
        """
        Perform Riemannian subtraction on the manifold
        This is a simplified version that works row-wise on the sphere
        """
        result = torch.zeros_like(current)
        
        for i in range(current.shape[0]):
            w_curr = ManualSphere.normalize(current[i])
            v_cum = cumulative[i]
            
            # Project cumulative to tangent space at current
            tau = ManualSphere.project_to_tangent(w_curr, v_cum)
            norm_tau = tau.norm()
            
            if norm_tau > 1e-8:
                # Apply exponential map with negative direction
                subtracted = ManualSphere.exp_map(w_curr, -eta * tau)
                # Restore original magnitude
                result[i] = subtracted * current[i].norm()
            else:
                result[i] = current[i]
        
        return result
    
    def get_riemannian_optimizer(self, lr: float = 1e-3) -> torch.optim.Optimizer:
        """Get appropriate optimizer for manifold parameters"""
        if self.use_geoopt:
            return geoopt.optim.RiemannianAdam(self.parameters(), lr=lr)
        else:
            # Fallback to regular Adam with manual retraction
            return torch.optim.Adam(self.parameters(), lr=lr)
    
    def apply_drs_projection(self, P_t: torch.Tensor, task_id: int):
        """
        Apply DRS projection to gradients before optimization
        
        Args:
            P_t: Projection matrix (input_dim x k) from tangent space PCA
            task_id: Current task ID
        """
        if task_id >= len(self.lora_A):
            return
        
        # Project A gradients along input dimension
        A = self.lora_A[task_id]
        if A.grad is not None:
            # A is rank x input_dim, project along input dimension
            proj_matrix = P_t @ P_t.T  # input_dim x input_dim
            A.grad.data = A.grad.data @ proj_matrix
        
        # B gradients don't need input projection in this formulation
        # as they operate on the output dimension


class RiemannianAttention(nn.Module):
    """
    Attention layer with Riemannian LoRA for K and V projections
    """
    
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, 
                 attn_drop: float = 0., proj_drop: float = 0., 
                 rank: int = 64, n_tasks: int = 10, use_geoopt: bool = True):
        super().__init__()
        
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # Standard QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Riemannian LoRA for K and V
        self.lora_k = RiemannianLoRALayer(dim, dim, rank, n_tasks, use_geoopt)
        self.lora_v = RiemannianLoRALayer(dim, dim, rank, n_tasks, use_geoopt)
        
        # Feature covariance tracking for DRS
        self.register_buffer('feature_cov', torch.eye(dim))
        self.register_buffer('n_samples', torch.tensor(0))
        
    def forward(self, x: torch.Tensor, task_id: int, 
                register_hook: bool = False, get_feat: bool = False, 
                get_cur_feat: bool = False, get_cur_x: bool = False) -> torch.Tensor:
        B, N, C = x.shape
        
        # Collect features for covariance if needed (when get_feat is True)
        if get_feat or get_cur_feat:
            self._update_feature_covariance(x)
        
        # Standard QKV computation
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply Riemannian LoRA to K and V
        x_flat = x.reshape(-1, C)  # (B*N, C)
        
        # LoRA modifications
        if task_id >= 0:
            k_delta = self.lora_k(x_flat, task_id, use_subtraction=(task_id > 0))
            v_delta = self.lora_v(x_flat, task_id, use_subtraction=(task_id > 0))
            
            k_delta = k_delta.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            v_delta = v_delta.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            
            k = k + k_delta
            v = v + v_delta
        
        # Attention computation
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x
    
    def _update_feature_covariance(self, x: torch.Tensor):
        """Update running feature covariance for DRS computation"""
        x_flat = x.reshape(-1, x.shape[-1])  # (B*N, C)
        n_new = x_flat.shape[0]
        
        # Incremental covariance update
        n_old = self.n_samples.item()
        n_total = n_old + n_new
        
        if n_old == 0:
            self.feature_cov.data = torch.cov(x_flat.T)
        else:
            # Incremental covariance formula
            cov_new = torch.cov(x_flat.T)
            self.feature_cov.data = (n_old * self.feature_cov + n_new * cov_new) / n_total
        
        self.n_samples.data = torch.tensor(n_total)
    
    def compute_drs_projection(self, k: int = 64, energy_threshold: float = 0.99) -> torch.Tensor:
        """
        Compute DRS projection matrix from feature covariance
        
        Args:
            k: Maximum number of components
            energy_threshold: Energy threshold for component selection
            
        Returns:
            Projection matrix P_t (dim x k_selected)
        """
        # Eigendecomposition of covariance matrix
        eigenvals, eigenvecs = torch.linalg.eigh(self.feature_cov)
        
        # Sort in descending order
        idx = torch.argsort(eigenvals, descending=True)
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # Select components based on energy threshold
        cumulative_energy = torch.cumsum(eigenvals, dim=0) / torch.sum(eigenvals)
        k_selected = min(k, (cumulative_energy >= energy_threshold).nonzero()[0].item() + 1)
        
        return eigenvecs[:, :k_selected]
    
    def apply_drs_projection(self, P_t: torch.Tensor, task_id: int):
        """Apply DRS projection to LoRA parameters"""
        self.lora_k.apply_drs_projection(P_t, task_id)
        self.lora_v.apply_drs_projection(P_t, task_id)
    
    def get_riemannian_optimizers(self, lr: float = 1e-3) -> List[torch.optim.Optimizer]:
        """Get optimizers for Riemannian parameters"""
        return [
            self.lora_k.get_riemannian_optimizer(lr),
            self.lora_v.get_riemannian_optimizer(lr)
        ]


def compute_karcher_mean_sphere(points: torch.Tensor, max_iter: int = 50, 
                               tol: float = 1e-6) -> torch.Tensor:
    """
    Compute Karcher mean (Riemannian center of mass) on sphere
    
    Args:
        points: Tensor of shape (n_points, dim) with unit norm points
        max_iter: Maximum iterations
        tol: Convergence tolerance
        
    Returns:
        Karcher mean point on sphere
    """
    # Initialize with Euclidean mean
    mu = points.mean(dim=0)
    mu = ManualSphere.normalize(mu)
    
    for _ in range(max_iter):
        # Compute log maps to tangent space
        tangent_vecs = torch.stack([ManualSphere.log_map(mu, p) for p in points])
        
        # Compute mean in tangent space
        mean_tangent = tangent_vecs.mean(dim=0)
        
        # Check convergence
        if mean_tangent.norm() < tol:
            break
        
        # Update mu using exponential map
        mu = ManualSphere.exp_map(mu, mean_tangent)
    
    return mu


def compute_tangent_pca(points: torch.Tensor, k: int = 64, 
                       energy_threshold: float = 0.99) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform PCA in tangent space of sphere manifold
    
    Args:
        points: Unit norm points on sphere (n_points, dim)
        k: Maximum number of components
        energy_threshold: Energy threshold for component selection
        
    Returns:
        Tuple of (Karcher mean, PCA basis in tangent space)
    """
    # Compute Karcher mean
    mu = compute_karcher_mean_sphere(points)
    
    # Map all points to tangent space
    tangent_vecs = torch.stack([ManualSphere.log_map(mu, p) for p in points])
    
    # Center the tangent vectors (should already be centered due to Karcher mean property)
    tangent_centered = tangent_vecs - tangent_vecs.mean(dim=0)
    
    # Compute covariance and PCA
    cov_matrix = torch.cov(tangent_centered.T)
    eigenvals, eigenvecs = torch.linalg.eigh(cov_matrix)
    
    # Sort in descending order
    idx = torch.argsort(eigenvals, descending=True)
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    
    # Select components based on energy threshold
    cumulative_energy = torch.cumsum(eigenvals, dim=0) / torch.sum(eigenvals)
    k_selected = min(k, (cumulative_energy >= energy_threshold).nonzero()[0].item() + 1)
    
    return mu, eigenvecs[:, :k_selected]
