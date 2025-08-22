#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Fisher Information Matrix utilities for AD-DRS implementation

import torch
import torch.nn.functional as F
from copy import deepcopy


def compute_diagonal_fim(model, dataloader, device):
    """
    Compute diagonal Fisher Information Matrix for the model on given dataloader.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for computing FIM
        device: Device to run computation on
        
    Returns:
        dict: Dictionary containing diagonal FIM for each parameter
    """
    fim = {name: torch.zeros_like(p, device=device)
           for name, p in model.named_parameters() if p.requires_grad}

    model.eval()
    total_samples = 0
    
    with torch.no_grad():
        # First pass: reset gradients
        model.zero_grad()
    
    for batch_idx, (_, images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        model.zero_grad()
        
        # Forward pass
        outputs = model(images)
        if isinstance(outputs, dict):
            logits = outputs['logits']
        else:
            logits = outputs
            
        # Use log_softmax to calculate log-likelihood
        log_probs = F.log_softmax(logits, dim=1)
        
        # Get log-likelihood of the correct class
        sampled_log_probs = log_probs.gather(1, labels.view(-1, 1))
        
        # Compute gradient of log-likelihood
        sampled_log_probs.mean().backward()
        
        # Accumulate squared gradients (diagonal FIM approximation)
        for name, p in model.named_parameters():
            if p.grad is not None and name in fim:
                # FIM = E[grad * grad^T]. With diagonal approx., we just need E[grad^2]
                fim[name] += p.grad.data ** 2 * images.size(0)
        
        total_samples += images.size(0)

    # Average the FIM over the whole dataset
    for name in fim:
        fim[name] /= total_samples

    return fim


class FisherManager:
    """
    Manages Fisher Information Matrix accumulation across tasks.
    """
    
    def __init__(self):
        self.accumulated_fisher = {}
        self.task_count = 0
    
    def update_fisher(self, new_fim):
        """
        Update accumulated Fisher Information Matrix with new task's FIM.
        
        Args:
            new_fim: Dictionary containing FIM for new task
        """
        if not self.accumulated_fisher:
            # First task - initialize
            self.accumulated_fisher = deepcopy(new_fim)
        else:
            # Accumulate FIM from new task
            for name in self.accumulated_fisher:
                if name in new_fim:
                    self.accumulated_fisher[name] += new_fim[name]
        
        self.task_count += 1
    
    def get_fisher(self):
        """
        Get current accumulated Fisher Information Matrix.
        
        Returns:
            dict: Accumulated FIM
        """
        return self.accumulated_fisher
    
    def reset(self):
        """Reset the Fisher manager."""
        self.accumulated_fisher = {}
        self.task_count = 0


def compute_lambda_star(theta_t_minus_1, theta_t_cand, current_fim, accumulated_fim, epsilon=1e-8):
    """
    Compute optimal merging coefficient using Bayesian merging theory.
    
    Args:
        theta_t_minus_1: Previous task's final parameters
        theta_t_cand: Current task's candidate parameters
        current_fim: Fisher Information Matrix for current task
        accumulated_fim: Accumulated Fisher Information Matrix from previous tasks
        epsilon: Small value for numerical stability
        
    Returns:
        torch.Tensor: Optimal lambda coefficient
    """
    # Calculate parameter differences
    delta_theta = {name: theta_t_cand[name] - theta_t_minus_1[name] 
                   for name in theta_t_cand if name in theta_t_minus_1}
    
    # Calculate terms for the lambda* formula
    numerator = 0.0
    denom_term1 = 0.0
    denom_term2 = 0.0
    
    for name in delta_theta:
        if name in current_fim and name in accumulated_fim:
            delta_sq = delta_theta[name] ** 2
            
            # Numerator: delta^T * F_t * delta
            numerator += torch.sum(delta_sq * current_fim[name])
            
            # Denominator term 1: delta^T * F_t * delta (same as numerator)
            denom_term1 += torch.sum(delta_sq * current_fim[name])
            
            # Denominator term 2: delta^T * Λ_{t-1} * delta
            denom_term2 += torch.sum(delta_sq * accumulated_fim[name])
    
    # Calculate lambda* using the closed-form solution
    lambda_star = numerator / (denom_term1 + denom_term2 + epsilon)
    lambda_star = torch.clamp(lambda_star, 0.0, 1.0)
    
    return lambda_star


def adaptive_merge_parameters(theta_t_minus_1, theta_t_cand, lambda_star):
    """
    Perform adaptive parameter merging.
    
    Args:
        theta_t_minus_1: Previous task's final parameters
        theta_t_cand: Current task's candidate parameters
        lambda_star: Optimal merging coefficient
        
    Returns:
        dict: Merged parameters
    """
    merged_theta = {}
    
    for name in theta_t_cand:
        if name in theta_t_minus_1:
            # Perform weighted merge: θ* = (1-λ*)θ_{t-1} + λ*θ_cand
            merged_theta[name] = (1 - lambda_star) * theta_t_minus_1[name] + lambda_star * theta_t_cand[name]
        else:
            # For new parameters (e.g., new classifier heads), keep candidate
            merged_theta[name] = theta_t_cand[name]
    
    return merged_theta

