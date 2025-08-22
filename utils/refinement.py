#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Refinement techniques for AD-DRS: Classifier Alignment and Self-Distillation

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import logging
from copy import deepcopy


class ClassifierAlignment:
    """
    Classifier Alignment technique for AD-DRS refinement.
    Stores class statistics and retrains classifier on synthetic features.
    """
    
    def __init__(self, feature_dim):
        self.feature_dim = feature_dim
        self.class_stats = {}  # {class_id: {'mean': tensor, 'cov': tensor}}
        
    def update_class_stats(self, model, dataloader, device, class_range):
        """
        Update class statistics with features from current task.
        
        Args:
            model: Current model
            dataloader: DataLoader for current task
            device: Device to run on
            class_range: Range of class IDs for current task
        """
        model.eval()
        class_features = {cls: [] for cls in range(class_range[0], class_range[1])}
        
        with torch.no_grad():
            for _, inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Extract features
                if hasattr(model, 'extract_vector'):
                    features = model.extract_vector(inputs)
                else:
                    outputs = model(inputs)
                    features = outputs['features'] if isinstance(outputs, dict) else outputs
                
                # Group features by class
                for i, target in enumerate(targets):
                    if target.item() in class_features:
                        class_features[target.item()].append(features[i].cpu())
        
        # Compute statistics for each class
        for cls, feat_list in class_features.items():
            if feat_list:
                feat_tensor = torch.stack(feat_list)
                mean = feat_tensor.mean(dim=0)
                cov = torch.cov(feat_tensor.T) + torch.eye(feat_tensor.size(1)) * 1e-4  # Regularization
                
                self.class_stats[cls] = {
                    'mean': mean,
                    'cov': cov
                }
                logging.info(f"Updated stats for class {cls}: {len(feat_list)} samples")
    
    def retrain_classifier(self, model, device, num_samples_per_class=100, num_epochs=10, lr=0.001):
        """
        Retrain classifier on synthetic features sampled from stored statistics.
        
        Args:
            model: Model to retrain classifier for
            device: Device to run on
            num_samples_per_class: Number of synthetic samples per class
            num_epochs: Training epochs
            lr: Learning rate
        """
        if not self.class_stats:
            logging.warning("No class statistics available for classifier alignment")
            return
        
        logging.info(f"Retraining classifier with {len(self.class_stats)} classes")
        
        # Generate synthetic features and labels
        synthetic_features = []
        synthetic_labels = []
        
        for cls, stats in self.class_stats.items():
            mean = stats['mean'].to(device)
            cov = stats['cov'].to(device)
            
            # Sample from multivariate normal distribution
            try:
                samples = torch.distributions.MultivariateNormal(mean, cov).sample((num_samples_per_class,))
                synthetic_features.append(samples)
                synthetic_labels.extend([cls] * num_samples_per_class)
            except Exception as e:
                logging.warning(f"Failed to sample for class {cls}: {e}")
                # Fallback: use mean with noise
                noise = torch.randn(num_samples_per_class, mean.size(0), device=device) * 0.1
                samples = mean.unsqueeze(0) + noise
                synthetic_features.append(samples)
                synthetic_labels.extend([cls] * num_samples_per_class)
        
        if not synthetic_features:
            logging.warning("No synthetic features generated")
            return
        
        features_tensor = torch.cat(synthetic_features, dim=0)
        labels_tensor = torch.tensor(synthetic_labels, device=device)
        
        # Create dataset and dataloader
        dataset = TensorDataset(features_tensor, labels_tensor)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
        
        # Freeze backbone, only train classifier
        for param in model.parameters():
            param.requires_grad = False
        
        # Enable gradients for classifier layers
        classifier_params = []
        for name, param in model.named_parameters():
            if 'classifier' in name or 'head' in name:
                param.requires_grad = True
                classifier_params.append(param)
        
        if not classifier_params:
            logging.warning("No classifier parameters found for retraining")
            return
        
        optimizer = torch.optim.Adam(classifier_params, lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        model.train()
        for epoch in range(num_epochs):
            total_loss = 0.0
            for features, labels in dataloader:
                optimizer.zero_grad()
                
                # Forward pass through classifier only
                if hasattr(model, 'interface2'):  # For SiNet
                    logits = model.interface2(features)
                else:
                    # Generic approach
                    logits = model.head(features) if hasattr(model, 'head') else features
                
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if epoch % 5 == 0:
                logging.info(f"Classifier alignment epoch {epoch}: loss = {total_loss/len(dataloader):.4f}")
        
        logging.info("Classifier alignment completed")


class SelfDistillationLoss(nn.Module):
    """
    Self-Distillation loss for AD-DRS refinement.
    Enforces consistency between class token and patch tokens.
    """
    
    def __init__(self, temperature=4.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kld = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, class_logits, patch_logits):
        """
        Compute self-distillation loss.
        
        Args:
            class_logits: Logits from class token
            patch_logits: Logits from patch tokens (averaged)
            
        Returns:
            Self-distillation loss
        """
        # Soften the logits
        class_probs = F.log_softmax(class_logits / self.temperature, dim=1)
        patch_probs = F.softmax(patch_logits / self.temperature, dim=1)
        
        # KL divergence loss
        distill_loss = self.kld(class_probs, patch_probs) * (self.temperature ** 2)
        
        return self.alpha * distill_loss


def apply_self_distillation(model, inputs, temperature=4.0, alpha=0.5):
    """
    Apply self-distillation during training.
    
    Args:
        model: Model with modified forward to return both class and patch tokens
        inputs: Input batch
        temperature: Softmax temperature for distillation
        alpha: Weight for distillation loss
        
    Returns:
        Self-distillation loss
    """
    # This would require modification to the ViT forward pass
    # to return both class token and patch token representations
    
    # Placeholder implementation - would need model architecture changes
    outputs = model(inputs)
    
    if isinstance(outputs, dict) and 'patch_logits' in outputs:
        class_logits = outputs['logits']
        patch_logits = outputs['patch_logits']
        
        distill_loss = SelfDistillationLoss(temperature, alpha)(class_logits, patch_logits)
        return distill_loss
    
    return torch.tensor(0.0, device=inputs.device, requires_grad=True)

