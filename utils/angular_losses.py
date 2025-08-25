#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Angular Losses for DRS-Hyperspherical

Implements:
- ArcFace head with annealing (s and m parameters)
- Angular Triplet Loss using geodesic distances
- Combined loss functions for spherical embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
import logging

from .spherical_geometry import SphericalGeometry


class AnnealingScheduler:
    """Handles parameter annealing for s and m in ArcFace"""

    def __init__(
        self,
        start_value: float,
        end_value: float,
        total_epochs: int,
        warmup_epochs: int = 0,
        schedule_type: str = "linear",
    ):
        self.start_value = start_value
        self.end_value = end_value
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.schedule_type = schedule_type

    def get_value(self, epoch: int) -> float:
        """Get annealed value for current epoch"""
        if epoch < self.warmup_epochs:
            # During warmup, use start value
            return self.start_value

        # After warmup, anneal from start to end
        progress = (epoch - self.warmup_epochs) / (
            self.total_epochs - self.warmup_epochs
        )
        progress = max(0.0, min(1.0, progress))

        if self.schedule_type == "linear":
            return self.start_value + progress * (self.end_value - self.start_value)
        elif self.schedule_type == "cosine":
            return self.start_value + 0.5 * (self.end_value - self.start_value) * (
                1 + math.cos(math.pi * (1 - progress))
            )
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")


class ArcFaceHead(nn.Module):
    """
    ArcFace head with adaptive margin and scale annealing

    Paper: ArcFace: Additive Angular Margin Loss for Deep Face Recognition
    Modified for continual learning with annealing schedule
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        s_start: float = 10.0,
        s_end: float = 30.0,
        m_start: float = 0.0,
        m_end: float = 0.2,
        total_epochs: int = 30,
        warmup_epochs: int = 5,
        easy_margin: bool = False,
        eps: float = 1e-7,
    ):
        super(ArcFaceHead, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.easy_margin = easy_margin
        self.eps = eps

        # Initialize normalized weight matrix
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Annealing schedulers
        self.s_scheduler = AnnealingScheduler(
            s_start, s_end, total_epochs, warmup_epochs
        )
        self.m_scheduler = AnnealingScheduler(
            m_start, m_end, total_epochs, warmup_epochs
        )

        # Precompute constants for optimization
        self.cos_m = math.cos(m_end)
        self.sin_m = math.sin(m_end)
        self.th = math.cos(math.pi - m_end)
        self.mm = math.sin(math.pi - m_end) * m_end

        self.current_epoch = 0

    def set_epoch(self, epoch: int):
        """Update current epoch for annealing"""
        self.current_epoch = epoch

    def get_current_params(self) -> Tuple[float, float]:
        """Get current s and m values"""
        s = self.s_scheduler.get_value(self.current_epoch)
        m = self.m_scheduler.get_value(self.current_epoch)
        return s, m

    def forward(self, input: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with ArcFace loss

        Args:
            input: Normalized input features [N, in_features]
            label: Ground truth labels [N]

        Returns:
            Logits for softmax [N, out_features]
        """
        # Normalize weights to unit sphere
        normalized_weight = F.normalize(self.weight, p=2, dim=1)

        # Ensure input is normalized (should be done externally, but safety check)
        normalized_input = F.normalize(input, p=2, dim=1)

        # Cosine similarity
        cosine = F.linear(normalized_input, normalized_weight)  # [N, out_features]
        cosine = torch.clamp(cosine, -1 + self.eps, 1 - self.eps)

        # Get current annealed parameters
        s, m = self.get_current_params()

        # Compute angles
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * math.cos(m) - sine * math.sin(m)

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Create one-hot encoding
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # Apply margin only to ground truth classes
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= s

        return output


class AngularTripletLoss(nn.Module):
    """
    Angular Triplet Loss using geodesic distances on hypersphere

    Modified triplet loss that uses angular distances instead of Euclidean
    Supports mining from both current batch and prototype memory
    """

    def __init__(
        self,
        margin: float = 0.2,
        mining_strategy: str = "hard",  # 'hard', 'semi_hard', 'all'
        use_prototypes: bool = True,
        distance_type: str = "geodesic",  # 'geodesic', 'cosine'
        eps: float = 1e-7,
    ):
        super(AngularTripletLoss, self).__init__()

        self.margin = margin
        self.mining_strategy = mining_strategy
        self.use_prototypes = use_prototypes
        self.distance_type = distance_type
        self.eps = eps

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        prototypes: Optional[torch.Tensor] = None,
        prototype_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute Angular Triplet Loss

        Args:
            embeddings: Normalized embeddings [N, d]
            labels: Labels [N]
            prototypes: Optional prototype embeddings [M, d]
            prototype_labels: Optional prototype labels [M]

        Returns:
            Triplet loss scalar
        """
        device = embeddings.device
        batch_size = embeddings.size(0)

        if batch_size < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Combine batch embeddings with prototypes if available
        if self.use_prototypes and prototypes is not None:
            all_embeddings = torch.cat([embeddings, prototypes], dim=0)
            all_labels = torch.cat([labels, prototype_labels], dim=0)
        else:
            all_embeddings = embeddings
            all_labels = labels

        # Compute distance matrix
        if self.distance_type == "geodesic":
            distances = self._compute_geodesic_distances(all_embeddings)
        else:
            distances = self._compute_cosine_distances(all_embeddings)

        # Extract valid triplets
        triplets = self._mine_triplets(distances, all_labels, batch_size)

        if len(triplets) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Compute triplet loss
        anchor_pos_dist = distances[triplets[:, 0], triplets[:, 1]]
        anchor_neg_dist = distances[triplets[:, 0], triplets[:, 2]]

        loss = F.relu(anchor_pos_dist - anchor_neg_dist + self.margin)

        return loss.mean()

    def _compute_geodesic_distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute pairwise geodesic distances"""
        n = embeddings.size(0)
        distances = torch.zeros(n, n, device=embeddings.device)

        for i in range(n):
            for j in range(i + 1, n):
                dist = SphericalGeometry.geodesic_distance(
                    embeddings[i : i + 1], embeddings[j : j + 1]
                )
                distances[i, j] = distances[j, i] = dist

        return distances

    def _compute_cosine_distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute pairwise cosine distances (1 - cosine_similarity)"""
        cosine_sim = torch.mm(embeddings, embeddings.t())
        cosine_sim = torch.clamp(cosine_sim, -1 + self.eps, 1 - self.eps)
        return 1.0 - cosine_sim

    def _mine_triplets(
        self, distances: torch.Tensor, labels: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        """Mine triplets based on strategy"""
        device = distances.device
        n = distances.size(0)

        # Create masks for positive and negative pairs
        label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
        label_ne = ~label_eq

        # Only consider anchors from the current batch
        triplets = []

        for i in range(batch_size):  # Only anchors from batch
            # Find positive and negative indices
            pos_mask = label_eq[i] & (torch.arange(n, device=device) != i)
            neg_mask = label_ne[i]

            pos_indices = torch.where(pos_mask)[0]
            neg_indices = torch.where(neg_mask)[0]

            if len(pos_indices) == 0 or len(neg_indices) == 0:
                continue

            if self.mining_strategy == "hard":
                # Hardest positive (farthest)
                pos_distances = distances[i, pos_indices]
                hardest_pos = pos_indices[torch.argmax(pos_distances)]

                # Hardest negative (closest)
                neg_distances = distances[i, neg_indices]
                hardest_neg = neg_indices[torch.argmin(neg_distances)]

                triplets.append([i, hardest_pos.item(), hardest_neg.item()])

            elif self.mining_strategy == "semi_hard":
                # Semi-hard negatives: closer than positive but still negative
                for pos_idx in pos_indices:
                    pos_dist = distances[i, pos_idx]

                    # Find negatives closer than this positive
                    valid_negs = neg_indices[
                        distances[i, neg_indices] < pos_dist + self.margin
                    ]
                    valid_negs = valid_negs[distances[i, valid_negs] > pos_dist]

                    if len(valid_negs) > 0:
                        # Choose hardest among valid
                        chosen_neg = valid_negs[torch.argmin(distances[i, valid_negs])]
                        triplets.append([i, pos_idx.item(), chosen_neg.item()])

            elif self.mining_strategy == "all":
                # All valid combinations
                for pos_idx in pos_indices:
                    for neg_idx in neg_indices:
                        triplets.append([i, pos_idx.item(), neg_idx.item()])

        if len(triplets) == 0:
            return torch.empty(0, 3, dtype=torch.long, device=device)

        return torch.tensor(triplets, device=device)


class SphericalContrastiveLoss(nn.Module):
    """
    Contrastive loss using angular distances
    Alternative to triplet loss for some scenarios
    """

    def __init__(self, margin: float = 0.5, distance_type: str = "geodesic"):
        super(SphericalContrastiveLoss, self).__init__()
        self.margin = margin
        self.distance_type = distance_type

    def forward(
        self, embeddings1: torch.Tensor, embeddings2: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss

        Args:
            embeddings1, embeddings2: Normalized embeddings [N, d]
            labels: Binary labels (1 for positive pairs, 0 for negative)

        Returns:
            Contrastive loss
        """
        if self.distance_type == "geodesic":
            distances = SphericalGeometry.geodesic_distance(embeddings1, embeddings2)
        else:
            cosine_sim = torch.sum(embeddings1 * embeddings2, dim=1)
            distances = 1.0 - cosine_sim

        # Positive pairs: minimize distance
        pos_loss = labels.float() * torch.pow(distances, 2)

        # Negative pairs: maximize distance up to margin
        neg_loss = (1.0 - labels.float()) * torch.pow(
            F.relu(self.margin - distances), 2
        )

        return torch.mean(pos_loss + neg_loss)


class CombinedSphericalLoss(nn.Module):
    """
    Combined loss function for DRS-Hyperspherical training
    Combines ArcFace + Angular Triplet + optional regularization
    """

    def __init__(
        self,
        arcface_head: ArcFaceHead,
        triplet_loss: AngularTripletLoss,
        triplet_weight: float = 0.5,
        label_smoothing: float = 0.0,
        regularization_weight: float = 0.0,
    ):
        super(CombinedSphericalLoss, self).__init__()

        self.arcface_head = arcface_head
        self.triplet_loss = triplet_loss
        self.triplet_weight = triplet_weight
        self.label_smoothing = label_smoothing
        self.regularization_weight = regularization_weight

        if label_smoothing > 0:
            self.label_smooth_ce = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        else:
            self.label_smooth_ce = None

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        prototypes: Optional[torch.Tensor] = None,
        prototype_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss

        Args:
            embeddings: Normalized embeddings [N, d]
            labels: Ground truth labels [N]
            prototypes: Optional prototype embeddings
            prototype_labels: Optional prototype labels

        Returns:
            total_loss: Combined loss tensor
            loss_dict: Dictionary with individual loss components
        """
        loss_dict = {}

        # ArcFace loss (Cross-entropy with angular margin)
        arcface_logits = self.arcface_head(embeddings, labels)

        if self.label_smooth_ce is not None:
            ce_loss = self.label_smooth_ce(arcface_logits, labels)
        else:
            ce_loss = F.cross_entropy(arcface_logits, labels)

        loss_dict["ce_loss"] = ce_loss.item()

        # Angular Triplet loss
        triplet_loss = self.triplet_loss(
            embeddings, labels, prototypes, prototype_labels
        )
        loss_dict["triplet_loss"] = triplet_loss.item()

        # Combine losses
        total_loss = ce_loss + self.triplet_weight * triplet_loss

        # Optional regularization (e.g., weight decay on prototypes)
        if self.regularization_weight > 0 and prototypes is not None:
            reg_loss = self.regularization_weight * torch.norm(prototypes, p=2)
            total_loss += reg_loss
            loss_dict["regularization"] = reg_loss.item()

        loss_dict["total_loss"] = total_loss.item()

        return total_loss, loss_dict


class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing cross entropy for improved training stability"""

    def __init__(self, smoothing: float = 0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Apply label smoothing

        Args:
            pred: Predictions [N, num_classes]
            target: Targets [N]

        Returns:
            Smoothed cross entropy loss
        """
        num_classes = pred.size(1)

        # Convert to one-hot and apply smoothing
        target_one_hot = torch.zeros_like(pred)
        target_one_hot.scatter_(1, target.unsqueeze(1), 1)

        smoothed_target = (
            1 - self.smoothing
        ) * target_one_hot + self.smoothing / num_classes

        log_probs = F.log_softmax(pred, dim=1)
        loss = -(smoothed_target * log_probs).sum(dim=1).mean()

        return loss


# Utility functions for annealing
def linear_anneal(
    epoch: int, start_epoch: int, end_epoch: int, start_val: float, end_val: float
) -> float:
    """Linear annealing between two values"""
    if epoch <= start_epoch:
        return start_val
    elif epoch >= end_epoch:
        return end_val
    else:
        progress = (epoch - start_epoch) / (end_epoch - start_epoch)
        return start_val + progress * (end_val - start_val)


def cosine_anneal(
    epoch: int, start_epoch: int, end_epoch: int, start_val: float, end_val: float
) -> float:
    """Cosine annealing between two values"""
    if epoch <= start_epoch:
        return start_val
    elif epoch >= end_epoch:
        return end_val
    else:
        progress = (epoch - start_epoch) / (end_epoch - start_epoch)
        cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
        return end_val + (start_val - end_val) * cosine_factor
