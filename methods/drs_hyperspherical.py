#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DRS-Hyperspherical: Adaptive Merging in Drift-Resistant Space with Spherical Geometry

This implementation combines:
1. LoRA⁻ (subtraction) for parameter-level drift prevention
2. Spherical geometry for feature-level constraints
3. Riemannian projection in tangent spaces
4. 2-phase training with annealing
5. EMA prototypes to handle distribution shift
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import logging
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import re
from collections import defaultdict
from typing import Optional, Dict, List, Tuple

from methods.lorasub_drs import LoRAsub_DRS
from utils.toolkit import tensor2numpy, accuracy
from models.sinet_lora import SiNet
from models.vit_lora import Attention_LoRA
from utils.schedulers import CosineSchedule

from utils.spherical_geometry import (
    SphericalGeometry,
    TangentPCA,
    PrototypeManager,
    gradient_projection_hook,
)
from utils.angular_losses import (
    ArcFaceHead,
    AngularTripletLoss,
    CombinedSphericalLoss,
    linear_anneal,
)


class DRS_Hyperspherical(LoRAsub_DRS):
    """
    DRS-Hyperspherical: Enhanced continual learning with spherical geometry

    Key innovations:
    1. LoRA⁻ subtraction before each task
    2. Spherical normalization of embeddings
    3. Riemannian gradient projection in tangent spaces
    4. DRS construction via PCA on tangent vectors
    5. 2-phase training: warm-up + main with annealing
    6. EMA prototype updates to handle distribution shift
    7. Angular losses: ArcFace + Angular Triplet
    """

    def __init__(self, args):
        super().__init__(args)

        # DRS-Hyperspherical specific parameters
        self.feature_dim = args.get("embd_dim", 768)
        self.pca_energy_threshold = args.get("pca_energy", 0.90)
        self.max_pca_components = args.get("k_max", 128)
        self.ema_momentum = args.get("ema_momentum", 0.97)
        self.per_class_prototypes = args.get("per_class_prototypes", True)
        self.multi_anchor = args.get("multi_anchor", False)
        self.num_anchors = args.get("num_anchors", 3)

        # 2-phase training parameters
        self.warmup_epochs = args.get("epochs_warm", 4)
        self.main_epochs = args.get("epochs_main", 26)
        self.total_epochs = self.warmup_epochs + self.main_epochs

        # ArcFace parameters with annealing
        self.s_start = args.get("s_start", 10.0)
        self.s_end = args.get("s_end", 30.0)
        self.m_start = args.get("m_start", 0.0)
        self.m_end = args.get("m_end", 0.2)

        # Angular Triplet parameters
        self.triplet_margin = args.get("triplet_margin", 0.2)
        self.triplet_weight = args.get("triplet_lambda", 0.5)
        self.triplet_mining = args.get("triplet_mining", "hard")

        # Label smoothing (only for warmup)
        self.label_smoothing = args.get("label_smoothing", 0.05)

        # Core components
        self.spherical_geom = SphericalGeometry()
        self.prototype_manager = None
        self.tangent_pca = None
        self.arcface_head = None
        self.angular_triplet = None
        self.combined_loss = None

        # Training state
        self.current_epoch = 0
        self.current_phase = "warmup"  # 'warmup' or 'main'
        self.gradient_hooks = []

        # Statistics tracking
        self.geodesic_drift_history = []
        self.prototype_shift_history = []
        self.pca_components_history = []

        logging.info("=" * 60)
        logging.info("Initializing DRS-Hyperspherical")
        logging.info("=" * 60)
        logging.info(f"Feature dimension: {self.feature_dim}")
        logging.info(f"PCA energy threshold: {self.pca_energy_threshold}")
        logging.info(f"Max PCA components: {self.max_pca_components}")
        logging.info(f"EMA momentum: {self.ema_momentum}")
        logging.info(
            f"2-phase training: {self.warmup_epochs} warmup + {self.main_epochs} main"
        )
        logging.info(
            f"ArcFace annealing: s({self.s_start}→{self.s_end}), m({self.m_start}→{self.m_end})"
        )
        logging.info(
            f"Angular Triplet: margin={self.triplet_margin}, weight={self.triplet_weight}"
        )
        logging.info("=" * 60)

    def incremental_train(self, data_manager):
        """Override to implement DRS-Hyperspherical training pipeline"""
        self.data_manager = data_manager
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )

        logging.info(f"\n{'=' * 60}")
        logging.info(
            f"DRS-Hyperspherical Task {self._cur_task}: Classes {self._known_classes}-{self._total_classes}"
        )
        logging.info(f"{'=' * 60}")

        # Step 1: Apply LoRA⁻ (subtraction) if not first task
        if self._cur_task > 0:
            self._apply_lora_subtraction()

        # Step 2: Update network structure
        self._network.update_fc(self._total_classes)

        # Step 3: Prepare data loaders
        self._prepare_data_loaders(data_manager)

        # Step 4: Initialize/update spherical components
        self._initialize_spherical_components()

        # Step 5: Build DRS after LoRA⁻
        self._build_drs_hyperspherical()

        # Step 6: 2-phase training
        if not self.eval:
            self._train_hyperspherical()

        # Step 7: Finalize task
        self._finalize_task()

    def _apply_lora_subtraction(self):
        """Apply LoRA⁻: W̃_t = W_0 - Σ_{j=1}^{t-1} ΔW_j"""
        logging.info("Applying LoRA⁻ (subtraction) before task training...")

        # Reset to base weights and subtract all previous LoRA adaptations
        with torch.no_grad():
            for module in self._network.modules():
                if isinstance(module, Attention_LoRA):
                    # Reset current weights to base
                    module.reset_to_base_weights()

                    # Subtract all previous LoRA adaptations
                    for prev_task in range(self._cur_task):
                        if (
                            hasattr(module, f"lora_A_k")
                            and len(module.lora_A_k) > prev_task
                        ):
                            # Subtract ΔW for key
                            delta_W_k = (
                                module.lora_B_k[prev_task].weight
                                @ module.lora_A_k[prev_task].weight
                            )
                            module.attention.key.weight -= delta_W_k

                            # Subtract ΔW for value
                            delta_W_v = (
                                module.lora_B_v[prev_task].weight
                                @ module.lora_A_v[prev_task].weight
                            )
                            module.attention.value.weight -= delta_W_v

        logging.info(f"Applied LoRA⁻ for {self._cur_task} previous tasks")

    def _prepare_data_loaders(self, data_manager):
        """Prepare data loaders for current task"""
        # Current task training data
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        # Full test data (all classes seen so far)
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        # Representative dataset for DRS construction
        repr_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="test",  # Use test mode for clean features
        )
        self.repr_loader = DataLoader(
            repr_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def _initialize_spherical_components(self):
        """Initialize spherical geometry components"""
        current_classes = self._total_classes - self._known_classes

        # Initialize prototype manager
        self.prototype_manager = PrototypeManager(
            feature_dim=self.feature_dim,
            num_classes=current_classes if self.per_class_prototypes else 1,
            momentum=self.ema_momentum,
            per_class=self.per_class_prototypes,
            multi_anchor=self.multi_anchor,
            num_anchors=self.num_anchors,
        )

        # Initialize TangentPCA for DRS
        self.tangent_pca = TangentPCA(
            energy_threshold=self.pca_energy_threshold,
            max_components=self.max_pca_components,
        )

        # Initialize ArcFace head
        self.arcface_head = ArcFaceHead(
            in_features=self.feature_dim,
            out_features=current_classes,
            s_start=self.s_start,
            s_end=self.s_end,
            m_start=self.m_start,
            m_end=self.m_end,
            total_epochs=self.total_epochs,
            warmup_epochs=self.warmup_epochs,
        ).to(self._device)

        # Initialize Angular Triplet Loss
        self.angular_triplet = AngularTripletLoss(
            margin=self.triplet_margin,
            mining_strategy=self.triplet_mining,
            use_prototypes=True,
        )

        # Initialize Combined Loss
        self.combined_loss = CombinedSphericalLoss(
            arcface_head=self.arcface_head,
            triplet_loss=self.angular_triplet,
            triplet_weight=self.triplet_weight,
            label_smoothing=self.label_smoothing,
        )

        logging.info("Initialized spherical components")

    def _build_drs_hyperspherical(self):
        """Build Drift-Resistant Space using spherical geometry"""
        logging.info("Building DRS-Hyperspherical...")

        # Step A: Collect embeddings after LoRA⁻
        embeddings_list = []
        labels_list = []

        self._network.eval()
        with torch.no_grad():
            for _, inputs, targets in self.repr_loader:
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)

                # Get features before classifier
                ret = self._network(inputs)
                features = ret["features"]

                # Normalize to sphere
                normalized_features = self.spherical_geom.normalize_to_sphere(features)

                embeddings_list.append(normalized_features.cpu())
                labels_list.append(
                    (targets - self._known_classes).cpu()
                )  # Local labels

        embeddings = torch.cat(embeddings_list, dim=0).to(self._device)
        labels = torch.cat(labels_list, dim=0).to(self._device)

        # Step B: Initialize prototypes
        self.prototype_manager.initialize_prototypes(embeddings, labels, self._device)

        # Step C: Map to tangent space and compute PCA
        if self.per_class_prototypes:
            # Use class-wise prototypes for tangent mapping
            tangent_vectors_list = []
            for class_idx in range(labels.max().item() + 1):
                class_mask = labels == class_idx
                if class_mask.sum() > 0:
                    class_embeddings = embeddings[class_mask]
                    class_prototype = self.prototype_manager.get_prototype(class_idx)

                    # Map to tangent space at class prototype
                    tangent_vecs = self.spherical_geom.log_map(
                        class_prototype, class_embeddings
                    )
                    tangent_vectors_list.append(tangent_vecs)

            all_tangent_vectors = torch.cat(tangent_vectors_list, dim=0)
        else:
            # Use global prototype
            global_prototype = self.prototype_manager.get_prototype()
            all_tangent_vectors = self.spherical_geom.log_map(
                global_prototype, embeddings
            )

        # Step D: Fit PCA on tangent vectors
        self.tangent_pca.fit(all_tangent_vectors.cpu())
        self.pca_components_history.append(self.tangent_pca.n_components)

        logging.info(f"DRS built with {self.tangent_pca.n_components} components")
        logging.info(
            f"Explained variance: {torch.sum(self.tangent_pca.explained_variance_ratio):.3f}"
        )

    def _train_hyperspherical(self):
        """2-phase training with spherical constraints"""
        logging.info("Starting 2-phase hyperspherical training...")

        # Setup parameter groups and optimizer
        self._setup_training_components()

        # Training loop
        for epoch in range(self.total_epochs):
            self.current_epoch = epoch
            self.current_phase = "warmup" if epoch < self.warmup_epochs else "main"

            # Update ArcFace annealing
            self.arcface_head.set_epoch(epoch)

            # Log phase transition
            if epoch == self.warmup_epochs:
                logging.info(f"{'=' * 40}")
                logging.info(f"Transitioning to MAIN phase (epoch {epoch})")
                logging.info(f"{'=' * 40}")

            # Train one epoch
            self._train_epoch()

            # Update learning rate
            if hasattr(self, "model_scheduler"):
                self.model_scheduler.step()

        logging.info("Completed 2-phase hyperspherical training")

    def _setup_training_components(self):
        """Setup training components for current task"""
        # Enable gradients for current task parameters
        for name, param in self._network.named_parameters():
            param.requires_grad_(False)

            # Enable current task LoRA parameters
            current_task_patterns = [
                f"classifier_pool.{self._network.numtask - 1}",
                f"lora_A_k.{self._network.numtask - 1}",
                f"lora_A_v.{self._network.numtask - 1}",
                f"lora_B_k.{self._network.numtask - 1}",
                f"lora_B_v.{self._network.numtask - 1}",
            ]

            for pattern in current_task_patterns:
                if pattern in name:
                    param.requires_grad_(True)
                    break

        # Setup optimizer
        trainable_params = [p for p in self._network.parameters() if p.requires_grad]
        arcface_params = [p for p in self.arcface_head.parameters()]

        self.model_optimizer = torch.optim.Adam(
            [
                {"params": trainable_params, "lr": self.lrate},
                {"params": arcface_params, "lr": self.fc_lrate},
            ],
            weight_decay=self.weight_decay,
        )

        self.model_scheduler = CosineSchedule(self.model_optimizer, K=self.total_epochs)

    def _train_epoch(self):
        """Train one epoch with spherical constraints"""
        self._network.train()
        self.arcface_head.train()

        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        # Progress bar
        prog_bar = tqdm(
            self.train_loader,
            desc=f"Task {self._cur_task} {self.current_phase.upper()} epoch {self.current_epoch}",
        )

        for batch_idx, (_, inputs, targets) in enumerate(prog_bar):
            inputs, targets = inputs.to(self._device), targets.to(self._device)

            # Filter to current task classes
            mask = (targets >= self._known_classes).nonzero().view(-1)
            if len(mask) == 0:
                continue

            inputs = inputs[mask]
            targets = targets[mask]
            local_targets = targets - self._known_classes  # Convert to local labels

            # Forward pass
            ret = self._network(inputs)
            features = ret["features"]

            # Normalize to sphere
            normalized_features = self.spherical_geom.normalize_to_sphere(features)

            # Attach gradient projection hook
            if self.current_phase == "main":
                # Use both Riemannian and DRS projection
                hook = gradient_projection_hook(
                    self.spherical_geom,
                    self.prototype_manager,
                    self.tangent_pca,
                    local_targets,
                )
            else:
                # Warmup: only Riemannian projection
                hook = gradient_projection_hook(
                    self.spherical_geom,
                    self.prototype_manager,
                    None,  # No DRS projection yet
                    local_targets,
                )

            hook_handle = normalized_features.register_hook(hook)
            self.gradient_hooks.append(hook_handle)

            # Compute losses
            if self.current_phase == "warmup":
                # Warmup: only cross-entropy with label smoothing
                loss_dict = self._compute_warmup_loss(
                    normalized_features, local_targets
                )
            else:
                # Main: combined losses
                loss_dict = self._compute_main_loss(normalized_features, local_targets)

            loss = loss_dict["total_loss"]

            # Backward pass
            self.model_optimizer.zero_grad()
            loss.backward()
            self.model_optimizer.step()

            # Update prototypes with EMA
            with torch.no_grad():
                self.prototype_manager.update_prototypes(
                    normalized_features.detach(), local_targets
                )

            # Statistics
            total_loss += loss.item()

            # Accuracy calculation (using ArcFace logits)
            with torch.no_grad():
                arcface_logits = self.arcface_head(normalized_features, local_targets)
                _, predicted = torch.max(arcface_logits, 1)
                correct_predictions += (predicted == local_targets).sum().item()
                total_samples += local_targets.size(0)

            # Update progress bar
            acc = (
                100.0 * correct_predictions / total_samples
                if total_samples > 0
                else 0.0
            )
            prog_bar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "Acc": f"{acc:.2f}%",
                    "Phase": self.current_phase,
                }
            )

            # Clean up hooks periodically
            if len(self.gradient_hooks) > 100:
                for handle in self.gradient_hooks:
                    handle.remove()
                self.gradient_hooks = []

        # Clean up remaining hooks
        for handle in self.gradient_hooks:
            handle.remove()
        self.gradient_hooks = []

        # Log epoch results
        avg_loss = total_loss / len(self.train_loader)
        epoch_acc = (
            100.0 * correct_predictions / total_samples if total_samples > 0 else 0.0
        )

        # Get current annealing values
        s_current, m_current = self.arcface_head.get_current_params()

        logging.info(
            f"Task {self._cur_task} {self.current_phase.upper()} Epoch {self.current_epoch}: "
            f"Loss={avg_loss:.4f}, Acc={epoch_acc:.2f}%, s={s_current:.1f}, m={m_current:.3f}"
        )

    def _compute_warmup_loss(
        self, embeddings: torch.Tensor, targets: torch.Tensor
    ) -> Dict:
        """Compute loss for warmup phase"""
        # Only cross-entropy with label smoothing
        arcface_logits = self.arcface_head(embeddings, targets)

        if self.label_smoothing > 0:
            loss = self.combined_loss.label_smooth_ce(arcface_logits, targets)
        else:
            loss = F.cross_entropy(arcface_logits, targets)

        return {"total_loss": loss, "ce_loss": loss.item(), "triplet_loss": 0.0}

    def _compute_main_loss(
        self, embeddings: torch.Tensor, targets: torch.Tensor
    ) -> Dict:
        """Compute combined loss for main phase"""
        # Get prototypes for triplet loss
        prototypes = None
        prototype_labels = None

        if self.per_class_prototypes:
            # Use class prototypes as negative mining source
            prototypes = self.prototype_manager.get_prototype()  # All class prototypes
            prototype_labels = torch.arange(prototypes.size(0), device=self._device)

        # Combined loss (ArcFace + Angular Triplet)
        total_loss, loss_dict = self.combined_loss(
            embeddings, targets, prototypes, prototype_labels
        )

        return {
            "total_loss": total_loss,
            "ce_loss": loss_dict.get("ce_loss", 0.0),
            "triplet_loss": loss_dict.get("triplet_loss", 0.0),
        }

    def _finalize_task(self):
        """Finalize current task and update global state"""
        logging.info(f"Finalizing Task {self._cur_task}...")

        # Re-estimate prototypes with final model
        self._reestimate_prototypes()

        # Compute and log geodesic drift
        if self._cur_task > 0:
            drift = self._compute_geodesic_drift()
            self.geodesic_drift_history.append(drift)
            logging.info(f"Geodesic drift: {drift:.4f}")

        # Update global state
        self._known_classes = self._total_classes

        # Build prototypes for evaluation (inherited from base class)
        self._build_protos()

        logging.info(f"Task {self._cur_task} completed successfully")

    def _reestimate_prototypes(self):
        """Re-estimate prototypes with final model"""
        logging.info("Re-estimating prototypes with final model...")

        embeddings_list = []
        labels_list = []

        self._network.eval()
        with torch.no_grad():
            for _, inputs, targets in self.repr_loader:
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)

                ret = self._network(inputs)
                features = ret["features"]
                normalized_features = self.spherical_geom.normalize_to_sphere(features)

                embeddings_list.append(normalized_features.cpu())
                labels_list.append((targets - self._known_classes).cpu())

        embeddings = torch.cat(embeddings_list, dim=0).to(self._device)
        labels = torch.cat(labels_list, dim=0).to(self._device)

        # Reinitialize prototypes (overwrites EMA estimates)
        self.prototype_manager.initialize_prototypes(embeddings, labels, self._device)

        logging.info("Prototypes re-estimated")

    def _compute_geodesic_drift(self) -> float:
        """Compute average geodesic drift of prototypes"""
        if not hasattr(self, "previous_prototypes"):
            return 0.0

        current_prototypes = self.prototype_manager.get_prototype()
        if self.per_class_prototypes:
            # Compute drift for each class
            drifts = []
            min_classes = min(
                current_prototypes.size(0), self.previous_prototypes.size(0)
            )
            for i in range(min_classes):
                drift = self.spherical_geom.geodesic_distance(
                    current_prototypes[i : i + 1], self.previous_prototypes[i : i + 1]
                )
                drifts.append(drift.item())
            return np.mean(drifts) if drifts else 0.0
        else:
            # Global prototype drift
            drift = self.spherical_geom.geodesic_distance(
                current_prototypes, self.previous_prototypes
            )
            return drift.item()

    def eval_task(self):
        """Evaluate current task performance"""
        y_pred, y_true = self._eval_model(
            self.test_loader,
            self._protos / np.linalg.norm(self._protos, axis=1)[:, None],
        )
        nme_accy = self._evaluate(y_pred.T[0], y_true)
        return nme_accy

    def get_hyperspherical_stats(self) -> Dict:
        """Get statistics specific to hyperspherical method"""
        stats = {
            "geodesic_drift_history": self.geodesic_drift_history,
            "pca_components_history": self.pca_components_history,
            "current_pca_components": self.tangent_pca.n_components
            if self.tangent_pca
            else 0,
            "current_pca_variance": torch.sum(
                self.tangent_pca.explained_variance_ratio
            ).item()
            if self.tangent_pca
            else 0.0,
        }

        if self.prototype_manager:
            stats["num_prototypes"] = (
                self.prototype_manager.prototypes.size(0)
                if not self.multi_anchor
                else self.prototype_manager.prototypes.size(0)
                * self.prototype_manager.prototypes.size(1)
            )

        return stats
