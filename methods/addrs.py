import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import logging
import numpy as np
from tqdm import tqdm

from methods.lorasub_drs import LoRAsub_DRS
from utils.toolkit import tensor2numpy, accuracy
from models.sinet_lora import SiNet
from models.vit_lora import Attention_LoRA
from copy import deepcopy
from utils.schedulers import CosineSchedule

# import ipdb  # Uncomment for debugging
import optimgrad
import re
from collections import defaultdict
from utils.losses import AugmentedTripletLoss
from scipy.spatial.distance import cdist
from utils.fisher_utils import (
    FisherManager,
    compute_diagonal_fim,
    compute_lambda_star,
    adaptive_merge_parameters,
)
from utils.refinement import (
    ClassifierAlignment,
    SelfDistillationLoss,
    apply_self_distillation,
)
from utils.analysis import ADDRSAnalyzer


class AD_DRS(LoRAsub_DRS):
    """
    AD-DRS (Adaptive Merging in Drift-Resistant Space) - Enhanced version of LoRA-Sub-DRS

    This class inherits from LoRAsub_DRS and implements the two-step AD-DRS methodology:
    1. Plasticity-Search Training in DRS (inherited from LoRAsub_DRS)
    2. Adaptive Merging using Bayesian optimization with Fisher Information Matrix
    """

    def __init__(self, args):
        super().__init__(args)

        # AD-DRS specific components
        self.use_classifier_alignment = args.get("use_classifier_alignment", False)
        self.use_self_distillation = args.get("use_self_distillation", False)
        self.fixed_lambda = args.get("fixed_lambda", None)  # For ablation studies

        if self.use_classifier_alignment:
            self.classifier_aligner = ClassifierAlignment(
                feature_dim=args.get("embd_dim", 768)
            )

        if self.use_self_distillation:
            self.self_distill_loss = SelfDistillationLoss(
                temperature=args.get("distill_temperature", 4.0),
                alpha=args.get("distill_alpha", 0.5),
            )

        # Initialize analyzer for comprehensive logging
        experiment_name = (
            f"AD-DRS_{args.get('dataset', 'unknown')}_{args.get('prefix', 'exp')}"
        )
        self.analyzer = ADDRSAnalyzer(experiment_name)

        # Log config after initialization (will be called in after_task if needed)
        # We defer this to avoid device serialization issues
        self._config_logged = False
        self._original_args = args.copy()  # Store original args for later logging

        # AD-DRS specific logging
        logging.info("=" * 50)
        logging.info("Initializing AD-DRS (Adaptive Merging in Drift-Resistant Space)")
        logging.info("=" * 50)
        logging.info("Foundation: LoRA Subtraction for Drift-Resistant Space")
        logging.info("Enhancement: Adaptive Merging via Bayesian Optimization")
        if self.use_classifier_alignment:
            logging.info("Refinement: Classifier Alignment ENABLED")
        if self.use_self_distillation:
            logging.info("Refinement: Self-Distillation ENABLED")
        logging.info("=" * 50)

    def after_task(self):
        """Override to add AD-DRS specific logging, analysis, and refinements."""
        super().after_task()

        # Log config on first task to avoid device serialization issues
        if not self._config_logged:
            self.analyzer.log_config(self._original_args)
            self._config_logged = True

        # Get current performance metrics
        if hasattr(self, "test_loader"):
            task_acc = self._compute_accuracy_domain(self._network, self.test_loader)
        else:
            task_acc = 0.0

        # Calculate average accuracy if we have access to all test data
        avg_acc = (
            sum(self.lambda_history) / len(self.lambda_history) * 100
            if self.lambda_history
            else task_acc
        )

        # Log to analyzer
        self.analyzer.log_task_results(self._cur_task, task_acc, avg_acc)

        # Log Fisher Information norm
        if hasattr(self, "fisher_manager") and self.fisher_manager.accumulated_fisher:
            fisher_norm = sum(
                torch.norm(fim).item()
                for fim in self.fisher_manager.accumulated_fisher.values()
            )
            self.analyzer.log_fisher_norm(self._cur_task, fisher_norm)

        # Log lambda history for analysis
        if self.lambda_history:
            logging.info(f"Lambda history: {self.lambda_history}")
            logging.info(f"Average lambda: {np.mean(self.lambda_history):.4f}")
            logging.info(f"Lambda std: {np.std(self.lambda_history):.4f}")

        # Apply refinement techniques
        if self.use_classifier_alignment and hasattr(self, "train_loader"):
            logging.info("=== Applying Classifier Alignment ===")
            current_classes = range(self._known_classes, self._total_classes)
            self.classifier_aligner.update_class_stats(
                self._network,
                self.train_loader,
                self._device,
                (self._known_classes, self._total_classes),
            )
            self.classifier_aligner.retrain_classifier(self._network, self._device)

        # Additional AD-DRS specific analysis can be added here
        logging.info(f"Completed Task {self._cur_task} with AD-DRS methodology")

    def finalize_experiment(self):
        """Finalize the experiment with comprehensive analysis."""
        if hasattr(self, "analyzer"):
            self.analyzer.finalize_experiment()
            logging.info("Experiment analysis completed and saved")

    def train_function(self, train_loader, test_loader):
        """Override to add self-distillation if enabled."""
        prog_bar = tqdm(range(self.run_epoch))
        criterion = AugmentedTripletLoss(margin=self.margin_inter).to(self._device)

        for _, epoch in enumerate(prog_bar):
            self._network.eval()
            losses = 0.0
            correct, total = 0, 0

            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                mask = (targets >= self._known_classes).nonzero().view(-1)
                inputs = torch.index_select(inputs, 0, mask)
                labels = torch.index_select(targets, 0, mask)
                targets = torch.index_select(targets, 0, mask) - self._known_classes

                ret = self._network(inputs)
                logits = ret["logits"]
                features = ret["features"]
                feature = features / features.norm(dim=-1, keepdim=True)

                # Main losses
                loss = F.cross_entropy(logits, targets)
                ATL = criterion(feature, labels, self._protos)
                loss += self.lambada * ATL

                # Self-distillation loss (if enabled)
                if self.use_self_distillation:
                    distill_loss = apply_self_distillation(self._network, inputs)
                    loss += distill_loss

                self.model_optimizer.zero_grad()
                loss.backward()
                self.model_optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            self.model_scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.run_epoch,
                losses / len(train_loader),
                train_acc,
            )
            prog_bar.set_description(info)

        logging.info(info)
