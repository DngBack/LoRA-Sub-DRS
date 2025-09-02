#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
import numpy as np
from tqdm import tqdm
import os
import re
from copy import deepcopy
from collections import defaultdict
from scipy.spatial.distance import cdist

from methods.base import BaseLearner
from utils.toolkit import tensor2numpy, accuracy
from utils.neuro_utils import (
    extract_subspace_from_BA,
    extract_subspace_adaptive_k,
    merge_cumulative_subspace,
    project_grad_B,
    project_grad_bi_directional,
    project_grad_bi_directional_simple,
    project_grad_delta_w,
    compute_plasticity_loss,
    save_subspace,
    load_subspace,
    sleep_phase_distill,
    get_lora_modules,
    create_noise_loader,
)
from models.sinet_lora import SiNet
from utils.losses import AugmentedTripletLoss
from utils.schedulers import CosineSchedule
import optimgrad


class NeuroLoRA(BaseLearner):
    """
    Neuro-LoRA: Biologically-inspired continual learning with LoRA
    Combines synaptic consolidation principles with parameter-efficient adaptation
    """

    def __init__(self, args):
        super().__init__(args)

        # Initialize model
        if args["net_type"] == "sip":
            self._network = SiNet(args)
        else:
            raise ValueError("Unknown net: {}.".format(args["net_type"]))

        # Store arguments
        self.args = args
        self.EPSILON = args["EPSILON"]
        self.init_epoch = max(args["init_epoch"], 1)  # Ensure at least 1 epoch
        self.init_lr = args["init_lr"]
        self.init_lr_decay = args["init_lr_decay"]
        self.init_weight_decay = args["init_weight_decay"]
        self.epochs = max(args["epochs"], 1)  # Ensure at least 1 epoch
        self.lrate = args["lrate"]
        self.lrate_decay = args["lrate_decay"]
        self.batch_size = args["batch_size"]
        self.weight_decay = args["weight_decay"]
        self.num_workers = args["num_workers"]
        self.lambada = args["lambada"]
        self.total_sessions = args["total_sessions"]
        self.dataset = args["dataset"]
        self.fc_lrate = args["fc_lrate"]
        self.margin_inter = args["margin_inter"]
        self.eval = args["eval"]
        self._protos = []

        # Neuro-LoRA specific parameters
        self.neuro_config = args.get("neuro_lora", {})
        self.k_per_task = self.neuro_config.get("k_per_task", 4)
        self.K_max = self.neuro_config.get("K_max", 64)
        self.lambda_plast = self.neuro_config.get("lambda_plast", 0.1)
        self.sleep_epochs = self.neuro_config.get("sleep_epochs", 0)
        self.sleep_bs = self.neuro_config.get("sleep_bs", 64)
        self.sleep_batches = self.neuro_config.get("sleep_batches", 10)
        self.sleep_lr = self.neuro_config.get("sleep_lr", 1e-4)

        # Bi-directional gradient projection settings
        self.use_bi_directional = self.neuro_config.get("use_bi_directional", True)
        self.bi_directional_method = self.neuro_config.get(
            "bi_directional_method", "simple"
        )  # "simple", "full", or "delta_w_projected"

        # Adaptive-k subspace building settings
        self.adaptive_k_enabled = self.neuro_config.get("adaptive_k_enabled", False)
        self.energy_threshold = self.neuro_config.get("energy_threshold", 0.95)

        # Subspace storage
        self.S_cumulative = {}  # layer_name -> tensor (d, K)
        self.checkpoint_dir = args.get("checkpoint_dir", "logs/neuro_lora")
        self.fea_in = defaultdict(dict)

        # Initialize LoRA parameters
        for module in self._network.modules():
            if hasattr(module, "init_param"):
                module.init_param()

        self.topk = 1
        self.class_num = self._network.class_num
        self.debug = False

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self.data_manager = data_manager
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)

        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        # Prepare data loaders
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

        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        # Multi-GPU setup
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        if not self.eval:
            self._train(self.train_loader, self.test_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        self._build_protos()

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)

        # Set current task for all LoRA modules
        for module in self._network.modules():
            if hasattr(module, "set_current_task"):
                module.set_current_task(self._cur_task)

        # Load cumulative subspaces from previous tasks
        self._load_cumulative_subspaces()

        # Setup parameter freezing/unfreezing
        self._setup_parameter_training()

        # Extract features from previous tasks (for Neuro-LoRA compatibility)
        with torch.no_grad():
            if self._cur_task > 0:
                for i, (_, inputs, targets) in enumerate(train_loader):
                    inputs, targets = inputs.to(self._device), targets.to(self._device)
                    self._network(inputs, get_cur_x=True)

                for module in self._network.modules():
                    if hasattr(module, "cur_matrix"):
                        self.fea_in[module.lora_A_k[self._cur_task].weight] = deepcopy(
                            module.cur_matrix
                        ).to(self._device)
                        self.fea_in[module.lora_A_v[self._cur_task].weight] = deepcopy(
                            module.cur_matrix
                        ).to(self._device)
                        self.fea_in[module.lora_B_k[self._cur_task].weight] = deepcopy(
                            module.cur_matrix
                        ).to(self._device)
                        self.fea_in[module.lora_B_v[self._cur_task].weight] = deepcopy(
                            module.cur_matrix
                        ).to(self._device)
                        module.cur_matrix.zero_()
                        module.matrix_kv = 0
                        module.n_cur_matrix = 0

        # Initialize optimizers
        self.init_model_optimizer()

        # Set training epochs
        if self._cur_task == 0:
            self.run_epoch = max(self.init_epoch, 1)  # Ensure at least 1 epoch
        else:
            self.update_optim_transforms()
            self.run_epoch = max(self.epochs, 1)  # Ensure at least 1 epoch

        # Training loop
        self.train_function(train_loader, test_loader)

        # Extract and save subspaces after training
        self._extract_and_save_subspaces()

        # Optional sleep phase
        if self.sleep_epochs > 0:
            self._run_sleep_phase()

    def _load_cumulative_subspaces(self):
        """Load cumulative subspaces from previous tasks"""
        self.S_cumulative = {}

        if self._cur_task > 0:
            for name, module in get_lora_modules(self._network):
                # Load key subspaces
                path_k = os.path.join(
                    self.checkpoint_dir, f"subspace_{name.replace('.', '_')}_k.pt"
                )
                S_k = load_subspace(path_k, self._device)
                self.S_cumulative[f"{name}_k"] = (
                    S_k.to(self._device) if S_k is not None else None
                )

                # Load value subspaces
                path_v = os.path.join(
                    self.checkpoint_dir, f"subspace_{name.replace('.', '_')}_v.pt"
                )
                S_v = load_subspace(path_v, self._device)
                self.S_cumulative[f"{name}_v"] = (
                    S_v.to(self._device) if S_v is not None else None
                )
        else:
            for name, module in get_lora_modules(self._network):
                self.S_cumulative[f"{name}_k"] = None
                self.S_cumulative[f"{name}_v"] = None

    def _setup_parameter_training(self):
        """Setup which parameters to train for current task"""
        # Freeze all parameters initially
        for name, param in self._network.named_parameters():
            param.requires_grad_(False)

        # Unfreeze classifier for current task
        for name, param in self._network.named_parameters():
            try:
                if (
                    "classifier_pool"
                    + "."
                    + str(self._network.module.numtask - 1)
                    + "."
                    in name
                ):
                    param.requires_grad_(True)
            except:
                if (
                    "classifier_pool" + "." + str(self._network.numtask - 1) + "."
                    in name
                ):
                    param.requires_grad_(True)

            # Unfreeze LoRA parameters for current task
            try:
                if (
                    "lora_A_k" + "." + str(self._network.module.numtask - 1) + "."
                    in name
                ):
                    param.requires_grad_(True)
                if (
                    "lora_A_v" + "." + str(self._network.module.numtask - 1) + "."
                    in name
                ):
                    param.requires_grad_(True)
                if (
                    "lora_B_k" + "." + str(self._network.module.numtask - 1) + "."
                    in name
                ):
                    param.requires_grad_(True)
                if (
                    "lora_B_v" + "." + str(self._network.module.numtask - 1) + "."
                    in name
                ):
                    param.requires_grad_(True)
            except:
                if "lora_A_k" + "." + str(self._network.numtask - 1) + "." in name:
                    param.requires_grad_(True)
                if "lora_A_v" + "." + str(self._network.numtask - 1) + "." in name:
                    param.requires_grad_(True)
                if "lora_B_k" + "." + str(self._network.numtask - 1) + "." in name:
                    param.requires_grad_(True)
                if "lora_B_v" + "." + str(self._network.numtask - 1) + "." in name:
                    param.requires_grad_(True)

    def train_function(self, train_loader, test_loader):
        """Main training loop with Neuro-LoRA components"""
        # Ensure at least 1 epoch to avoid empty range
        epochs_to_run = max(self.run_epoch, 1)
        prog_bar = tqdm(range(epochs_to_run))
        criterion = AugmentedTripletLoss(margin=self.margin_inter).to(self._device)

        for _, epoch in enumerate(prog_bar):
            self._network.eval()
            losses = 0.0
            correct, total = 0, 0

            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                # Filter current task samples
                mask = (targets >= self._known_classes).nonzero().view(-1)
                inputs = torch.index_select(inputs, 0, mask)
                labels = torch.index_select(targets, 0, mask)
                targets = torch.index_select(targets, 0, mask) - self._known_classes

                # Forward pass
                ret = self._network(inputs)
                logits = ret["logits"]
                features = ret["features"]
                feature = features / features.norm(dim=-1, keepdim=True)

                # Compute losses
                loss = torch.nn.functional.cross_entropy(logits, targets)
                ATL = criterion(feature, labels, self._protos)
                loss += self.lambada * ATL

                # Neuro-LoRA: Add plasticity loss
                if self.lambda_plast > 0:
                    plast_loss = 0.0
                    for name, module in get_lora_modules(self._network):
                        if hasattr(module, "get_lora_activation_k"):
                            # Get LoRA activations for plasticity loss
                            act_k = module.get_lora_activation_k(
                                inputs.view(inputs.size(0), -1)
                            )
                            act_v = module.get_lora_activation_v(
                                inputs.view(inputs.size(0), -1)
                            )
                            plast_loss += compute_plasticity_loss(act_k)
                            plast_loss += compute_plasticity_loss(act_v)
                    loss += self.lambda_plast * plast_loss

                # Backward pass
                self.model_optimizer.zero_grad()
                loss.backward()

                # Neuro-LoRA: Project gradients
                if self._cur_task > 0:
                    self._project_gradients()

                # Optimizer step
                self.model_optimizer.step()

                # Update metrics
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

    def _project_gradients(self):
        """Project gradients using bi-directional projection for better subspace protection"""
        if not self.use_bi_directional:
            # Fallback to original method (only B projection)
            for name, param in self._network.named_parameters():
                if param.grad is not None and "lora_B" in name:
                    # Extract module name and projection type (k or v)
                    parts = name.split(".")
                    if len(parts) >= 2:
                        module_name = parts[0]
                        proj_type = (
                            "k"
                            if "lora_B_k" in name
                            else "v"
                            if "lora_B_v" in name
                            else None
                        )

                        if proj_type:
                            subspace_key = f"{module_name}_{proj_type}"
                            S = self.S_cumulative.get(subspace_key, None)
                            if S is not None:
                                gB_proj = project_grad_B(param.grad, S)
                                param.grad.data.copy_(gB_proj)
        else:
            # Bi-directional gradient projection
            # Group parameters by module for coordinated projection
            module_grads = defaultdict(dict)

            # Collect gradients for each module
            for name, param in self._network.named_parameters():
                if param.grad is not None and ("lora_A" in name or "lora_B" in name):
                    parts = name.split(".")
                    if len(parts) >= 2:
                        module_name = parts[0]
                        param_type = None
                        if "lora_A_k" in name:
                            param_type = "A_k"
                        elif "lora_A_v" in name:
                            param_type = "A_v"
                        elif "lora_B_k" in name:
                            param_type = "B_k"
                        elif "lora_B_v" in name:
                            param_type = "B_v"

                        if param_type:
                            if module_name not in module_grads:
                                module_grads[module_name] = {}
                            module_grads[module_name][param_type] = param.grad

            # Apply bi-directional projection for each module
            for module_name, grads in module_grads.items():
                # Project key gradients
                if "A_k" in grads and "B_k" in grads:
                    subspace_key = f"{module_name}_k"
                    S = self.S_cumulative.get(subspace_key, None)
                    if S is not None:
                        # Get current A and B matrices for this module
                        module = self._network.get_submodule(module_name)
                        if hasattr(module, "get_A_k") and hasattr(module, "get_B_k"):
                            A_k = module.get_A_k()
                            B_k = module.get_B_k()

                            if self.bi_directional_method == "delta_w_projected":
                                gA_proj, gB_proj = project_grad_delta_w(
                                    grads["A_k"], grads["B_k"], A_k, B_k, S
                                )
                            elif self.bi_directional_method == "full":
                                gA_proj, gB_proj = project_grad_bi_directional(
                                    grads["A_k"], grads["B_k"], A_k, B_k, S
                                )
                            else:  # simple method
                                gA_proj, gB_proj = project_grad_bi_directional_simple(
                                    grads["A_k"], grads["B_k"], A_k, B_k, S
                                )

                            # Update gradients
                            for name, param in self._network.named_parameters():
                                if f"{module_name}.lora_A_k" in name:
                                    param.grad.data.copy_(gA_proj)
                                elif f"{module_name}.lora_B_k" in name:
                                    param.grad.data.copy_(gB_proj)

                # Project value gradients
                if "A_v" in grads and "B_v" in grads:
                    subspace_key = f"{module_name}_v"
                    S = self.S_cumulative.get(subspace_key, None)
                    if S is not None:
                        # Get current A and B matrices for this module
                        module = self._network.get_submodule(module_name)
                        if hasattr(module, "get_A_v") and hasattr(module, "get_B_v"):
                            A_v = module.get_A_v()
                            B_v = module.get_B_v()

                            if self.bi_directional_method == "delta_w_projected":
                                gA_proj, gB_proj = project_grad_delta_w(
                                    grads["A_v"], grads["B_v"], A_v, B_v, S
                                )
                            elif self.bi_directional_method == "full":
                                gA_proj, gB_proj = project_grad_bi_directional(
                                    grads["A_v"], grads["B_v"], A_v, B_v, S
                                )
                            else:  # simple method
                                gA_proj, gB_proj = project_grad_bi_directional_simple(
                                    grads["A_v"], grads["B_v"], A_v, B_v, S
                                )

                            # Update gradients
                            for name, param in self._network.named_parameters():
                                if f"{module_name}.lora_A_v" in name:
                                    param.grad.data.copy_(gA_proj)
                                elif f"{module_name}.lora_B_v" in name:
                                    param.grad.data.copy_(gB_proj)

    def _extract_and_save_subspaces(self):
        """Extract new subspaces and merge with cumulative subspaces"""
        logging.info(f"Extracting subspaces for task {self._cur_task}")

        # Set current task for all LoRA modules
        for name, module in get_lora_modules(self._network):
            if hasattr(module, "set_current_task"):
                module.set_current_task(self._cur_task)

        # Extract subspaces from each LoRA module
        for name, module in get_lora_modules(self._network):
            if hasattr(module, "get_A_k") and hasattr(module, "get_B_k"):
                # Extract subspaces for key projection
                A_k = module.get_A_k()
                B_k = module.get_B_k()

                if self.adaptive_k_enabled:
                    S_k_new = extract_subspace_adaptive_k(
                        B_k,
                        A_k,
                        energy_threshold=self.energy_threshold,
                        k_max=self.k_per_task,
                    )
                else:
                    S_k_new = extract_subspace_from_BA(B_k, A_k, self.k_per_task)

                # Extract subspaces for value projection
                A_v = module.get_A_v()
                B_v = module.get_B_v()

                if self.adaptive_k_enabled:
                    S_v_new = extract_subspace_adaptive_k(
                        B_v,
                        A_v,
                        energy_threshold=self.energy_threshold,
                        k_max=self.k_per_task,
                    )
                else:
                    S_v_new = extract_subspace_from_BA(B_v, A_v, self.k_per_task)

                # Merge with cumulative subspaces
                S_k_cum = self.S_cumulative.get(f"{name}_k", None)
                S_v_cum = self.S_cumulative.get(f"{name}_v", None)

                if S_k_cum is not None:
                    S_k_merged = merge_cumulative_subspace(S_k_cum, S_k_new, self.K_max)
                else:
                    S_k_merged = S_k_new

                if S_v_cum is not None:
                    S_v_merged = merge_cumulative_subspace(S_v_cum, S_v_new, self.K_max)
                else:
                    S_v_merged = S_v_new

                # Update cumulative subspaces
                self.S_cumulative[f"{name}_k"] = S_k_merged
                self.S_cumulative[f"{name}_v"] = S_v_merged

                # Save subspaces to disk
                save_subspace(
                    S_k_merged,
                    os.path.join(
                        self.checkpoint_dir, f"subspace_{name.replace('.', '_')}_k.pt"
                    ),
                )
                save_subspace(
                    S_v_merged,
                    os.path.join(
                        self.checkpoint_dir, f"subspace_{name.replace('.', '_')}_v.pt"
                    ),
                )

                logging.info(
                    f"Extracted and saved subspaces for {name}: k={S_k_merged.shape}, v={S_v_merged.shape}"
                )

        logging.info(f"Subspace extraction completed for task {self._cur_task}")

    def _run_sleep_phase(self):
        """Run sleep-phase consolidation"""
        logging.info("Running sleep-phase consolidation...")

        # Check if there are trainable parameters
        trainable_params = [p for p in self._network.parameters() if p.requires_grad]
        if not trainable_params:
            logging.warning(
                "No trainable parameters found for sleep phase consolidation"
            )
            return

        # Create noise dataloader
        noise_loader = create_noise_loader(
            batch_size=self.sleep_bs, n_batches=self.sleep_batches, device=self._device
        )

        # Create teacher model (frozen)
        teacher_model = deepcopy(self._network)
        for p in teacher_model.parameters():
            p.requires_grad = False

        # Run distillation
        try:
            sleep_phase_distill(
                teacher_model,
                self._network,
                noise_loader,
                device=self._device,
                epochs=self.sleep_epochs,
                lr=self.sleep_lr,
            )
            logging.info("Sleep-phase consolidation completed successfully")
        except Exception as e:
            logging.error(f"Error during sleep phase consolidation: {e}")
            logging.info("Continuing without sleep phase consolidation")

    def _build_protos(self):
        """Build prototypes for triplet loss"""
        self._network.to(self._device)
        with torch.no_grad():
            for class_idx in range(self._known_classes, self._total_classes):
                data, targets, idx_dataset = self.data_manager.get_dataset(
                    np.arange(class_idx, class_idx + 1),
                    source="train",
                    mode="test",
                    ret_data=True,
                )
                idx_loader = DataLoader(
                    idx_dataset,
                    batch_size=self.args["batch_size"],
                    shuffle=False,
                    num_workers=4,
                )
                vectors, _ = self._extract_vectors(idx_loader)
                class_mean = np.mean(vectors, axis=0)
                self._protos.append(class_mean)

    def _extract_vectors(self, loader):
        """Extract feature vectors from data loader"""
        self._network.eval()
        vectors = []
        targets = []
        for i, (_, inputs, targets_batch) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                features = self._network.extract_vector(inputs)
            vectors.append(features.cpu().numpy())
            targets.append(targets_batch.numpy())
        return np.concatenate(vectors), np.concatenate(targets)

    def _evaluate(self, y_pred, y_true):
        """Evaluate predictions"""
        ret = {}
        print(len(y_pred), len(y_true))
        grouped = accuracy(y_pred, y_true, self._known_classes, self.class_num)
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        return ret

    def eval_task(self):
        """Evaluate current task"""
        y_pred, y_true = self._eval_model(
            self.test_loader,
            self._protos / np.linalg.norm(self._protos, axis=1)[:, None],
        )
        nme_accy = self._evaluate(y_pred.T[0], y_true)
        return nme_accy

    def _eval_model(self, loader, class_means):
        """Evaluate model on data loader"""
        self._network.eval()
        vectors, y_true = self._extract_vectors(loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + self.EPSILON)).T

        dists = cdist(class_means, vectors, "sqeuclidean")
        scores = dists.T

        return np.argsort(scores, axis=1)[:, : self.topk], y_true  # [N, topk]

    def init_model_optimizer(self):
        """Initialize optimizer for current task"""
        if self._cur_task == 0:
            lr = self.init_lr
        else:
            lr = self.lrate

        fea_params = [
            p
            for n, p in self._network.named_parameters()
            if not bool(re.search("classifier_pool", n)) and p.requires_grad == True
        ]

        cls_params = [
            p
            for n, p in self._network.named_parameters()
            if bool(re.search("classifier_pool", n))
        ]
        model_optimizer_arg = {
            "params": [
                {"params": fea_params, "svd": True, "lr": lr, "thres": 0.99},
                {
                    "params": cls_params,
                    "weight_decay": self.weight_decay,
                    "lr": self.fc_lrate,
                },
            ],
            "weight_decay": self.weight_decay,
            "betas": (0.9, 0.999),
        }

        self.model_optimizer = getattr(optimgrad, self.args["optim"])(
            **model_optimizer_arg
        )
        # Ensure epochs is at least 2 to avoid division by zero in CosineSchedule
        scheduler_epochs = max(self.epochs, 2)
        self.model_scheduler = CosineSchedule(self.model_optimizer, K=scheduler_epochs)

    def update_optim_transforms(self):
        """Update optimizer and transforms for incremental learning"""
        self.model_optimizer.get_eigens(self.fea_in)
        self.model_optimizer.get_transforms()
        self.fea_in = defaultdict(dict)
