import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import logging
import numpy as np
from tqdm import tqdm

from methods.base import BaseLearner
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


class LoRAsub_DRS(BaseLearner):
    def __init__(self, args):
        super().__init__(args)

        if args["net_type"] == "sip":
            self._network = SiNet(args)
        else:
            raise ValueError("Unknown net: {}.".format(args["net_type"]))

        self.args = args
        self.EPSILON = args["EPSILON"]
        self.init_epoch = args["init_epoch"]
        self.init_lr = args["init_lr"]
        self.init_lr_decay = args["init_lr_decay"]
        self.init_weight_decay = args["init_weight_decay"]
        self.epochs = args["epochs"]
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

        self.topk = 1  # origin is 5
        self.class_num = self._network.class_num
        self.debug = False
        self.fea_in = defaultdict(dict)

        # AD-DRS specific components
        self.fisher_manager = FisherManager()
        self.theta_t_minus_1_star = None  # Previous task's final parameters
        self.lambda_history = []  # Track lambda values for analysis

        for module in self._network.modules():
            if isinstance(module, Attention_LoRA):
                module.init_param()

    def after_task(self):
        self._known_classes = self._total_classes
        # logging.info('Exemplar size: {}'.format(self.exemplar_size))

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

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        if not self.eval:
            self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        self._build_protos()

    def _train(self, train_loader, test_loader):
        """
        AD-DRS Training Pipeline:
        Step 1: Plasticity-Search Training in DRS (produces candidate model)
        Step 2: Adaptive Merging to find optimal balance
        """
        self._network.to(self._device)

        # Store previous task's final parameters (Step 1 preparation)
        if self._cur_task > 0 and self.theta_t_minus_1_star is None:
            self.theta_t_minus_1_star = {
                name: p.clone().detach()
                for name, p in self._network.named_parameters()
                if p.requires_grad
            }
            logging.info(
                f"Stored theta_t_minus_1_star with {len(self.theta_t_minus_1_star)} parameters"
            )

        # STEP 1: PLASTICITY-SEARCH TRAINING IN DRS
        logging.info(
            f"=== AD-DRS Step 1: Plasticity-Search Training for Task {self._cur_task} ==="
        )

        for name, param in self._network.named_parameters():
            param.requires_grad_(False)
            try:
                if (
                    "classifier_pool"
                    + "."
                    + str(self._network.module.numtask - 1)
                    + "."
                    in name
                ):
                    param.requires_grad_(True)
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
                if (
                    "classifier_pool" + "." + str(self._network.numtask - 1) + "."
                    in name
                ):
                    param.requires_grad_(True)
                if "lora_A_k" + "." + str(self._network.numtask - 1) + "." in name:
                    param.requires_grad_(True)
                if "lora_A_v" + "." + str(self._network.numtask - 1) + "." in name:
                    param.requires_grad_(True)
                if "lora_B_k" + "." + str(self._network.numtask - 1) + "." in name:
                    param.requires_grad_(True)
                if "lora_B_v" + "." + str(self._network.numtask - 1) + "." in name:
                    param.requires_grad_(True)

        # Double check
        enabled = set()
        for name, param in self._network.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        with torch.no_grad():
            if self._cur_task > 0:
                for i, (_, inputs, targets) in enumerate(train_loader):
                    inputs, targets = inputs.to(self._device), targets.to(self._device)
                    self._network(inputs, get_cur_x=True)

                for module in self._network.modules():
                    if isinstance(module, Attention_LoRA):
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

            self.init_model_optimizer()
            if self._cur_task == 0:
                self.run_epoch = self.init_epoch
            else:
                self.update_optim_transforms()
                self.run_epoch = self.epochs

        # Perform plasticity-search training (produces candidate model)
        self.train_function(train_loader, test_loader)

        # STEP 2: ADAPTIVE MERGING
        if self._cur_task > 0:
            logging.info(
                f"=== AD-DRS Step 2: Adaptive Merging for Task {self._cur_task} ==="
            )
            self.adaptive_merge_step(train_loader)

        return

    def adaptive_merge_step(self, train_loader):
        """
        Implement Step 2 of AD-DRS: Adaptive Merging to find optimal balance.
        """
        # Get candidate model parameters (result of Step 1)
        theta_t_cand = {
            name: p.clone().detach()
            for name, p in self._network.named_parameters()
            if p.requires_grad and name in self.theta_t_minus_1_star
        }

        logging.info(f"Computing Fisher Information Matrix for current task...")

        # Create a special dataloader for FIM computation with proper label mapping
        # We need to use the test_loader or create a new loader with proper labels
        test_dataset = self.data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        fim_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        # Compute FIM for current task with candidate model
        current_fim = compute_diagonal_fim(self._network, fim_loader, self._device)

        # Get accumulated FIM from previous tasks
        accumulated_fim = self.fisher_manager.get_fisher()

        if not accumulated_fim:
            # First task after initial - no merging needed
            logging.info("No previous FIM found, skipping adaptive merging")
            lambda_star = torch.tensor(1.0)
        else:
            # Check if using fixed lambda for ablation studies
            if hasattr(self, "fixed_lambda") and self.fixed_lambda is not None:
                lambda_star = torch.tensor(float(self.fixed_lambda))
                logging.info(
                    f"Using fixed lambda = {lambda_star.item():.6f} (ablation study)"
                )
            else:
                # Calculate optimal lambda using Bayesian merging theory
                lambda_star = compute_lambda_star(
                    self.theta_t_minus_1_star,
                    theta_t_cand,
                    current_fim,
                    accumulated_fim,
                )
                logging.info(
                    f"Computed adaptive lambda_star = {lambda_star.item():.6f}"
                )

        self.lambda_history.append(lambda_star.item())

        # Log to analyzer if available
        if hasattr(self, "analyzer"):
            self.analyzer.log_lambda_value(self._cur_task, lambda_star.item())

        # Perform adaptive parameter merging
        if lambda_star.item() < 1.0:  # Only merge if lambda < 1
            final_theta = adaptive_merge_parameters(
                self.theta_t_minus_1_star, theta_t_cand, lambda_star
            )

            # Load merged parameters back into model
            current_state = self._network.state_dict()
            for name, param in final_theta.items():
                if name in current_state:
                    current_state[name] = param

            self._network.load_state_dict(current_state, strict=False)
            logging.info("Applied adaptive parameter merging")
        else:
            logging.info("lambda_star >= 1.0, keeping candidate model")

        # Update Fisher manager with final model's FIM
        final_fim = compute_diagonal_fim(self._network, fim_loader, self._device)
        self.fisher_manager.update_fisher(final_fim)

        # Store current final parameters for next task
        self.theta_t_minus_1_star = {
            name: p.clone().detach()
            for name, p in self._network.named_parameters()
            if p.requires_grad
        }

    def train_function(self, train_loader, test_loader):
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
                loss = F.cross_entropy(logits, targets)
                ATL = criterion(feature, labels, self._protos)
                loss += self.lambada * ATL

                self.model_optimizer.zero_grad()
                loss.backward()

                self.model_optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
                # if self.debug and i > 10: break

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

    def _build_protos(self):
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

    def _evaluate(self, y_pred, y_true):
        ret = {}
        print(len(y_pred), len(y_true))
        grouped = accuracy(y_pred, y_true, self._known_classes, self.class_num)
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        return ret

    def eval_task(self):
        y_pred, y_true = self._eval_model(
            self.test_loader,
            self._protos / np.linalg.norm(self._protos, axis=1)[:, None],
        )
        nme_accy = self._evaluate(y_pred.T[0], y_true)
        return nme_accy

    def _eval_model(self, loader, class_means):
        self._network.eval()
        vectors, y_true = self._extract_vectors(loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + self.EPSILON)).T

        dists = cdist(class_means, vectors, "sqeuclidean")
        scores = dists.T

        return np.argsort(scores, axis=1)[:, : self.topk], y_true  # [N, topk]

    def _compute_accuracy_domain(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]

            predicts = torch.max(outputs, dim=1)[1]
            correct += (
                (predicts % self.class_num).cpu() == (targets % self.class_num)
            ).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def init_model_optimizer(self):
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
        # self.args['model_optimizer'] = 'Adam'
        self.model_optimizer = getattr(optimgrad, self.args["optim"])(
            **model_optimizer_arg
        )
        self.model_scheduler = CosineSchedule(self.model_optimizer, K=self.epochs)

    def update_optim_transforms(self):
        self.model_optimizer.get_eigens(self.fea_in)
        self.model_optimizer.get_transforms()
        self.fea_in = defaultdict(dict)
