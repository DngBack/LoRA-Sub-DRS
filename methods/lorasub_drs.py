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
import ipdb
import optimgrad
import re
from collections import defaultdict
from utils.losses import AugmentedTripletLoss
from scipy.spatial.distance import cdist

# Hyperspherical imports
from utils.hyperspherical import (
    normalize_to_sphere,
    angular_distance,
    HypersphericalProjector,
    save_spherical_prototypes,
    load_spherical_prototypes,
    kl_spcauchy,
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

        # Hyperspherical DRS parameters
        self.use_hyperspherical = args.get("use_hyperspherical", False)
        self.spcauchy_rho = args.get("spcauchy_rho", 0.5)
        self.sphere_dim = args.get("sphere_dim", 768)
        self.kl_beta = args.get("kl_beta", 0.1)
        self.angular_margin = args.get("angular_margin", 0.1)
        self.variance_threshold = args.get("variance_threshold", 0.95)
        self.enable_spherical_projection = args.get(
            "enable_spherical_projection", False
        )
        self.save_prototypes = args.get("save_prototypes", True)
        self.prototype_dir = args.get("prototype_dir", "./prototypes")

        # Initialize hyperspherical projector if enabled
        if self.use_hyperspherical:
            self.h_projector = HypersphericalProjector(
                sphere_dim=self.sphere_dim,
                spcauchy_rho=self.spcauchy_rho,
                variance_threshold=self.variance_threshold,
            )
            self._spherical_protos = {}  # Store spherical prototypes
            logging.info(
                "Hyperspherical DRS enabled with spCauchy rho={:.3f}".format(
                    self.spcauchy_rho
                )
            )
        else:
            self.h_projector = None

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
        self._network.to(self._device)

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

        self.train_function(train_loader, test_loader)

        return

    def compute_angular_atl_loss(self, features, labels, old_prototypes):
        """
        Compute Augmented Triplet Loss using angular distance on hypersphere
        """
        if not self.use_hyperspherical:
            # Fall back to original ATL
            criterion = AugmentedTripletLoss(margin=self.margin_inter).to(self._device)
            return criterion(features, labels, self._protos)

        # Normalize features to sphere
        features_norm = normalize_to_sphere(features)

        atl_loss = 0.0
        valid_triplets = 0

        for i in range(len(features_norm)):
            anchor = features_norm[i]
            pos_label = labels[i]

            # Find hardest positive (same class in current batch)
            pos_mask = labels == pos_label
            if pos_mask.sum() > 1:
                pos_indices = pos_mask.nonzero(as_tuple=False).squeeze()
                if pos_indices.dim() == 0:
                    pos_indices = pos_indices.unsqueeze(0)
                pos_distances = angular_distance(
                    anchor.unsqueeze(0), features_norm[pos_indices]
                )
                e_ap = pos_distances.max()
            else:
                e_ap = torch.tensor(0.0).to(self._device)

            # Find hardest negative from current batch
            neg_mask = labels != pos_label
            if neg_mask.sum() > 0:
                neg_indices = neg_mask.nonzero(as_tuple=False).squeeze()
                if neg_indices.dim() == 0:
                    neg_indices = neg_indices.unsqueeze(0)
                neg_distances = angular_distance(
                    anchor.unsqueeze(0), features_norm[neg_indices]
                )
                e_an_current = neg_distances.min()
            else:
                e_an_current = torch.tensor(float("inf")).to(self._device)

            # Find hardest negative from old prototypes (spherical)
            e_an_old = torch.tensor(float("inf")).to(self._device)
            if old_prototypes:
                for proto_array in old_prototypes:
                    if len(proto_array) > 0:
                        proto_tensor = torch.from_numpy(proto_array).to(self._device)
                        if proto_tensor.dim() == 1:
                            proto_tensor = proto_tensor.unsqueeze(0)
                        proto_distances = angular_distance(
                            anchor.unsqueeze(0), proto_tensor
                        )
                        e_an_old = min(e_an_old, proto_distances.min())

            # Take minimum of current and old negatives
            e_an = min(e_an_current, e_an_old)

            # Compute triplet loss with angular margin
            if e_an < float("inf"):
                triplet_loss = torch.relu(e_ap - e_an + self.angular_margin)
                atl_loss += triplet_loss
                valid_triplets += 1

        if valid_triplets > 0:
            atl_loss = atl_loss / valid_triplets

        return atl_loss

    def train_function(self, train_loader, test_loader):
        prog_bar = tqdm(range(self.run_epoch))

        # Load old spherical prototypes for ATL computation
        old_prototypes = []
        if self.use_hyperspherical:
            for task_id in range(self._cur_task):
                task_protos = load_spherical_prototypes(task_id, self.prototype_dir)
                if task_protos:
                    for class_proto in task_protos.values():
                        old_prototypes.append(class_proto)

        for _, epoch in enumerate(prog_bar):
            self._network.eval()
            losses = 0.0
            correct, total = 0, 0
            kl_losses = 0.0

            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                mask = (targets >= self._known_classes).nonzero().view(-1)
                inputs = torch.index_select(inputs, 0, mask)
                labels = torch.index_select(targets, 0, mask)
                targets = torch.index_select(targets, 0, mask) - self._known_classes

                ret = self._network(inputs)
                logits = ret["logits"]
                features = ret["features"]

                if self.use_hyperspherical:
                    # Normalize features to sphere
                    feature = normalize_to_sphere(features)
                    # Compute angular ATL
                    ATL = self.compute_angular_atl_loss(feature, labels, old_prototypes)

                    # Optional: Add KL regularization for spCauchy
                    kl_loss = 0
                    if self.kl_beta > 0:
                        kl_loss = kl_spcauchy(self.spcauchy_rho, d=self.sphere_dim)
                        kl_losses += kl_loss.item()
                else:
                    # Original implementation
                    feature = features / features.norm(dim=-1, keepdim=True)
                    criterion = AugmentedTripletLoss(margin=self.margin_inter).to(
                        self._device
                    )
                    ATL = criterion(feature, labels, self._protos)
                    kl_loss = 0

                loss = F.cross_entropy(logits, targets)
                loss += self.lambada * ATL

                if self.use_hyperspherical and self.kl_beta > 0:
                    loss += self.kl_beta * kl_loss

                self.model_optimizer.zero_grad()
                loss.backward()

                # Apply hyperspherical gradient projection if enabled
                if (
                    self.use_hyperspherical
                    and self.enable_spherical_projection
                    and self.h_projector.projection_matrix is not None
                ):
                    for param in self._network.parameters():
                        if param.grad is not None:
                            param.grad.data = self.h_projector.project_gradients(
                                param.grad.data
                            )

                self.model_optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
                # if self.debug and i > 10: break

            self.model_scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if self.use_hyperspherical and self.kl_beta > 0:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, KL {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.run_epoch,
                    losses / len(train_loader),
                    kl_losses / len(train_loader),
                    train_acc,
                )
            else:
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
        current_task_protos = {}

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

                if self.use_hyperspherical:
                    # Convert to tensor and normalize to sphere
                    class_mean_tensor = torch.from_numpy(class_mean).to(self._device)
                    spherical_proto = normalize_to_sphere(
                        class_mean_tensor.unsqueeze(0)
                    ).squeeze(0)

                    # Store spherical prototype
                    current_task_protos[class_idx] = spherical_proto.cpu().numpy()
                    self._protos.append(spherical_proto.cpu().numpy())

                    # Update hyperspherical projector if needed
                    if self.enable_spherical_projection and self._cur_task > 0:
                        # Collect features for projection computation
                        all_features = []
                        for batch_idx, (_, inputs, _) in enumerate(idx_loader):
                            inputs = inputs.to(self._device)
                            features = self._network.extract_vector(inputs)
                            all_features.append(features)

                        if all_features:
                            all_features = torch.cat(all_features, dim=0)
                            # Update projector with current task features
                            self.h_projector.compute_projection(
                                all_features, use_spcauchy=True
                            )
                else:
                    self._protos.append(class_mean)

        # Save spherical prototypes for this task
        if self.use_hyperspherical and self.save_prototypes and current_task_protos:
            save_spherical_prototypes(
                current_task_protos, self._cur_task, self.prototype_dir
            )
            self._spherical_protos[self._cur_task] = current_task_protos
            logging.info(f"Saved spherical prototypes for task {self._cur_task}")

    def _evaluate(self, y_pred, y_true):
        ret = {}
        print(len(y_pred), len(y_true))
        grouped = accuracy(y_pred, y_true, self._known_classes, self.class_num)
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        return ret

    def eval_task(self):
        if self.use_hyperspherical:
            # Use spherical evaluation with angular distance
            protos_array = np.array(self._protos)
            # Prototypes are already normalized when using hyperspherical mode
            y_pred, y_true = self._eval_model(self.test_loader, protos_array)
        else:
            # Original evaluation
            y_pred, y_true = self._eval_model(
                self.test_loader,
                self._protos / np.linalg.norm(self._protos, axis=1)[:, None],
            )

        nme_accy = self._evaluate(y_pred.T[0], y_true)

        # Add drift measurement for hyperspherical mode
        if self.use_hyperspherical and self._cur_task > 0:
            drift_score = self._compute_spherical_drift()
            logging.info(f"Spherical drift score: {drift_score:.4f}")

        return nme_accy

    def _compute_spherical_drift(self):
        """
        Compute spherical drift by measuring angular distance between
        current and stored prototypes
        """
        if not self.use_hyperspherical or self._cur_task == 0:
            return 0.0

        total_drift = 0.0
        num_comparisons = 0

        # Compare current prototypes with stored ones
        for task_id in range(self._cur_task):
            stored_protos = load_spherical_prototypes(task_id, self.prototype_dir)
            if stored_protos:
                for class_id, stored_proto in stored_protos.items():
                    # Extract current representation for this class
                    current_proto = self._get_current_prototype(class_id)
                    if current_proto is not None:
                        stored_tensor = torch.from_numpy(stored_proto).to(self._device)
                        current_tensor = torch.from_numpy(current_proto).to(
                            self._device
                        )

                        # Compute angular distance
                        drift = angular_distance(
                            stored_tensor.unsqueeze(0), current_tensor.unsqueeze(0)
                        )
                        total_drift += drift.item()
                        num_comparisons += 1

        return total_drift / num_comparisons if num_comparisons > 0 else 0.0

    def _get_current_prototype(self, class_id):
        """
        Get current prototype for a specific class by re-extracting features
        """
        try:
            data, targets, idx_dataset = self.data_manager.get_dataset(
                np.arange(class_id, class_id + 1),
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

            if len(vectors) > 0:
                class_mean = np.mean(vectors, axis=0)
                if self.use_hyperspherical:
                    class_mean_tensor = torch.from_numpy(class_mean).to(self._device)
                    spherical_proto = normalize_to_sphere(
                        class_mean_tensor.unsqueeze(0)
                    ).squeeze(0)
                    return spherical_proto.cpu().numpy()
                return class_mean
            return None
        except:
            return None

    def _eval_model(self, loader, class_means):
        self._network.eval()
        vectors, y_true = self._extract_vectors(loader)

        if self.use_hyperspherical:
            # Normalize vectors to sphere for angular distance computation
            vectors_tensor = torch.from_numpy(vectors).to(self._device)
            vectors_norm = normalize_to_sphere(vectors_tensor)
            vectors = vectors_norm.cpu().numpy()

            # Use angular distance instead of Euclidean
            # Convert to torch tensors for angular distance computation
            class_means_tensor = torch.from_numpy(class_means).to(self._device)
            vectors_tensor = torch.from_numpy(vectors).to(self._device)

            # Compute angular distances
            scores = []
            for i in range(len(vectors)):
                dists = angular_distance(vectors_tensor[i : i + 1], class_means_tensor)
                scores.append(dists.cpu().numpy())
            scores = np.array(scores)
        else:
            # Original Euclidean distance computation
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
