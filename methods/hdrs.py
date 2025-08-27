"""
Hyperspherical Drift-Resistant Space (HDRS) method
Full Riemannian LoRA implementation for continual learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from methods.base import BaseLearner
from utils.toolkit import tensor2numpy, accuracy
from models.riemannian_lora import RiemannianAttention, compute_tangent_pca, ManualSphere
from utils.losses import AugmentedTripletLoss
from utils.schedulers import CosineSchedule
from collections import defaultdict

# Try to import geoopt for advanced Riemannian optimization
try:
    import geoopt
    GEOOPT_AVAILABLE = True
except ImportError:
    GEOOPT_AVAILABLE = False


class HypersphericalDRS(BaseLearner):
    """
    Hyperspherical Drift-Resistant Space (HDRS) using Riemannian LoRA
    
    This method implements:
    1. LoRA parameters on Riemannian manifolds (Sphere/Stiefel)
    2. Manifold-aware LoRA subtraction
    3. Tangent space PCA for drift resistance
    4. Cosine/geodesic distance metrics
    """
    
    def __init__(self, args):
        super().__init__(args)
        
        # Create network with Riemannian attention
        if args["net_type"] == "sip":
            self._network = self._create_riemannian_network(args)
        else:
            raise ValueError('Unknown net: {}.'.format(args["net_type"]))
        
        # Hyperparameters
        self.args = args
        self.EPSILON = args["EPSILON"]
        self.init_epoch = args["init_epoch"]
        self.init_lr = args["init_lr"]
        self.epochs = args["epochs"]
        self.lrate = args["lrate"]
        self.batch_size = args["batch_size"]
        self.weight_decay = args["weight_decay"]
        self.num_workers = args["num_workers"]
        self.lambada = args["lambada"]
        self.fc_lrate = args["fc_lrate"]
        self.margin_inter = args["margin_inter"]
        self.eval = args['eval']
        
        # Riemannian-specific parameters
        self.manifold_A = args.get("manifold_A", "stiefel")  # "stiefel" or "sphere"
        self.manifold_B = args.get("manifold_B", "sphere")   # "sphere" or "stiefel"
        self.use_geoopt = args.get("use_geoopt", GEOOPT_AVAILABLE)
        self.eta_subtraction = args.get("eta_subtraction", 0.1)  # LoRA subtraction scale
        self.use_tangent_pca = args.get("use_tangent_pca", True)
        self.drs_energy_threshold = args.get("drs_energy_threshold", 0.99)
        self.drs_max_components = args.get("drs_max_components", 64)
        
        # Prototype storage for classification
        self._protos = []
        self.topk = 1
        self.class_num = self._network.class_num if hasattr(self._network, 'class_num') else 100
        
        # Feature tracking for DRS
        self.feature_stats = defaultdict(dict)
        
        logging.info(f"HDRS initialized with manifolds: A={self.manifold_A}, B={self.manifold_B}")
        logging.info(f"Geoopt available: {self.use_geoopt}, Use tangent PCA: {self.use_tangent_pca}")
    
    def _create_riemannian_network(self, args):
        """Create network with Riemannian attention layers"""
        # For now, we'll create a simple adapter. In practice, you'd integrate with SiNet
        # This is a placeholder - you would replace attention layers in your actual network
        from models.sinet_lora import SiNet
        
        # Create base network
        network = SiNet(args)
        
        # Replace attention layers with Riemannian versions
        # This is conceptual - actual implementation depends on your network structure
        self._replace_attention_layers(network, args)
        
        return network
    
    def _replace_attention_layers(self, network, args):
        """Replace standard attention with Riemannian attention"""
        # This is a conceptual implementation
        # In practice, you'd need to identify and replace the specific attention modules
        # in your Vision Transformer or SiNet architecture
        
        # Example of replacing modules (adapt to your specific architecture)
        for name, module in network.named_modules():
            if hasattr(module, 'attn') and hasattr(module.attn, 'qkv'):
                # Replace attention with Riemannian version
                old_attn = module.attn
                new_attn = RiemannianAttention(
                    dim=old_attn.qkv.in_features // 3,
                    num_heads=getattr(old_attn, 'num_heads', 8),
                    rank=args["rank"],
                    n_tasks=args["total_sessions"],
                    use_geoopt=self.use_geoopt
                )
                setattr(module, 'attn', new_attn)
                logging.info(f"Replaced attention layer: {name}")
    
    def after_task(self):
        """Called after each task completion"""
        self._known_classes = self._total_classes
        logging.info(f'Task {self._cur_task} completed. Known classes: {self._known_classes}')
    
    def incremental_train(self, data_manager):
        """Main training loop for incremental learning"""
        self.data_manager = data_manager
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        
        # Update classifier
        self._network.update_fc(self._total_classes)
        
        logging.info(f'Learning on task {self._cur_task}: {self._known_classes}-{self._total_classes}')
        
        # Prepare data loaders
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), 
            source='train', mode='train'
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, 
            shuffle=True, num_workers=self.num_workers
        )
        
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), 
            source='test', mode='test'
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, 
            shuffle=False, num_workers=self.num_workers
        )
        
        # Multi-GPU setup
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        
        # Training
        if not self.eval:
            self._train_task()
        
        # Clean up multi-GPU
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        
        # Build prototypes for current task
        self._build_prototypes()
    
    def _train_task(self):
        """Train the current task with Riemannian optimization"""
        self._network.to(self._device)
        
        # Collect features for DRS if not first task
        if self._cur_task > 0:
            self._collect_features_for_drs()
        
        # Setup optimizers
        self._setup_optimizers()
        
        # Set training epochs
        epochs = self.init_epoch if self._cur_task == 0 else self.epochs
        
        # Training loop
        self._train_epochs(epochs)
    
    def _collect_features_for_drs(self):
        """Collect features for Drift-Resistant Space computation"""
        self._network.eval()
        all_features = []
        
        with torch.no_grad():
            # Use previous task data for DRS computation
            prev_classes = np.arange(0, self._known_classes)
            if len(prev_classes) > 0:
                prev_dataset = self.data_manager.get_dataset(
                    prev_classes, source='train', mode='test'
                )
                prev_loader = DataLoader(
                    prev_dataset, batch_size=self.batch_size, 
                    shuffle=False, num_workers=4
                )
                
                for _, inputs, targets in prev_loader:
                    inputs = inputs.to(self._device)
                    # Forward pass to collect features
                    ret = self._network(inputs, task_id=self._cur_task-1, collect_features=True)
                    if 'features' in ret:
                        features = ret['features']
                        all_features.append(features.cpu())
        
        if all_features:
            all_features = torch.cat(all_features, dim=0)
            self._compute_drs_projection(all_features)
    
    def _compute_drs_projection(self, features):
        """Compute DRS projection using tangent space PCA or standard PCA"""
        # Normalize features to unit sphere
        features_norm = ManualSphere.normalize(features)
        
        if self.use_tangent_pca:
            # Compute tangent space PCA
            try:
                mu, P_t = compute_tangent_pca(
                    features_norm, 
                    k=self.drs_max_components,
                    energy_threshold=self.drs_energy_threshold
                )
                logging.info(f"Tangent PCA computed: {P_t.shape[1]} components")
            except Exception as e:
                logging.warning(f"Tangent PCA failed: {e}. Using standard PCA.")
                self.use_tangent_pca = False
        
        if not self.use_tangent_pca:
            # Fallback to standard PCA on normalized features
            cov_matrix = torch.cov(features_norm.T)
            eigenvals, eigenvecs = torch.linalg.eigh(cov_matrix)
            
            # Sort in descending order
            idx = torch.argsort(eigenvals, descending=True)
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]
            
            # Select components
            cumulative_energy = torch.cumsum(eigenvals, dim=0) / torch.sum(eigenvals)
            k_selected = min(
                self.drs_max_components, 
                (cumulative_energy >= self.drs_energy_threshold).nonzero()[0].item() + 1
            )
            P_t = eigenvecs[:, :k_selected]
            logging.info(f"Standard PCA computed: {k_selected} components")
        
        # Store projection for use in gradient projection
        self.current_P_t = P_t.to(self._device)
    
    def _setup_optimizers(self):
        """Setup optimizers for Riemannian and standard parameters"""
        # Collect Riemannian parameters
        riemannian_params = []
        standard_params = []
        classifier_params = []
        
        for name, param in self._network.named_parameters():
            if not param.requires_grad:
                continue
            
            if hasattr(param, 'manifold'):  # Riemannian parameter
                riemannian_params.append(param)
            elif 'classifier' in name or 'fc' in name:
                classifier_params.append(param)
            else:
                standard_params.append(param)
        
        # Setup optimizers
        lr = self.init_lr if self._cur_task == 0 else self.lrate
        
        self.optimizers = []
        
        # Riemannian optimizer
        if riemannian_params and self.use_geoopt:
            riem_opt = geoopt.optim.RiemannianAdam(riemannian_params, lr=lr)
            self.optimizers.append(('riemannian', riem_opt))
        
        # Standard parameter optimizer
        if standard_params:
            std_opt = torch.optim.Adam(standard_params, lr=lr, weight_decay=self.weight_decay)
            self.optimizers.append(('standard', std_opt))
        
        # Classifier optimizer
        if classifier_params:
            cls_opt = torch.optim.Adam(classifier_params, lr=self.fc_lrate, weight_decay=self.weight_decay)
            self.optimizers.append(('classifier', cls_opt))
        
        # Scheduler
        combined_params = riemannian_params + standard_params + classifier_params
        if combined_params:
            self.scheduler = CosineSchedule(
                torch.optim.Adam(combined_params, lr=lr), 
                K=self.epochs
            )
        
        logging.info(f"Optimizers setup: {len(self.optimizers)} optimizers")
    
    def _train_epochs(self, epochs):
        """Training loop for specified epochs"""
        criterion = AugmentedTripletLoss(margin=self.margin_inter).to(self._device)
        
        prog_bar = tqdm(range(epochs))
        for epoch in prog_bar:
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            
            for i, (_, inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                
                # Filter current task samples
                mask = (targets >= self._known_classes).nonzero().view(-1)
                if len(mask) == 0:
                    continue
                
                inputs = torch.index_select(inputs, 0, mask)
                labels = torch.index_select(targets, 0, mask)
                targets_rel = labels - self._known_classes  # Relative to current task
                
                # Forward pass
                ret = self._network(inputs, task_id=self._cur_task)
                logits = ret['logits']
                features = ret['features']
                
                # Normalize features for cosine similarity
                features_norm = ManualSphere.normalize(features)
                
                # Compute losses
                loss_ce = F.cross_entropy(logits, targets_rel)
                
                # Augmented Triplet Loss with cosine distance
                if len(self._protos) > 0:
                    loss_atl = criterion(features_norm, labels, self._protos)
                    loss = loss_ce + self.lambada * loss_atl
                else:
                    loss = loss_ce
                
                # Backward pass
                for _, optimizer in self.optimizers:
                    optimizer.zero_grad()
                
                loss.backward()
                
                # Apply DRS projection if available
                if hasattr(self, 'current_P_t') and self._cur_task > 0:
                    self._apply_drs_projection()
                
                # Optimizer step
                for _, optimizer in self.optimizers:
                    optimizer.step()
                
                # Statistics
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets_rel).cpu().sum()
                total += len(targets_rel)
            
            # Scheduler step
            if hasattr(self, 'scheduler'):
                self.scheduler.step()
            
            # Logging
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            info = f'Task {self._cur_task}, Epoch {epoch + 1}/{epochs} => Loss {losses / len(self.train_loader):.3f}, Train_acc {train_acc:.2f}'
            prog_bar.set_description(info)
        
        logging.info(info)
    
    def _apply_drs_projection(self):
        """Apply DRS projection to gradients"""
        # Project gradients of Riemannian LoRA parameters
        for name, module in self._network.named_modules():
            if hasattr(module, 'apply_drs_projection'):
                module.apply_drs_projection(self.current_P_t, self._cur_task)
    
    def _build_prototypes(self):
        """Build class prototypes for current task"""
        self._network.eval()
        
        with torch.no_grad():
            for class_idx in range(self._known_classes, self._total_classes):
                # Get data for current class
                data, targets, idx_dataset = self.data_manager.get_dataset(
                    np.arange(class_idx, class_idx + 1),
                    source='train', mode='test', ret_data=True
                )
                idx_loader = DataLoader(
                    idx_dataset, batch_size=self.batch_size, 
                    shuffle=False, num_workers=4
                )
                
                # Extract features
                vectors, _ = self._extract_vectors(idx_loader)
                
                # Compute prototype (mean of normalized features)
                vectors_norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + self.EPSILON)
                class_prototype = np.mean(vectors_norm, axis=0)
                class_prototype = class_prototype / (np.linalg.norm(class_prototype) + self.EPSILON)
                
                self._protos.append(class_prototype)
        
        logging.info(f"Built {len(self._protos)} prototypes")
    
    def _extract_vectors(self, loader):
        """Extract feature vectors from data loader"""
        self._network.eval()
        vectors, targets = [], []
        
        with torch.no_grad():
            for _, inputs, labels in loader:
                inputs = inputs.to(self._device)
                ret = self._network(inputs, task_id=self._cur_task)
                features = ret['features']
                
                vectors.append(tensor2numpy(features))
                targets.append(tensor2numpy(labels))
        
        return np.concatenate(vectors), np.concatenate(targets)
    
    def eval_task(self):
        """Evaluate current task performance"""
        # Normalize prototypes
        if len(self._protos) > 0:
            proto_array = np.array(self._protos)
            proto_norm = proto_array / (np.linalg.norm(proto_array, axis=1, keepdims=True) + self.EPSILON)
        else:
            proto_norm = np.array([])
        
        y_pred, y_true = self._eval_model(self.test_loader, proto_norm)
        nme_acc = self._evaluate(y_pred.T[0], y_true)
        return nme_acc
    
    def _eval_model(self, loader, class_means):
        """Evaluate model using nearest prototype classification"""
        self._network.eval()
        vectors, y_true = self._extract_vectors(loader)
        
        # Normalize vectors
        vectors_norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + self.EPSILON)
        
        # Compute cosine distances (converted to Euclidean in normalized space)
        if len(class_means) > 0:
            # Use cosine similarity (1 - cosine distance)
            similarities = np.dot(vectors_norm, class_means.T)
            scores = -similarities  # Convert to distances (lower is better)
        else:
            scores = np.zeros((len(vectors_norm), 1))
        
        return np.argsort(scores, axis=1)[:, :self.topk], y_true
    
    def _evaluate(self, y_pred, y_true):
        """Evaluate predictions"""
        ret = {}
        grouped = accuracy(y_pred, y_true, self._known_classes, self.class_num)
        ret['grouped'] = grouped
        ret['top1'] = grouped['total']
        return ret
