#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analysis and Ablation Framework for DRS-Hyperspherical

Provides comprehensive analysis tools including:
- Ablation study generation
- Performance metrics specific to spherical geometry
- Visualization of geodesic drift and prototype evolution
- Statistical analysis of DRS components
"""

import os
import json
import shutil
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime


class HypersphericalAnalyzer:
    """Comprehensive analyzer for DRS-Hyperspherical experiments"""

    def __init__(self, experiment_name: str, save_dir: str = "./experiments/"):
        self.experiment_name = experiment_name
        self.save_dir = save_dir
        self.experiment_dir = os.path.join(
            save_dir, f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # Create experiment directory
        os.makedirs(self.experiment_dir, exist_ok=True)

        # Initialize tracking
        self.metrics = {
            "task_accuracies": [],
            "average_accuracies": [],
            "forgetting_measures": [],
            "geodesic_drifts": [],
            "pca_components": [],
            "pca_variances": [],
            "lambda_values": [],
            "prototype_shifts": [],
        }

        logging.info(f"Initialized analyzer: {self.experiment_dir}")

    def log_task_results(
        self,
        task_id: int,
        task_acc: float,
        avg_acc: float,
        geodesic_drift: float = 0.0,
        pca_components: int = 0,
        pca_variance: float = 0.0,
    ):
        """Log results for a completed task"""
        self.metrics["task_accuracies"].append(task_acc)
        self.metrics["average_accuracies"].append(avg_acc)
        self.metrics["geodesic_drifts"].append(geodesic_drift)
        self.metrics["pca_components"].append(pca_components)
        self.metrics["pca_variances"].append(pca_variance)

        logging.info(
            f"Task {task_id}: Acc={task_acc:.2f}%, AvgAcc={avg_acc:.2f}%, "
            f"Drift={geodesic_drift:.4f}, PCA={pca_components}"
        )

    def log_lambda_value(self, task_id: int, lambda_val: float):
        """Log lambda value for adaptive merging"""
        if len(self.metrics["lambda_values"]) <= task_id:
            self.metrics["lambda_values"].extend(
                [0.0] * (task_id + 1 - len(self.metrics["lambda_values"]))
            )
        self.metrics["lambda_values"][task_id] = lambda_val

    def compute_forgetting_measure(
        self, all_task_accuracies: List[List[float]]
    ) -> float:
        """
        Compute Backward Transfer (BWT) forgetting measure
        BWT = (1/(T-1)) * Σ_{t=1}^{T-1} (R_{T,t} - R_{t,t})
        """
        if len(all_task_accuracies) < 2:
            return 0.0

        T = len(all_task_accuracies)
        bwt_sum = 0.0

        for t in range(T - 1):  # t from 0 to T-2
            if t < len(all_task_accuracies[T - 1]) and t < len(all_task_accuracies[t]):
                final_acc = all_task_accuracies[T - 1][t]  # R_{T,t}
                original_acc = all_task_accuracies[t][t]  # R_{t,t}
                bwt_sum += final_acc - original_acc

        return bwt_sum / (T - 1) if T > 1 else 0.0

    def generate_visualizations(self):
        """Generate comprehensive visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            f"DRS-Hyperspherical Analysis: {self.experiment_name}", fontsize=16
        )

        # 1. Accuracy Evolution
        axes[0, 0].plot(
            self.metrics["average_accuracies"], "b-o", label="Average Accuracy"
        )
        axes[0, 0].set_title("Accuracy Evolution")
        axes[0, 0].set_xlabel("Task")
        axes[0, 0].set_ylabel("Accuracy (%)")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # 2. Geodesic Drift
        if len(self.metrics["geodesic_drifts"]) > 1:
            axes[0, 1].plot(
                range(1, len(self.metrics["geodesic_drifts"]) + 1),
                self.metrics["geodesic_drifts"],
                "r-s",
                label="Geodesic Drift",
            )
            axes[0, 1].set_title("Geodesic Drift Evolution")
            axes[0, 1].set_xlabel("Task")
            axes[0, 1].set_ylabel("Average Geodesic Distance")
            axes[0, 1].legend()
            axes[0, 1].grid(True)

        # 3. PCA Components
        if self.metrics["pca_components"]:
            axes[0, 2].bar(
                range(len(self.metrics["pca_components"])),
                self.metrics["pca_components"],
                color="green",
                alpha=0.7,
            )
            axes[0, 2].set_title("PCA Components per Task")
            axes[0, 2].set_xlabel("Task")
            axes[0, 2].set_ylabel("Number of Components")
            axes[0, 2].grid(True)

        # 4. PCA Explained Variance
        if self.metrics["pca_variances"]:
            axes[1, 0].plot(
                self.metrics["pca_variances"], "g-^", label="Explained Variance"
            )
            axes[1, 0].set_title("PCA Explained Variance")
            axes[1, 0].set_xlabel("Task")
            axes[1, 0].set_ylabel("Explained Variance Ratio")
            axes[1, 0].legend()
            axes[1, 0].grid(True)

        # 5. Lambda Evolution (if available)
        if self.metrics["lambda_values"] and any(
            v > 0 for v in self.metrics["lambda_values"]
        ):
            non_zero_lambdas = [v for v in self.metrics["lambda_values"] if v > 0]
            lambda_tasks = [
                i for i, v in enumerate(self.metrics["lambda_values"]) if v > 0
            ]
            axes[1, 1].plot(
                lambda_tasks, non_zero_lambdas, "m-d", label="Lambda Values"
            )
            axes[1, 1].set_title("Adaptive Lambda Evolution")
            axes[1, 1].set_xlabel("Task")
            axes[1, 1].set_ylabel("Lambda Value")
            axes[1, 1].legend()
            axes[1, 1].grid(True)

        # 6. Stability Gap Analysis
        if len(self.metrics["task_accuracies"]) > 1:
            stability_gaps = []
            for i in range(1, len(self.metrics["task_accuracies"])):
                gap = (
                    self.metrics["task_accuracies"][0]
                    - self.metrics["task_accuracies"][i]
                )
                stability_gaps.append(gap)

            axes[1, 2].plot(
                range(1, len(stability_gaps) + 1),
                stability_gaps,
                "c-o",
                label="Stability Gap",
            )
            axes[1, 2].set_title("Stability Gap (vs Initial Task)")
            axes[1, 2].set_xlabel("Task")
            axes[1, 2].set_ylabel("Accuracy Drop (%)")
            axes[1, 2].legend()
            axes[1, 2].grid(True)

        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(self.experiment_dir, "analysis_plots.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        logging.info(f"Visualizations saved to {plot_path}")

    def save_results(self, additional_metrics: Optional[Dict] = None):
        """Save all results and metrics"""
        results = {
            "experiment_name": self.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "metrics": self.metrics,
            "summary": {
                "final_average_accuracy": self.metrics["average_accuracies"][-1]
                if self.metrics["average_accuracies"]
                else 0.0,
                "average_geodesic_drift": np.mean(self.metrics["geodesic_drifts"])
                if self.metrics["geodesic_drifts"]
                else 0.0,
                "average_pca_components": np.mean(self.metrics["pca_components"])
                if self.metrics["pca_components"]
                else 0.0,
                "total_tasks": len(self.metrics["task_accuracies"]),
            },
        }

        if additional_metrics:
            results["additional_metrics"] = additional_metrics

        # Save as JSON
        results_path = os.path.join(self.experiment_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        logging.info(f"Results saved to {results_path}")

        return results


class AblationStudyGenerator:
    """Generate ablation study configurations for DRS-Hyperspherical"""

    def __init__(self, base_config_path: str, output_dir: str = "./ablation_configs/"):
        self.base_config_path = base_config_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Load base configuration
        with open(base_config_path, "r", encoding="utf-8") as f:
            self.base_config = json.load(f)

    def generate_ablation_configs(self) -> List[str]:
        """Generate all ablation study configurations"""
        generated_configs = []

        # 1. Baseline LoRA-Sub (Linear DRS)
        baseline_config = self._create_baseline_config()
        baseline_path = os.path.join(self.output_dir, "baseline_lorasub.json")
        self._save_config(baseline_config, baseline_path)
        generated_configs.append(baseline_path)

        # 2. Riemannian-only (without DRS subspace projection)
        riemann_only_config = self._create_riemannian_only_config()
        riemann_path = os.path.join(self.output_dir, "riemann_only.json")
        self._save_config(riemann_only_config, riemann_path)
        generated_configs.append(riemann_path)

        # 3. Without DRS subspace (no PCA projection)
        no_drs_config = self._create_no_drs_config()
        no_drs_path = os.path.join(self.output_dir, "no_drs_projection.json")
        self._save_config(no_drs_config, no_drs_path)
        generated_configs.append(no_drs_path)

        # 4. Without Angular Triplet
        no_triplet_config = self._create_no_triplet_config()
        no_triplet_path = os.path.join(self.output_dir, "no_angular_triplet.json")
        self._save_config(no_triplet_config, no_triplet_path)
        generated_configs.append(no_triplet_path)

        # 5. Without EMA (fixed prototypes)
        no_ema_config = self._create_no_ema_config()
        no_ema_path = os.path.join(self.output_dir, "no_ema_prototypes.json")
        self._save_config(no_ema_config, no_ema_path)
        generated_configs.append(no_ema_path)

        # 6. Without warmup & annealing
        no_warmup_config = self._create_no_warmup_config()
        no_warmup_path = os.path.join(self.output_dir, "no_warmup_anneal.json")
        self._save_config(no_warmup_config, no_warmup_path)
        generated_configs.append(no_warmup_path)

        # 7. Full DRS-Hyperspherical (reference)
        full_config = self.base_config.copy()
        full_config["experiment"]["prefix"] = "DRS-Hyperspherical-Full"
        full_path = os.path.join(self.output_dir, "drs_hyperspherical_full.json")
        self._save_config(full_config, full_path)
        generated_configs.append(full_path)

        logging.info(f"Generated {len(generated_configs)} ablation configurations:")
        for config in generated_configs:
            logging.info(f"  - {config}")

        return generated_configs

    def _create_baseline_config(self) -> Dict:
        """Create baseline LoRA-Sub configuration"""
        config = self.base_config.copy()
        config["model"]["model_name"] = "lorasub_drs"
        config["experiment"]["prefix"] = "Baseline-LoRASub"

        # Disable all hyperspherical features
        config["spherical"] = {
            "per_class_prototypes": False,
            "multi_anchor": False,
            "pca_energy": 0.0,
            "k_max": 0,
        }

        # Use standard training
        config["train"]["epochs_warm"] = 0
        config["train"]["epochs_main"] = config["train"]["epochs"]

        return config

    def _create_riemannian_only_config(self) -> Dict:
        """Create Riemannian-only configuration (no DRS subspace)"""
        config = self.base_config.copy()
        config["experiment"]["prefix"] = "Riemannian-Only"

        # Disable DRS subspace projection
        config["spherical"]["pca_energy"] = 0.0
        config["spherical"]["k_max"] = 0

        return config

    def _create_no_drs_config(self) -> Dict:
        """Create configuration without DRS subspace projection"""
        config = self.base_config.copy()
        config["experiment"]["prefix"] = "No-DRS-Projection"

        # Disable PCA-based DRS
        config["spherical"]["pca_energy"] = 0.0
        config["spherical"]["k_max"] = 0

        return config

    def _create_no_triplet_config(self) -> Dict:
        """Create configuration without Angular Triplet loss"""
        config = self.base_config.copy()
        config["experiment"]["prefix"] = "No-Angular-Triplet"

        # Disable triplet loss
        config["loss"]["triplet_lambda"] = 0.0

        return config

    def _create_no_ema_config(self) -> Dict:
        """Create configuration without EMA prototype updates"""
        config = self.base_config.copy()
        config["experiment"]["prefix"] = "No-EMA-Prototypes"

        # Disable EMA
        config["train"]["ema_momentum"] = 0.0  # No momentum = no EMA

        return config

    def _create_no_warmup_config(self) -> Dict:
        """Create configuration without warmup and annealing"""
        config = self.base_config.copy()
        config["experiment"]["prefix"] = "No-Warmup-Anneal"

        # Disable warmup
        config["train"]["epochs_warm"] = 0

        # Fixed ArcFace parameters
        config["loss"]["s_start"] = config["loss"]["s_end"]
        config["loss"]["m_start"] = config["loss"]["m_end"]

        # No label smoothing
        config["loss"]["label_smoothing"] = 0.0

        return config

    def _save_config(self, config: Dict, path: str):
        """Save configuration to file"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)


def run_comprehensive_ablation(
    base_config_path: str,
    output_dir: str = "./ablation_results/",
    run_experiments: bool = False,
):
    """Run comprehensive ablation study"""

    # Generate ablation configurations
    generator = AblationStudyGenerator(base_config_path, "./ablation_configs/")
    config_paths = generator.generate_ablation_configs()

    if not run_experiments:
        print("Ablation configurations generated. To run experiments, use:")
        for config_path in config_paths:
            print(f"  python train_hyperspherical.py --config {config_path}")
        return config_paths

    # TODO: If run_experiments=True, automatically run all configurations
    # This would require integrating with the training script

    return config_paths


def compare_ablation_results(results_dir: str = "./ablation_results/"):
    """Compare results from ablation study"""

    # Collect all result files
    result_files = []
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file == "results.json":
                result_files.append(os.path.join(root, file))

    if not result_files:
        logging.warning(f"No result files found in {results_dir}")
        return

    # Load and compare results
    comparison_data = []
    for result_file in result_files:
        with open(result_file, "r") as f:
            result = json.load(f)
        comparison_data.append(result)

    # Create comparison table
    print("\nAblation Study Comparison")
    print("=" * 80)
    print(
        f"{'Method':<25} {'Final Acc':<12} {'Avg Drift':<12} {'Avg PCA':<10} {'Tasks':<8}"
    )
    print("-" * 80)

    for result in comparison_data:
        method = result["experiment_name"]
        final_acc = result["summary"]["final_average_accuracy"]
        avg_drift = result["summary"]["average_geodesic_drift"]
        avg_pca = result["summary"]["average_pca_components"]
        total_tasks = result["summary"]["total_tasks"]

        print(
            f"{method:<25} {final_acc:<12.2f} {avg_drift:<12.4f} {avg_pca:<10.1f} {total_tasks:<8}"
        )

    return comparison_data
