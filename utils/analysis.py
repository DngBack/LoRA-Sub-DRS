#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Analysis and logging utilities for AD-DRS experiments

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
import logging


class ADDRSAnalyzer:
    """
    Comprehensive analysis and logging for AD-DRS experiments.
    """
    
    def __init__(self, experiment_name, save_dir="experiments"):
        self.experiment_name = experiment_name
        self.save_dir = save_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = os.path.join(save_dir, f"{experiment_name}_{self.timestamp}")
        
        # Create experiment directory
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Initialize logging
        self.setup_logging()
        
        # Data storage
        self.results = {
            'lambda_history': [],
            'task_accuracies': [],
            'average_accuracies': [],
            'forgetting_measures': [],
            'config': {},
            'task_times': [],
            'fisher_norms': []
        }
    
    def setup_logging(self):
        """Setup comprehensive logging for the experiment."""
        log_file = os.path.join(self.experiment_dir, "experiment.log")
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        
        # Get root logger
        logger = logging.getLogger()
        logger.addHandler(file_handler)
        
        logging.info(f"=== AD-DRS Experiment Started: {self.experiment_name} ===")
        logging.info(f"Experiment directory: {self.experiment_dir}")
    
    def log_config(self, config):
        """Log experiment configuration."""
        self.results['config'] = config
        
        config_file = os.path.join(self.experiment_dir, "config.json")
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logging.info(f"Configuration saved to {config_file}")
    
    def log_lambda_value(self, task_id, lambda_value):
        """Log lambda value for a specific task."""
        self.results['lambda_history'].append({
            'task': task_id,
            'lambda': float(lambda_value),
            'timestamp': datetime.now().isoformat()
        })
        
        logging.info(f"Task {task_id}: λ* = {lambda_value:.6f}")
    
    def log_task_results(self, task_id, task_acc, avg_acc, task_time=None):
        """Log results for a completed task."""
        self.results['task_accuracies'].append({
            'task': task_id,
            'accuracy': float(task_acc),
            'timestamp': datetime.now().isoformat()
        })
        
        self.results['average_accuracies'].append({
            'task': task_id,
            'average_accuracy': float(avg_acc),
            'timestamp': datetime.now().isoformat()
        })
        
        if task_time is not None:
            self.results['task_times'].append({
                'task': task_id,
                'time_seconds': float(task_time),
                'timestamp': datetime.now().isoformat()
            })
        
        logging.info(f"Task {task_id} completed: Acc = {task_acc:.2f}%, Avg = {avg_acc:.2f}%")
    
    def log_fisher_norm(self, task_id, fisher_norm):
        """Log Fisher Information Matrix norm."""
        self.results['fisher_norms'].append({
            'task': task_id,
            'fisher_norm': float(fisher_norm),
            'timestamp': datetime.now().isoformat()
        })
    
    def compute_forgetting_measure(self, all_task_accs):
        """
        Compute forgetting measure for each task.
        
        Args:
            all_task_accs: List of lists, where all_task_accs[i][j] is accuracy of task j after learning task i
        """
        if len(all_task_accs) < 2:
            return
        
        num_tasks = len(all_task_accs[-1])
        forgetting = []
        
        for j in range(num_tasks - 1):  # Don't compute for the last task
            max_acc = max(all_task_accs[i][j] for i in range(j, len(all_task_accs)))
            final_acc = all_task_accs[-1][j]
            forgetting.append(max_acc - final_acc)
        
        avg_forgetting = np.mean(forgetting) if forgetting else 0.0
        
        self.results['forgetting_measures'].append({
            'task': len(all_task_accs) - 1,
            'forgetting_per_task': forgetting,
            'average_forgetting': float(avg_forgetting),
            'timestamp': datetime.now().isoformat()
        })
        
        logging.info(f"Average forgetting after task {len(all_task_accs) - 1}: {avg_forgetting:.2f}%")
    
    def generate_plots(self):
        """Generate analysis plots."""
        self._plot_lambda_evolution()
        self._plot_accuracy_evolution()
        self._plot_forgetting_analysis()
        self._plot_fisher_evolution()
    
    def _plot_lambda_evolution(self):
        """Plot lambda values over tasks."""
        if not self.results['lambda_history']:
            return
        
        tasks = [item['task'] for item in self.results['lambda_history']]
        lambdas = [item['lambda'] for item in self.results['lambda_history']]
        
        plt.figure(figsize=(10, 6))
        plt.plot(tasks, lambdas, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Task')
        plt.ylabel('λ* (Optimal Merging Coefficient)')
        plt.title('AD-DRS: Evolution of Optimal Lambda Values')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Add statistics
        avg_lambda = np.mean(lambdas)
        std_lambda = np.std(lambdas)
        plt.axhline(y=avg_lambda, color='r', linestyle='--', alpha=0.7, 
                   label=f'Mean = {avg_lambda:.3f} ± {std_lambda:.3f}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.experiment_dir, 'lambda_evolution.png'), dpi=300)
        plt.close()
    
    def _plot_accuracy_evolution(self):
        """Plot accuracy evolution over tasks."""
        if not self.results['average_accuracies']:
            return
        
        tasks = [item['task'] for item in self.results['average_accuracies']]
        accs = [item['average_accuracy'] for item in self.results['average_accuracies']]
        
        plt.figure(figsize=(10, 6))
        plt.plot(tasks, accs, 'go-', linewidth=2, markersize=8)
        plt.xlabel('Task')
        plt.ylabel('Average Accuracy (%)')
        plt.title('AD-DRS: Average Accuracy Evolution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.experiment_dir, 'accuracy_evolution.png'), dpi=300)
        plt.close()
    
    def _plot_forgetting_analysis(self):
        """Plot forgetting analysis."""
        if not self.results['forgetting_measures']:
            return
        
        tasks = [item['task'] for item in self.results['forgetting_measures']]
        avg_forgetting = [item['average_forgetting'] for item in self.results['forgetting_measures']]
        
        plt.figure(figsize=(10, 6))
        plt.plot(tasks, avg_forgetting, 'ro-', linewidth=2, markersize=8)
        plt.xlabel('Task')
        plt.ylabel('Average Forgetting (%)')
        plt.title('AD-DRS: Forgetting Analysis')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.experiment_dir, 'forgetting_analysis.png'), dpi=300)
        plt.close()
    
    def _plot_fisher_evolution(self):
        """Plot Fisher Information Matrix norm evolution."""
        if not self.results['fisher_norms']:
            return
        
        tasks = [item['task'] for item in self.results['fisher_norms']]
        norms = [item['fisher_norm'] for item in self.results['fisher_norms']]
        
        plt.figure(figsize=(10, 6))
        plt.plot(tasks, norms, 'mo-', linewidth=2, markersize=8)
        plt.xlabel('Task')
        plt.ylabel('Fisher Information Norm')
        plt.title('AD-DRS: Fisher Information Evolution')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.experiment_dir, 'fisher_evolution.png'), dpi=300)
        plt.close()
    
    def save_results(self):
        """Save all results to JSON file."""
        results_file = os.path.join(self.experiment_dir, "results.json")
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logging.info(f"Results saved to {results_file}")
    
    def finalize_experiment(self):
        """Finalize the experiment - generate plots and save results."""
        logging.info("=== Finalizing AD-DRS Experiment ===")
        
        # Generate analysis plots
        try:
            self.generate_plots()
            logging.info("Analysis plots generated successfully")
        except Exception as e:
            logging.warning(f"Failed to generate plots: {e}")
        
        # Save results
        self.save_results()
        
        # Print summary
        self._print_summary()
        
        logging.info(f"=== Experiment completed: {self.experiment_name} ===")
    
    def _print_summary(self):
        """Print experiment summary."""
        logging.info("="*50)
        logging.info("EXPERIMENT SUMMARY")
        logging.info("="*50)
        
        if self.results['lambda_history']:
            lambdas = [item['lambda'] for item in self.results['lambda_history']]
            logging.info(f"Lambda Statistics:")
            logging.info(f"  Mean: {np.mean(lambdas):.4f}")
            logging.info(f"  Std:  {np.std(lambdas):.4f}")
            logging.info(f"  Min:  {np.min(lambdas):.4f}")
            logging.info(f"  Max:  {np.max(lambdas):.4f}")
        
        if self.results['average_accuracies']:
            final_acc = self.results['average_accuracies'][-1]['average_accuracy']
            logging.info(f"Final Average Accuracy: {final_acc:.2f}%")
        
        if self.results['forgetting_measures']:
            final_forgetting = self.results['forgetting_measures'][-1]['average_forgetting']
            logging.info(f"Final Average Forgetting: {final_forgetting:.2f}%")
        
        logging.info(f"Results saved in: {self.experiment_dir}")
        logging.info("="*50)


def run_ablation_study(base_config_path, output_dir="ablation_study"):
    """
    Run ablation study for AD-DRS.
    
    Args:
        base_config_path: Path to base configuration file
        output_dir: Directory to save ablation results
    """
    import json
    
    # Load base config
    with open(base_config_path, 'r') as f:
        base_config = json.load(f)
    
    # Define ablation variations
    ablations = {
        'baseline_lorasub': {
            'model_name': 'lorasub_drs',
            'description': 'Original LoRA-Sub without adaptive merging'
        },
        'addrs_no_refinement': {
            'model_name': 'ad_drs',
            'use_classifier_alignment': False,
            'use_self_distillation': False,
            'description': 'AD-DRS without refinement techniques'
        },
        'addrs_fixed_lambda_05': {
            'model_name': 'ad_drs',
            'fixed_lambda': 0.5,
            'description': 'AD-DRS with fixed lambda = 0.5'
        },
        'addrs_full': {
            'model_name': 'ad_drs',
            'use_classifier_alignment': True,
            'use_self_distillation': False,
            'description': 'Full AD-DRS with classifier alignment'
        }
    }
    
    # Create ablation configs
    os.makedirs(output_dir, exist_ok=True)
    
    for name, modifications in ablations.items():
        config = base_config.copy()
        config.update(modifications)
        config['prefix'] = f"Ablation_{name}"
        
        config_path = os.path.join(output_dir, f"{name}.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Created ablation config: {config_path}")
        print(f"Description: {modifications['description']}")
        print()
    
    print("Ablation study configs created!")
    print("Run experiments using:")
    for name in ablations.keys():
        print(f"  python main.py --config {output_dir}/{name}.json")
