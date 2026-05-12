"""Visualization utilities for online learning results."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_learning_curves(
    results: Dict[str, Any],
    metric: str = 'accuracy',
    save_path: Optional[str] = None
) -> None:
    """Plot learning curves for online learning experiments.
    
    Args:
        results: Results from online learning experiments
        metric: Metric to plot ('accuracy', 'mse', etc.)
        save_path: Optional path to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for algorithm_name, algorithm_results in results.items():
        if 'error' in algorithm_results:
            continue
            
        learning_curve = algorithm_results.get('learning_curve', [])
        if not learning_curve:
            continue
            
        # Extract data
        batches = [point['batch'] for point in learning_curve]
        samples_seen = [point['samples_seen'] for point in learning_curve]
        
        if metric in learning_curve[0]:
            metric_values = [point[metric] for point in learning_curve]
        else:
            continue
            
        # Plot
        ax.plot(samples_seen, metric_values, label=algorithm_name, linewidth=2, marker='o', markersize=4)
    
    ax.set_xlabel('Samples Seen', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'Online Learning Curves - {metric.replace("_", " ").title()}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Learning curves saved to {save_path}")
    
    plt.show()


def plot_algorithm_comparison(
    results: Dict[str, Any],
    metric: str = 'final_accuracy',
    save_path: Optional[str] = None
) -> None:
    """Plot comparison of final performance across algorithms.
    
    Args:
        results: Results from online learning experiments
        metric: Final metric to compare
        save_path: Optional path to save the plot
    """
    # Extract final metrics
    algorithm_names = []
    final_metrics = []
    
    for algorithm_name, algorithm_results in results.items():
        if 'error' in algorithm_results:
            continue
            
        final_metrics_dict = algorithm_results.get('final_metrics', {})
        if metric in final_metrics_dict:
            algorithm_names.append(algorithm_name)
            final_metrics.append(final_metrics_dict[metric])
    
    if not algorithm_names:
        logger.warning(f"No data found for metric {metric}")
        return
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(algorithm_names, final_metrics, alpha=0.7)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'Algorithm Comparison - {metric.replace("_", " ").title()}', fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, final_metrics):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{value:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Algorithm comparison saved to {save_path}")
    
    plt.show()


def plot_batch_size_analysis(
    results: Dict[str, Any],
    save_path: Optional[str] = None
) -> None:
    """Plot analysis of batch size effects on learning.
    
    Args:
        results: Results from online learning experiments
        save_path: Optional path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    for algorithm_name, algorithm_results in results.items():
        if 'error' in algorithm_results:
            continue
            
        learning_curve = algorithm_results.get('learning_curve', [])
        if not learning_curve:
            continue
            
        # Extract data
        samples_seen = [point['samples_seen'] for point in learning_curve]
        
        # Plot accuracy/MSE over time
        if 'accuracy' in learning_curve[0]:
            metric_values = [point['accuracy'] for point in learning_curve]
            ax1.plot(samples_seen, metric_values, label=algorithm_name, linewidth=2)
        elif 'mse' in learning_curve[0]:
            metric_values = [point['mse'] for point in learning_curve]
            ax2.plot(samples_seen, metric_values, label=algorithm_name, linewidth=2)
    
    ax1.set_xlabel('Samples Seen')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Learning Curves - Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Samples Seen')
    ax2.set_ylabel('MSE')
    ax2.set_title('Learning Curves - MSE')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Batch size analysis saved to {save_path}")
    
    plt.show()


def create_results_summary_table(
    results: Dict[str, Any],
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """Create a summary table of results.
    
    Args:
        results: Results from online learning experiments
        save_path: Optional path to save the table
        
    Returns:
        DataFrame with summary results
    """
    summary_data = []
    
    for algorithm_name, algorithm_results in results.items():
        if 'error' in algorithm_results:
            summary_data.append({
                'Algorithm': algorithm_name,
                'Status': 'Error',
                'Error': algorithm_results['error']
            })
            continue
            
        final_metrics = algorithm_results.get('final_metrics', {})
        
        row = {'Algorithm': algorithm_name}
        
        # Add final metrics
        for metric, value in final_metrics.items():
            row[metric] = f"{value:.4f}"
        
        # Add training info
        row['Batch Size'] = algorithm_results.get('batch_size', 'N/A')
        row['Task Type'] = algorithm_results.get('task_type', 'N/A')
        
        summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    
    if save_path:
        df.to_csv(save_path, index=False)
        logger.info(f"Results summary saved to {save_path}")
    
    return df


def plot_convergence_analysis(
    results: Dict[str, Any],
    metric: str = 'accuracy',
    threshold: float = 0.95,
    save_path: Optional[str] = None
) -> None:
    """Plot convergence analysis showing when algorithms reach performance threshold.
    
    Args:
        results: Results from online learning experiments
        metric: Metric to analyze
        threshold: Performance threshold to reach
        save_path: Optional path to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    convergence_data = []
    
    for algorithm_name, algorithm_results in results.items():
        if 'error' in algorithm_results:
            continue
            
        learning_curve = algorithm_results.get('learning_curve', [])
        if not learning_curve or metric not in learning_curve[0]:
            continue
            
        # Find convergence point
        samples_seen = [point['samples_seen'] for point in learning_curve]
        metric_values = [point[metric] for point in learning_curve]
        
        convergence_point = None
        for i, value in enumerate(metric_values):
            if value >= threshold:
                convergence_point = samples_seen[i]
                break
        
        convergence_data.append({
            'Algorithm': algorithm_name,
            'Convergence Point': convergence_point,
            'Final Performance': metric_values[-1]
        })
        
        # Plot learning curve
        ax.plot(samples_seen, metric_values, label=algorithm_name, linewidth=2)
        
        # Mark convergence point
        if convergence_point:
            ax.axvline(x=convergence_point, color='red', linestyle='--', alpha=0.5)
    
    ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, label=f'Threshold ({threshold})')
    ax.set_xlabel('Samples Seen', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'Convergence Analysis - {metric.replace("_", " ").title()}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Convergence analysis saved to {save_path}")
    
    plt.show()
    
    # Print convergence summary
    convergence_df = pd.DataFrame(convergence_data)
    print("\nConvergence Summary:")
    print(convergence_df.to_string(index=False))
