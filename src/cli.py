"""Command-line interface for running online learning experiments."""

import argparse
import yaml
import logging
from pathlib import Path
from typing import Dict, Any

from src.train import benchmark_online_algorithms, create_synthetic_experiment
from src.viz import (
    plot_learning_curves, plot_algorithm_comparison,
    create_results_summary_table, plot_convergence_analysis
)


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_experiment_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run experiment based on configuration."""
    experiment_config = config['experiment']
    data_config = config['data']
    
    # Create synthetic experiment
    results = create_synthetic_experiment(
        n_samples=data_config['n_samples'],
        n_features=data_config['n_features'],
        n_classes=data_config.get('n_classes', 3),
        batch_size=experiment_config['batch_size'],
        test_size=data_config['test_size'],
        task_type=experiment_config['task_type'],
        random_state=experiment_config['random_state']
    )
    
    return results


def save_results(results: Dict[str, Any], output_dir: str) -> None:
    """Save experiment results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save summary table
    summary_df = create_results_summary_table(results)
    summary_df.to_csv(output_path / "results_summary.csv", index=False)
    
    # Save plots
    plot_path = output_path / "plots"
    plot_path.mkdir(exist_ok=True)
    
    # Plot learning curves
    plot_learning_curves(
        results['results'],
        metric='accuracy' if results['data_info']['n_classes'] else 'mse',
        save_path=str(plot_path / "learning_curves.png")
    )
    
    # Plot algorithm comparison
    metric = 'final_accuracy' if results['data_info']['n_classes'] else 'final_mse'
    plot_algorithm_comparison(
        results['results'],
        metric=metric,
        save_path=str(plot_path / "algorithm_comparison.png")
    )
    
    # Plot convergence analysis
    plot_convergence_analysis(
        results['results'],
        metric='accuracy' if results['data_info']['n_classes'] else 'mse',
        save_path=str(plot_path / "convergence_analysis.png")
    )


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Online Learning Implementation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run basic experiment
  python -m src.cli --config configs/basic_experiment.yaml
  
  # Run synthetic experiment
  python -m src.cli --synthetic --n-samples 1000 --task classification
  
  # Run benchmark with custom parameters
  python -m src.cli --benchmark --batch-size 20 --verbose
        """
    )
    
    # Main options
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--synthetic', '-s',
        action='store_true',
        help='Run synthetic experiment with default parameters'
    )
    
    parser.add_argument(
        '--benchmark', '-b',
        action='store_true',
        help='Run benchmark comparison of all algorithms'
    )
    
    # Experiment parameters
    parser.add_argument(
        '--n-samples',
        type=int,
        default=1000,
        help='Number of samples for synthetic data (default: 1000)'
    )
    
    parser.add_argument(
        '--n-features',
        type=int,
        default=4,
        help='Number of features (default: 4)'
    )
    
    parser.add_argument(
        '--n-classes',
        type=int,
        default=3,
        help='Number of classes for classification (default: 3)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='Batch size for online learning (default: 10)'
    )
    
    parser.add_argument(
        '--task',
        choices=['classification', 'regression'],
        default='classification',
        help='Task type (default: classification)'
    )
    
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='assets/results',
        help='Output directory for results (default: assets/results)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        if args.config:
            # Run experiment from config file
            logger.info(f"Loading configuration from {args.config}")
            config = load_config(args.config)
            results = run_experiment_from_config(config)
            
        elif args.synthetic:
            # Run synthetic experiment
            logger.info("Running synthetic experiment")
            results = create_synthetic_experiment(
                n_samples=args.n_samples,
                n_features=args.n_features,
                n_classes=args.n_classes if args.task == 'classification' else None,
                batch_size=args.batch_size,
                task_type=args.task,
                random_state=args.random_state
            )
            
        elif args.benchmark:
            # Run benchmark comparison
            logger.info("Running benchmark comparison")
            results = create_synthetic_experiment(
                n_samples=args.n_samples,
                n_features=args.n_features,
                n_classes=args.n_classes if args.task == 'classification' else None,
                batch_size=args.batch_size,
                task_type=args.task,
                random_state=args.random_state
            )
            
        else:
            parser.print_help()
            return
        
        # Save results
        logger.info(f"Saving results to {args.output_dir}")
        save_results(results, args.output_dir)
        
        # Print summary
        print("\n" + "="*50)
        print("EXPERIMENT SUMMARY")
        print("="*50)
        
        data_info = results['data_info']
        print(f"Task Type: {data_info.get('task_type', 'classification')}")
        print(f"Total Samples: {data_info['n_samples']}")
        print(f"Features: {data_info['n_features']}")
        if data_info.get('n_classes'):
            print(f"Classes: {data_info['n_classes']}")
        print(f"Train/Test Split: {data_info['train_size']}/{data_info['test_size']}")
        
        print("\nAlgorithm Results:")
        for algorithm, result in results['results'].items():
            if 'error' in result:
                print(f"  {algorithm}: ERROR - {result['error']}")
            else:
                final_metrics = result.get('final_metrics', {})
                if 'final_accuracy' in final_metrics:
                    print(f"  {algorithm}: Accuracy = {final_metrics['final_accuracy']:.4f}")
                elif 'final_mse' in final_metrics:
                    print(f"  {algorithm}: MSE = {final_metrics['final_mse']:.4f}")
        
        print(f"\nResults saved to: {args.output_dir}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Error running experiment: {e}")
        raise


if __name__ == "__main__":
    main()
