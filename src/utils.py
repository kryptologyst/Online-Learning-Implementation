"""Utility functions for online learning implementation."""

import numpy as np
import random
import logging
from typing import Optional, Dict, Any, Union
import os

logger = logging.getLogger(__name__)


def set_deterministic_seed(seed: int) -> None:
    """Set deterministic seed for all random number generators.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Set environment variables for CUDA/MPS determinism
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    logger.info(f"Set deterministic seed: {seed}")


def get_device_info() -> Dict[str, Any]:
    """Get information about available compute devices.
    
    Returns:
        Dictionary with device information
    """
    device_info = {
        'cpu_available': True,
        'cuda_available': False,
        'mps_available': False,
        'device': 'cpu'
    }
    
    try:
        import torch
        if torch.cuda.is_available():
            device_info['cuda_available'] = True
            device_info['device'] = 'cuda'
            device_info['cuda_device_count'] = torch.cuda.device_count()
            device_info['cuda_device_name'] = torch.cuda.get_device_name(0)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device_info['mps_available'] = True
            device_info['device'] = 'mps'
    except ImportError:
        logger.info("PyTorch not available, using CPU only")
    
    logger.info(f"Using device: {device_info['device']}")
    return device_info


def validate_data(X: np.ndarray, y: Optional[np.ndarray] = None, 
                 task_type: str = 'classification') -> bool:
    """Validate input data for online learning.
    
    Args:
        X: Feature matrix
        y: Optional target vector
        task_type: Type of task ('classification' or 'regression')
        
    Returns:
        True if data is valid, False otherwise
    """
    if X is None or len(X) == 0:
        logger.error("Feature matrix X is empty or None")
        return False
    
    if X.ndim != 2:
        logger.error(f"Feature matrix X must be 2D, got {X.ndim}D")
        return False
    
    if y is not None:
        if len(y) != len(X):
            logger.error(f"Length mismatch: X has {len(X)} samples, y has {len(y)}")
            return False
        
        if task_type == 'classification':
            unique_classes = np.unique(y)
            if len(unique_classes) < 2:
                logger.error("Classification requires at least 2 classes")
                return False
    
    logger.info(f"Data validation passed: {X.shape[0]} samples, {X.shape[1]} features")
    return True


def create_directory_structure(base_path: str) -> None:
    """Create standard directory structure for the project.
    
    Args:
        base_path: Base path for the project
    """
    directories = [
        'src',
        'configs',
        'data/raw',
        'data/processed',
        'assets/plots',
        'assets/results',
        'tests',
        'scripts',
        'demo',
        'notebooks'
    ]
    
    for directory in directories:
        dir_path = os.path.join(base_path, directory)
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> Dict[str, str]:
    """Format metrics dictionary with specified precision.
    
    Args:
        metrics: Dictionary of metric names and values
        precision: Number of decimal places
        
    Returns:
        Dictionary with formatted metric values
    """
    formatted = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            formatted[key] = f"{value:.{precision}f}"
        else:
            formatted[key] = str(value)
    
    return formatted


def log_experiment_info(experiment_name: str, config: Dict[str, Any]) -> None:
    """Log experiment configuration information.
    
    Args:
        experiment_name: Name of the experiment
        config: Experiment configuration
    """
    logger.info(f"Starting experiment: {experiment_name}")
    logger.info("Configuration:")
    
    for key, value in config.items():
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for sub_key, sub_value in value.items():
                logger.info(f"    {sub_key}: {sub_value}")
        else:
            logger.info(f"  {key}: {value}")


def save_experiment_config(config: Dict[str, Any], file_path: str) -> None:
    """Save experiment configuration to file.
    
    Args:
        config: Configuration dictionary
        file_path: Path to save the configuration
    """
    import yaml
    
    with open(file_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    logger.info(f"Configuration saved to: {file_path}")


def load_experiment_config(file_path: str) -> Dict[str, Any]:
    """Load experiment configuration from file.
    
    Args:
        file_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    import yaml
    
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Configuration loaded from: {file_path}")
    return config


def calculate_statistics(values: Union[list, np.ndarray]) -> Dict[str, float]:
    """Calculate basic statistics for a list of values.
    
    Args:
        values: List or array of numeric values
        
    Returns:
        Dictionary with statistics
    """
    values = np.array(values)
    
    stats = {
        'mean': np.mean(values),
        'std': np.std(values),
        'min': np.min(values),
        'max': np.max(values),
        'median': np.median(values),
        'q25': np.percentile(values, 25),
        'q75': np.percentile(values, 75)
    }
    
    return stats


def print_experiment_summary(results: Dict[str, Any]) -> None:
    """Print a formatted summary of experiment results.
    
    Args:
        results: Experiment results dictionary
    """
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    # Data information
    if 'data_info' in results:
        data_info = results['data_info']
        print(f"Task Type: {data_info.get('task_type', 'classification')}")
        print(f"Total Samples: {data_info['n_samples']}")
        print(f"Features: {data_info['n_features']}")
        if data_info.get('n_classes'):
            print(f"Classes: {data_info['n_classes']}")
        print(f"Train/Test Split: {data_info['train_size']}/{data_info['test_size']}")
    
    # Algorithm results
    if 'results' in results:
        print("\nAlgorithm Performance:")
        print("-" * 40)
        
        for algorithm, result in results['results'].items():
            if 'error' in result:
                print(f"{algorithm:20s}: ERROR - {result['error']}")
            else:
                final_metrics = result.get('final_metrics', {})
                if 'final_accuracy' in final_metrics:
                    print(f"{algorithm:20s}: Accuracy = {final_metrics['final_accuracy']:.4f}")
                elif 'final_mse' in final_metrics:
                    print(f"{algorithm:20s}: MSE = {final_metrics['final_mse']:.4f}")
                elif 'final_r2' in final_metrics:
                    print(f"{algorithm:20s}: R² = {final_metrics['final_r2']:.4f}")
    
    print("="*60)


def check_dependencies() -> Dict[str, bool]:
    """Check if required dependencies are available.
    
    Returns:
        Dictionary with dependency availability status
    """
    dependencies = {
        'numpy': False,
        'pandas': False,
        'sklearn': False,
        'matplotlib': False,
        'seaborn': False,
        'plotly': False,
        'streamlit': False,
        'yaml': False
    }
    
    for dep in dependencies:
        try:
            if dep == 'sklearn':
                import sklearn
            elif dep == 'yaml':
                import yaml
            else:
                __import__(dep)
            dependencies[dep] = True
        except ImportError:
            logger.warning(f"Dependency {dep} not available")
    
    return dependencies
