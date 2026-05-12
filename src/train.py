"""Training and evaluation utilities for online learning."""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from .models import OnlineLearner
from .metrics import evaluate_online_learning, compare_online_vs_batch
from .data import OnlineDataGenerator, preprocess_data

logger = logging.getLogger(__name__)


def train_online_model(
    model: OnlineLearner,
    X_train: np.ndarray,
    y_train: np.ndarray,
    batch_size: int = 10,
    verbose: bool = True
) -> Dict[str, Any]:
    """Train an online learning model incrementally.
    
    Args:
        model: Online learning model
        X_train: Training features
        y_train: Training labels
        batch_size: Size of batches for incremental learning
        verbose: Whether to print progress
        
    Returns:
        Training results and metrics
    """
    training_metrics = []
    n_batches = len(X_train) // batch_size
    
    for i in range(0, len(X_train), batch_size):
        end_idx = min(i + batch_size, len(X_train))
        X_batch = X_train[i:end_idx]
        y_batch = y_train[i:end_idx]
        
        # Update model
        model.partial_fit(X_batch, y_batch)
        
        # Track progress
        batch_num = i // batch_size + 1
        if verbose and batch_num % 10 == 0:
            logger.info(f"Processed batch {batch_num}/{n_batches}")
        
        training_metrics.append({
            'batch': batch_num,
            'samples_seen': end_idx,
            'batch_size': len(X_batch)
        })
    
    return {
        'training_metrics': training_metrics,
        'total_samples': len(X_train),
        'total_batches': n_batches
    }


def run_online_learning_experiment(
    model_class,
    model_params: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 10,
    task_type: str = 'classification',
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """Run a complete online learning experiment.
    
    Args:
        model_class: Online learning model class
        model_params: Parameters for the model
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        batch_size: Batch size for online learning
        task_type: Type of task ('classification' or 'regression')
        random_state: Random seed
        
    Returns:
        Experiment results
    """
    # Set random seed
    if random_state is not None:
        np.random.seed(random_state)
    
    # Initialize model
    model = model_class(**model_params)
    
    # Prepare streaming data
    X_stream = []
    y_stream = []
    
    for i in range(0, len(X_train), batch_size):
        end_idx = min(i + batch_size, len(X_train))
        X_stream.append(X_train[i:end_idx])
        y_stream.append(y_train[i:end_idx])
    
    # Run evaluation
    results = evaluate_online_learning(
        model=model,
        X_stream=X_stream,
        y_stream=y_stream,
        X_test=X_test,
        y_test=y_test,
        task_type=task_type
    )
    
    # Add training info
    results['model_params'] = model.get_params()
    results['batch_size'] = batch_size
    results['task_type'] = task_type
    
    return results


def benchmark_online_algorithms(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 10,
    task_type: str = 'classification',
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """Benchmark multiple online learning algorithms.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        batch_size: Batch size for online learning
        task_type: Type of task
        random_state: Random seed
        
    Returns:
        Benchmark results
    """
    from .models import (
        OnlineSGDClassifier, OnlineSGDRegressor,
        OnlinePerceptron, OnlinePassiveAggressive,
        AdaptiveOnlineLearner
    )
    
    # Define algorithms to test
    if task_type == 'classification':
        algorithms = {
            'SGD_Classifier': (OnlineSGDClassifier, {'loss': 'log', 'random_state': random_state}),
            'Perceptron': (OnlinePerceptron, {'random_state': random_state}),
            'Passive_Aggressive': (OnlinePassiveAggressive, {'random_state': random_state}),
            'Adaptive_Learner': (AdaptiveOnlineLearner, {
                'n_features': X_train.shape[1],
                'n_classes': len(np.unique(y_train)),
                'random_state': random_state
            })
        }
    else:
        algorithms = {
            'SGD_Regressor': (OnlineSGDRegressor, {'random_state': random_state})
        }
    
    results = {}
    
    for name, (model_class, params) in algorithms.items():
        logger.info(f"Running experiment with {name}")
        
        try:
            experiment_results = run_online_learning_experiment(
                model_class=model_class,
                model_params=params,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                batch_size=batch_size,
                task_type=task_type,
                random_state=random_state
            )
            results[name] = experiment_results
        except Exception as e:
            logger.error(f"Error with {name}: {e}")
            results[name] = {'error': str(e)}
    
    return results


def create_synthetic_experiment(
    n_samples: int = 1000,
    n_features: int = 4,
    n_classes: int = 3,
    batch_size: int = 10,
    test_size: float = 0.2,
    task_type: str = 'classification',
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """Create a synthetic online learning experiment.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features
        n_classes: Number of classes (for classification)
        batch_size: Batch size for online learning
        test_size: Fraction of data for testing
        task_type: Type of task
        random_state: Random seed
        
    Returns:
        Experiment data and results
    """
    # Generate synthetic data
    data_generator = OnlineDataGenerator(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        batch_size=batch_size,
        random_state=random_state
    )
    
    if task_type == 'classification':
        X, y = data_generator.generate_classification_data()
    else:
        X, y = data_generator.generate_regression_data()
    
    # Preprocess data
    X_scaled, y_scaled, scaler = preprocess_data(X, y)
    
    # Split data
    split_idx = int(len(X_scaled) * (1 - test_size))
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]
    
    # Run benchmark
    results = benchmark_online_algorithms(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        batch_size=batch_size,
        task_type=task_type,
        random_state=random_state
    )
    
    return {
        'data_info': {
            'n_samples': n_samples,
            'n_features': n_features,
            'n_classes': n_classes if task_type == 'classification' else None,
            'train_size': len(X_train),
            'test_size': len(X_test)
        },
        'results': results
    }
