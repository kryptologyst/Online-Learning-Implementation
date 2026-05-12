#!/usr/bin/env python3
"""Example script demonstrating online learning implementation.

This script shows how to use the online learning framework for both
classification and regression tasks.

Author: kryptologyst
GitHub: https://github.com/kryptologyst
"""

import numpy as np
import logging
from src.models import (
    OnlineSGDClassifier, OnlineSGDRegressor,
    OnlinePerceptron, OnlinePassiveAggressive,
    AdaptiveOnlineLearner
)
from src.data import OnlineDataGenerator, preprocess_data
from src.train import benchmark_online_algorithms
from src.viz import plot_learning_curves, plot_algorithm_comparison
from src.utils import set_deterministic_seed, print_experiment_summary

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_classification_example():
    """Run classification example."""
    logger.info("Running classification example...")
    
    # Set random seed for reproducibility
    set_deterministic_seed(42)
    
    # Generate synthetic classification data
    data_gen = OnlineDataGenerator(
        n_samples=1000,
        n_features=4,
        n_classes=3,
        batch_size=10,
        noise=0.1,
        random_state=42
    )
    
    X, y = data_gen.generate_classification_data()
    
    # Preprocess data
    X_scaled, y_scaled, scaler = preprocess_data(X, y)
    
    # Split data
    split_idx = int(0.8 * len(X_scaled))
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]
    
    # Run benchmark
    results = benchmark_online_algorithms(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        batch_size=10,
        task_type='classification',
        random_state=42
    )
    
    # Print summary
    print_experiment_summary({'results': results})
    
    # Create visualizations
    plot_learning_curves(results, metric='accuracy')
    plot_algorithm_comparison(results, metric='final_accuracy')
    
    return results


def run_regression_example():
    """Run regression example."""
    logger.info("Running regression example...")
    
    # Set random seed for reproducibility
    set_deterministic_seed(42)
    
    # Generate synthetic regression data
    data_gen = OnlineDataGenerator(
        n_samples=1000,
        n_features=4,
        batch_size=20,
        noise=0.15,
        random_state=42
    )
    
    X, y = data_gen.generate_regression_data()
    
    # Preprocess data
    X_scaled, y_scaled, scaler = preprocess_data(X, y)
    
    # Split data
    split_idx = int(0.8 * len(X_scaled))
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]
    
    # Run benchmark
    results = benchmark_online_algorithms(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        batch_size=20,
        task_type='regression',
        random_state=42
    )
    
    # Print summary
    print_experiment_summary({'results': results})
    
    # Create visualizations
    plot_learning_curves(results, metric='mse')
    plot_algorithm_comparison(results, metric='final_mse')
    
    return results


def run_single_algorithm_example():
    """Run example with a single algorithm."""
    logger.info("Running single algorithm example...")
    
    # Set random seed for reproducibility
    set_deterministic_seed(42)
    
    # Generate data
    data_gen = OnlineDataGenerator(n_samples=500, random_state=42)
    X, y = data_gen.generate_classification_data()
    
    # Preprocess data
    X_scaled, y_scaled, scaler = preprocess_data(X, y)
    
    # Split data
    split_idx = int(0.8 * len(X_scaled))
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]
    
    # Initialize and train model
    model = OnlineSGDClassifier(loss='log', random_state=42)
    
    # Train incrementally
    batch_size = 10
    for i in range(0, len(X_train), batch_size):
        end_idx = min(i + batch_size, len(X_train))
        X_batch = X_train[i:end_idx]
        y_batch = y_train[i:end_idx]
        
        model.partial_fit(X_batch, y_batch)
        
        # Evaluate on test set
        y_pred = model.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        
        logger.info(f"Batch {i//batch_size + 1}: Accuracy = {accuracy:.4f}")
    
    # Final evaluation
    final_predictions = model.predict(X_test)
    final_accuracy = np.mean(final_predictions == y_test)
    
    logger.info(f"Final accuracy: {final_accuracy:.4f}")
    
    return model


def main():
    """Main function to run examples."""
    logger.info("Starting online learning examples...")
    
    try:
        # Run classification example
        classification_results = run_classification_example()
        
        # Run regression example
        regression_results = run_regression_example()
        
        # Run single algorithm example
        single_model = run_single_algorithm_example()
        
        logger.info("All examples completed successfully!")
        
    except Exception as e:
        logger.error(f"Error running examples: {e}")
        raise


if __name__ == "__main__":
    main()
