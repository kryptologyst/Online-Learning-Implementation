"""Evaluation metrics and utilities for online learning."""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import logging

logger = logging.getLogger(__name__)


class OnlineLearningMetrics:
    """Metrics tracker for online learning evaluation."""
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.reset()
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.predictions = []
        self.true_labels = []
        self.losses = []
        self.batch_sizes = []
        
    def update(self, y_pred: np.ndarray, y_true: np.ndarray, 
               loss: Optional[float] = None, batch_size: Optional[int] = None) -> None:
        """Update metrics with new predictions.
        
        Args:
            y_pred: Predicted labels/values
            y_true: True labels/values
            loss: Optional loss value
            batch_size: Optional batch size
        """
        self.predictions.extend(y_pred)
        self.true_labels.extend(y_true)
        
        if loss is not None:
            self.losses.append(loss)
        if batch_size is not None:
            self.batch_sizes.append(batch_size)
    
    def get_classification_metrics(self) -> Dict[str, float]:
        """Get classification metrics."""
        if not self.predictions:
            return {}
            
        y_pred = np.array(self.predictions)
        y_true = np.array(self.true_labels)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        return metrics
    
    def get_regression_metrics(self) -> Dict[str, float]:
        """Get regression metrics."""
        if not self.predictions:
            return {}
            
        y_pred = np.array(self.predictions)
        y_true = np.array(self.true_labels)
        
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
        
        return metrics
    
    def get_online_metrics(self) -> Dict[str, Any]:
        """Get online learning specific metrics."""
        metrics = {}
        
        if self.losses:
            metrics['avg_loss'] = np.mean(self.losses)
            metrics['final_loss'] = self.losses[-1]
            metrics['loss_std'] = np.std(self.losses)
            
        if self.batch_sizes:
            metrics['avg_batch_size'] = np.mean(self.batch_sizes)
            metrics['total_samples'] = sum(self.batch_sizes)
            
        return metrics


def evaluate_online_learning(
    model,
    X_stream: List[np.ndarray],
    y_stream: List[np.ndarray],
    X_test: np.ndarray,
    y_test: np.ndarray,
    task_type: str = 'classification'
) -> Dict[str, Any]:
    """Evaluate online learning performance.
    
    Args:
        model: Online learning model
        X_stream: List of feature batches for streaming
        y_stream: List of label batches for streaming
        X_test: Test features
        y_test: Test labels
        task_type: Type of task ('classification' or 'regression')
        
    Returns:
        Dictionary of evaluation results
    """
    metrics_tracker = OnlineLearningMetrics()
    online_metrics = []
    
    # Stream through data and update model
    for i, (X_batch, y_batch) in enumerate(zip(X_stream, y_stream)):
        # Update model
        model.partial_fit(X_batch, y_batch)
        
        # Evaluate on test set
        y_pred = model.predict(X_test)
        
        if task_type == 'classification':
            accuracy = accuracy_score(y_test, y_pred)
            online_metrics.append({
                'batch': i,
                'accuracy': accuracy,
                'samples_seen': sum(len(y) for y in y_stream[:i+1])
            })
        else:
            mse = mean_squared_error(y_test, y_pred)
            online_metrics.append({
                'batch': i,
                'mse': mse,
                'samples_seen': sum(len(y) for y in y_stream[:i+1])
            })
    
    # Final evaluation
    final_predictions = model.predict(X_test)
    
    if task_type == 'classification':
        final_metrics = {
            'final_accuracy': accuracy_score(y_test, final_predictions),
            'final_precision': precision_score(y_test, final_predictions, average='weighted', zero_division=0),
            'final_recall': recall_score(y_test, final_predictions, average='weighted', zero_division=0),
            'final_f1': f1_score(y_test, final_predictions, average='weighted', zero_division=0)
        }
    else:
        final_metrics = {
            'final_mse': mean_squared_error(y_test, final_predictions),
            'final_rmse': np.sqrt(mean_squared_error(y_test, final_predictions)),
            'final_mae': mean_absolute_error(y_test, final_predictions),
            'final_r2': r2_score(y_test, final_predictions)
        }
    
    return {
        'online_metrics': online_metrics,
        'final_metrics': final_metrics,
        'learning_curve': online_metrics
    }


def compare_online_vs_batch(
    online_model,
    batch_model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 10,
    task_type: str = 'classification'
) -> Dict[str, Any]:
    """Compare online learning vs batch learning performance.
    
    Args:
        online_model: Online learning model
        batch_model: Batch learning model
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        batch_size: Batch size for online learning
        task_type: Type of task
        
    Returns:
        Comparison results
    """
    # Train batch model
    batch_model.fit(X_train, y_train)
    batch_predictions = batch_model.predict(X_test)
    
    # Train online model incrementally
    online_metrics = []
    for i in range(0, len(X_train), batch_size):
        end_idx = min(i + batch_size, len(X_train))
        X_batch = X_train[i:end_idx]
        y_batch = y_train[i:end_idx]
        
        online_model.partial_fit(X_batch, y_batch)
        
        # Evaluate online model
        online_predictions = online_model.predict(X_test)
        
        if task_type == 'classification':
            online_acc = accuracy_score(y_test, online_predictions)
            batch_acc = accuracy_score(y_test, batch_predictions)
            online_metrics.append({
                'samples_seen': end_idx,
                'online_accuracy': online_acc,
                'batch_accuracy': batch_acc,
                'accuracy_gap': batch_acc - online_acc
            })
        else:
            online_mse = mean_squared_error(y_test, online_predictions)
            batch_mse = mean_squared_error(y_test, batch_predictions)
            online_metrics.append({
                'samples_seen': end_idx,
                'online_mse': online_mse,
                'batch_mse': batch_mse,
                'mse_gap': online_mse - batch_mse
            })
    
    # Final comparison
    final_online_predictions = online_model.predict(X_test)
    
    if task_type == 'classification':
        final_metrics = {
            'online_accuracy': accuracy_score(y_test, final_online_predictions),
            'batch_accuracy': accuracy_score(y_test, batch_predictions),
            'accuracy_difference': accuracy_score(y_test, batch_predictions) - accuracy_score(y_test, final_online_predictions)
        }
    else:
        final_metrics = {
            'online_mse': mean_squared_error(y_test, final_online_predictions),
            'batch_mse': mean_squared_error(y_test, batch_predictions),
            'mse_difference': mean_squared_error(y_test, final_online_predictions) - mean_squared_error(y_test, batch_predictions)
        }
    
    return {
        'comparison_metrics': online_metrics,
        'final_comparison': final_metrics,
        'learning_curves': online_metrics
    }
