"""Test suite for online learning implementation."""

import pytest
import numpy as np
from typing import Dict, Any

from src.models import (
    OnlineSGDClassifier, OnlineSGDRegressor,
    OnlinePerceptron, OnlinePassiveAggressive,
    AdaptiveOnlineLearner
)
from src.data import OnlineDataGenerator, preprocess_data
from src.metrics import OnlineLearningMetrics, evaluate_online_learning
from src.train import train_online_model, run_online_learning_experiment


class TestOnlineModels:
    """Test online learning models."""
    
    def test_sgd_classifier(self):
        """Test SGD classifier functionality."""
        model = OnlineSGDClassifier(random_state=42)
        
        # Generate test data
        data_gen = OnlineDataGenerator(n_samples=100, n_features=4, n_classes=3, random_state=42)
        X, y = data_gen.generate_classification_data()
        
        # Test partial fit
        model.partial_fit(X[:50], y[:50])
        assert model.is_fitted
        
        # Test prediction
        predictions = model.predict(X[50:])
        assert len(predictions) == 50
        assert all(pred in [0, 1, 2] for pred in predictions)
        
        # Test incremental learning
        model.partial_fit(X[50:], y[50:])
        predictions_after = model.predict(X[50:])
        assert len(predictions_after) == 50
    
    def test_sgd_regressor(self):
        """Test SGD regressor functionality."""
        model = OnlineSGDRegressor(random_state=42)
        
        # Generate test data
        data_gen = OnlineDataGenerator(n_samples=100, n_features=4, random_state=42)
        X, y = data_gen.generate_regression_data()
        
        # Test partial fit
        model.partial_fit(X[:50], y[:50])
        assert model.is_fitted
        
        # Test prediction
        predictions = model.predict(X[50:])
        assert len(predictions) == 50
        assert all(isinstance(pred, (int, float)) for pred in predictions)
    
    def test_perceptron(self):
        """Test Perceptron functionality."""
        model = OnlinePerceptron(random_state=42)
        
        # Generate test data
        data_gen = OnlineDataGenerator(n_samples=100, n_features=4, n_classes=2, random_state=42)
        X, y = data_gen.generate_classification_data()
        
        # Test partial fit
        model.partial_fit(X[:50], y[:50])
        assert model.is_fitted
        
        # Test prediction
        predictions = model.predict(X[50:])
        assert len(predictions) == 50
    
    def test_passive_aggressive(self):
        """Test Passive-Aggressive classifier functionality."""
        model = OnlinePassiveAggressive(random_state=42)
        
        # Generate test data
        data_gen = OnlineDataGenerator(n_samples=100, n_features=4, n_classes=3, random_state=42)
        X, y = data_gen.generate_classification_data()
        
        # Test partial fit
        model.partial_fit(X[:50], y[:50])
        assert model.is_fitted
        
        # Test prediction
        predictions = model.predict(X[50:])
        assert len(predictions) == 50
    
    def test_adaptive_learner(self):
        """Test Adaptive Online Learner functionality."""
        model = AdaptiveOnlineLearner(n_features=4, n_classes=3, random_state=42)
        
        # Generate test data
        data_gen = OnlineDataGenerator(n_samples=100, n_features=4, n_classes=3, random_state=42)
        X, y = data_gen.generate_classification_data()
        
        # Test partial fit
        model.partial_fit(X[:50], y[:50])
        
        # Test prediction
        predictions = model.predict(X[50:])
        assert len(predictions) == 50
        assert all(pred in [0, 1, 2] for pred in predictions)
        
        # Test parameter tracking
        params = model.get_params()
        assert 't' in params
        assert params['t'] > 0


class TestDataGeneration:
    """Test data generation utilities."""
    
    def test_online_data_generator(self):
        """Test online data generator."""
        data_gen = OnlineDataGenerator(
            n_samples=100,
            n_features=4,
            n_classes=3,
            batch_size=10,
            random_state=42
        )
        
        # Test classification data generation
        X, y = data_gen.generate_classification_data()
        assert X.shape == (100, 4)
        assert y.shape == (100,)
        assert len(np.unique(y)) == 3
        
        # Test regression data generation
        X_reg, y_reg = data_gen.generate_regression_data()
        assert X_reg.shape == (100, 4)
        assert y_reg.shape == (100,)
    
    def test_data_streaming(self):
        """Test data streaming functionality."""
        data_gen = OnlineDataGenerator(n_samples=100, batch_size=10, random_state=42)
        X, y = data_gen.generate_classification_data()
        
        batches = list(data_gen.stream_data(X, y))
        assert len(batches) == 10  # 100 samples / 10 batch size
        
        total_samples = sum(len(batch[0]) for batch in batches)
        assert total_samples == 100
    
    def test_preprocess_data(self):
        """Test data preprocessing."""
        data_gen = OnlineDataGenerator(n_samples=100, random_state=42)
        X, y = data_gen.generate_classification_data()
        
        X_scaled, y_scaled, scaler = preprocess_data(X, y)
        
        assert X_scaled.shape == X.shape
        assert y_scaled.shape == y.shape
        assert scaler is not None


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_online_learning_metrics(self):
        """Test online learning metrics tracker."""
        metrics = OnlineLearningMetrics()
        
        # Test update
        y_pred = np.array([0, 1, 0, 1])
        y_true = np.array([0, 1, 1, 1])
        
        metrics.update(y_pred, y_true, loss=0.5, batch_size=4)
        
        # Test classification metrics
        class_metrics = metrics.get_classification_metrics()
        assert 'accuracy' in class_metrics
        assert 'precision' in class_metrics
        assert 'recall' in class_metrics
        assert 'f1_score' in class_metrics
        
        # Test online metrics
        online_metrics = metrics.get_online_metrics()
        assert 'avg_loss' in online_metrics
        assert 'total_samples' in online_metrics
    
    def test_evaluate_online_learning(self):
        """Test online learning evaluation."""
        model = OnlineSGDClassifier(random_state=42)
        
        # Generate test data
        data_gen = OnlineDataGenerator(n_samples=200, n_features=4, n_classes=3, random_state=42)
        X, y = data_gen.generate_classification_data()
        
        # Split data
        X_train, X_test = X[:150], X[150:]
        y_train, y_test = y[:150], y[150:]
        
        # Create streaming data
        X_stream = [X_train[i:i+10] for i in range(0, len(X_train), 10)]
        y_stream = [y_train[i:i+10] for i in range(0, len(y_train), 10)]
        
        # Run evaluation
        results = evaluate_online_learning(
            model=model,
            X_stream=X_stream,
            y_stream=y_stream,
            X_test=X_test,
            y_test=y_test,
            task_type='classification'
        )
        
        assert 'online_metrics' in results
        assert 'final_metrics' in results
        assert 'learning_curve' in results


class TestTraining:
    """Test training utilities."""
    
    def test_train_online_model(self):
        """Test online model training."""
        model = OnlineSGDClassifier(random_state=42)
        
        # Generate test data
        data_gen = OnlineDataGenerator(n_samples=100, random_state=42)
        X, y = data_gen.generate_classification_data()
        
        # Train model
        results = train_online_model(model, X, y, batch_size=10)
        
        assert 'training_metrics' in results
        assert 'total_samples' in results
        assert 'total_batches' in results
        assert results['total_samples'] == 100
        assert results['total_batches'] == 10
    
    def test_run_online_learning_experiment(self):
        """Test complete online learning experiment."""
        # Generate test data
        data_gen = OnlineDataGenerator(n_samples=200, random_state=42)
        X, y = data_gen.generate_classification_data()
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Run experiment
        results = run_online_learning_experiment(
            model_class=OnlineSGDClassifier,
            model_params={'random_state': 42},
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            batch_size=10,
            task_type='classification',
            random_state=42
        )
        
        assert 'online_metrics' in results
        assert 'final_metrics' in results
        assert 'learning_curve' in results
        assert 'model_params' in results
        assert 'batch_size' in results
        assert 'task_type' in results


if __name__ == "__main__":
    pytest.main([__file__])
