"""Data loading and preprocessing utilities for online learning."""

import numpy as np
import pandas as pd
from typing import Tuple, Generator, Optional, Union
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class OnlineDataGenerator:
    """Generator for simulating streaming data in online learning scenarios."""
    
    def __init__(
        self,
        n_samples: int = 1000,
        n_features: int = 4,
        n_classes: int = 3,
        batch_size: int = 10,
        noise: float = 0.1,
        random_state: Optional[int] = None
    ):
        """Initialize the data generator.
        
        Args:
            n_samples: Total number of samples to generate
            n_features: Number of features per sample
            n_classes: Number of classes for classification
            batch_size: Size of each batch for online learning
            noise: Amount of noise to add to the data
            random_state: Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.noise = noise
        self.random_state = random_state
        
        # Set random seeds for reproducibility
        if random_state is not None:
            np.random.seed(random_state)
            
    def generate_classification_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate classification dataset.
        
        Returns:
            Tuple of (X, y) where X is features and y is labels
        """
        X, y = make_classification(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_classes=self.n_classes,
            n_redundant=0,
            n_informative=self.n_features,
            random_state=self.random_state,
            noise=self.noise
        )
        return X, y
    
    def generate_regression_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate regression dataset.
        
        Returns:
            Tuple of (X, y) where X is features and y is targets
        """
        X, y = make_regression(
            n_samples=self.n_samples,
            n_features=self.n_features,
            noise=self.noise,
            random_state=self.random_state
        )
        return X, y
    
    def stream_data(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Stream data in batches for online learning.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Yields:
            Batches of (X_batch, y_batch)
        """
        n_samples = len(X)
        for i in range(0, n_samples, self.batch_size):
            end_idx = min(i + self.batch_size, n_samples)
            yield X[i:end_idx], y[i:end_idx]


def load_iris_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load the classic Iris dataset for demonstration.
    
    Returns:
        Tuple of (X, y) where X is features and y is labels
    """
    from sklearn.datasets import load_iris
    
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    logger.info(f"Loaded Iris dataset: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y


def preprocess_data(
    X: np.ndarray, 
    y: Optional[np.ndarray] = None,
    scaler: Optional[StandardScaler] = None,
    fit_scaler: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray], StandardScaler]:
    """Preprocess data with optional scaling.
    
    Args:
        X: Feature matrix
        y: Optional target vector
        scaler: Optional pre-fitted scaler
        fit_scaler: Whether to fit the scaler on the data
        
    Returns:
        Tuple of (X_scaled, y, scaler)
    """
    if scaler is None:
        scaler = StandardScaler()
        
    if fit_scaler:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
        
    return X_scaled, y, scaler
