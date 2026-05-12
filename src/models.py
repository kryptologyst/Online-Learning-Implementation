"""Online learning models implementation."""

import numpy as np
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod
from sklearn.linear_model import SGDClassifier, SGDRegressor, Perceptron, PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
import logging

logger = logging.getLogger(__name__)


class OnlineLearner(ABC):
    """Abstract base class for online learning algorithms."""
    
    @abstractmethod
    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Update the model with new data incrementally."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        pass


class OnlineSGDClassifier(OnlineLearner):
    """Online Stochastic Gradient Descent Classifier."""
    
    def __init__(
        self,
        loss: str = 'log',
        learning_rate: str = 'optimal',
        eta0: float = 0.01,
        random_state: Optional[int] = None
    ):
        """Initialize SGD classifier.
        
        Args:
            loss: Loss function ('log', 'hinge', 'modified_huber')
            learning_rate: Learning rate schedule
            eta0: Initial learning rate
            random_state: Random seed
        """
        self.model = SGDClassifier(
            loss=loss,
            learning_rate=learning_rate,
            eta0=eta0,
            random_state=random_state
        )
        self.is_fitted = False
        
    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Update the model with new data."""
        if not self.is_fitted:
            classes = np.unique(y)
            self.model.partial_fit(X, y, classes=classes)
            self.is_fitted = True
        else:
            self.model.partial_fit(X, y)
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(X)
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return self.model.get_params()


class OnlineSGDRegressor(OnlineLearner):
    """Online Stochastic Gradient Descent Regressor."""
    
    def __init__(
        self,
        loss: str = 'squared_error',
        learning_rate: str = 'optimal',
        eta0: float = 0.01,
        random_state: Optional[int] = None
    ):
        """Initialize SGD regressor.
        
        Args:
            loss: Loss function ('squared_error', 'huber', 'epsilon_insensitive')
            learning_rate: Learning rate schedule
            eta0: Initial learning rate
            random_state: Random seed
        """
        self.model = SGDRegressor(
            loss=loss,
            learning_rate=learning_rate,
            eta0=eta0,
            random_state=random_state
        )
        self.is_fitted = False
        
    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Update the model with new data."""
        self.model.partial_fit(X, y)
        self.is_fitted = True
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return self.model.get_params()


class OnlinePerceptron(OnlineLearner):
    """Online Perceptron classifier."""
    
    def __init__(
        self,
        eta0: float = 1.0,
        random_state: Optional[int] = None
    ):
        """Initialize Perceptron.
        
        Args:
            eta0: Learning rate
            random_state: Random seed
        """
        self.model = Perceptron(
            eta0=eta0,
            random_state=random_state
        )
        self.is_fitted = False
        
    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Update the model with new data."""
        if not self.is_fitted:
            classes = np.unique(y)
            self.model.partial_fit(X, y, classes=classes)
            self.is_fitted = True
        else:
            self.model.partial_fit(X, y)
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return self.model.get_params()


class OnlinePassiveAggressive(OnlineLearner):
    """Online Passive-Aggressive classifier."""
    
    def __init__(
        self,
        C: float = 1.0,
        random_state: Optional[int] = None
    ):
        """Initialize Passive-Aggressive classifier.
        
        Args:
            C: Regularization parameter
            random_state: Random seed
        """
        self.model = PassiveAggressiveClassifier(
            C=C,
            random_state=random_state
        )
        self.is_fitted = False
        
    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Update the model with new data."""
        if not self.is_fitted:
            classes = np.unique(y)
            self.model.partial_fit(X, y, classes=classes)
            self.is_fitted = True
        else:
            self.model.partial_fit(X, y)
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return self.model.get_params()


class AdaptiveOnlineLearner(OnlineLearner):
    """Advanced online learner with adaptive learning rates."""
    
    def __init__(
        self,
        n_features: int,
        n_classes: int,
        learning_rate: float = 0.01,
        decay_rate: float = 0.95,
        random_state: Optional[int] = None
    ):
        """Initialize adaptive online learner.
        
        Args:
            n_features: Number of input features
            n_classes: Number of output classes
            learning_rate: Initial learning rate
            decay_rate: Learning rate decay factor
            random_state: Random seed
        """
        self.n_features = n_features
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        
        if random_state is not None:
            np.random.seed(random_state)
            
        # Initialize weights and bias
        self.weights = np.random.normal(0, 0.01, (n_features, n_classes))
        self.bias = np.zeros(n_classes)
        self.t = 0  # Time step counter
        
    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Update the model with new data using adaptive learning rate."""
        for i in range(len(X)):
            x = X[i]
            y_true = y[i]
            
            # Forward pass
            scores = np.dot(x, self.weights) + self.bias
            y_pred = np.argmax(scores)
            
            # Update if prediction is wrong
            if y_pred != y_true:
                # Adaptive learning rate
                current_lr = self.learning_rate * (self.decay_rate ** self.t)
                
                # Update weights
                self.weights[:, y_true] += current_lr * x
                self.weights[:, y_pred] -= current_lr * x
                
                # Update bias
                self.bias[y_true] += current_lr
                self.bias[y_pred] -= current_lr
                
            self.t += 1
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        scores = np.dot(X, self.weights) + self.bias
        return np.argmax(scores, axis=1)
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            'n_features': self.n_features,
            'n_classes': self.n_classes,
            'learning_rate': self.learning_rate,
            'decay_rate': self.decay_rate,
            't': self.t
        }
