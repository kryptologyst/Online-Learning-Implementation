"""Safety and compliance utilities for online learning systems.

This module provides safety checks, ethical considerations, and compliance
utilities for online learning implementations.

Author: kryptologyst
GitHub: https://github.com/kryptologyst
"""

import logging
from typing import Dict, Any, List, Optional, Union
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SafetyConfig:
    """Configuration for safety checks and compliance."""
    
    # Data privacy settings
    max_samples_per_user: int = 1000
    data_retention_days: int = 30
    anonymize_features: bool = True
    
    # Model safety settings
    max_model_size_mb: int = 100
    max_training_time_hours: int = 24
    require_human_oversight: bool = True
    
    # Ethical AI settings
    bias_threshold: float = 0.1
    fairness_metrics: List[str] = None
    explainability_required: bool = True
    
    # Compliance settings
    gdpr_compliant: bool = True
    ccpa_compliant: bool = True
    audit_logging: bool = True
    
    def __post_init__(self):
        if self.fairness_metrics is None:
            self.fairness_metrics = ['demographic_parity', 'equalized_odds']


class SafetyChecker:
    """Safety checker for online learning systems."""
    
    def __init__(self, config: Optional[SafetyConfig] = None):
        """Initialize safety checker.
        
        Args:
            config: Safety configuration
        """
        self.config = config or SafetyConfig()
        self.violations = []
        
    def check_data_privacy(self, X: np.ndarray, user_ids: Optional[List[str]] = None) -> bool:
        """Check data privacy compliance.
        
        Args:
            X: Feature matrix
            user_ids: Optional user identifiers
            
        Returns:
            True if data privacy checks pass
        """
        violations = []
        
        # Check for PII in feature names or data
        if self._contains_pii(X):
            violations.append("Potential PII detected in features")
        
        # Check user data limits
        if user_ids:
            unique_users = len(set(user_ids))
            if unique_users > self.config.max_samples_per_user:
                violations.append(f"Too many samples per user: {unique_users}")
        
        # Check data anonymization
        if self.config.anonymize_features and not self._is_anonymized(X):
            violations.append("Data not properly anonymized")
        
        if violations:
            self.violations.extend(violations)
            logger.warning(f"Data privacy violations: {violations}")
            return False
        
        logger.info("Data privacy checks passed")
        return True
    
    def check_model_safety(self, model: Any, training_time: float) -> bool:
        """Check model safety compliance.
        
        Args:
            model: Trained model
            training_time: Training time in hours
            
        Returns:
            True if model safety checks pass
        """
        violations = []
        
        # Check model size
        model_size = self._estimate_model_size(model)
        if model_size > self.config.max_model_size_mb:
            violations.append(f"Model too large: {model_size:.2f}MB")
        
        # Check training time
        if training_time > self.config.max_training_time_hours:
            violations.append(f"Training time too long: {training_time:.2f}h")
        
        # Check for human oversight requirement
        if self.config.require_human_oversight and not self._has_human_oversight():
            violations.append("Human oversight required but not detected")
        
        if violations:
            self.violations.extend(violations)
            logger.warning(f"Model safety violations: {violations}")
            return False
        
        logger.info("Model safety checks passed")
        return True
    
    def check_bias_and_fairness(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               sensitive_features: Optional[np.ndarray] = None) -> bool:
        """Check for bias and fairness issues.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            sensitive_features: Optional sensitive attributes
            
        Returns:
            True if bias checks pass
        """
        violations = []
        
        # Check overall bias
        bias_score = self._calculate_bias_score(y_true, y_pred)
        if bias_score > self.config.bias_threshold:
            violations.append(f"High bias detected: {bias_score:.3f}")
        
        # Check fairness metrics if sensitive features provided
        if sensitive_features is not None:
            fairness_scores = self._calculate_fairness_metrics(
                y_true, y_pred, sensitive_features
            )
            
            for metric, score in fairness_scores.items():
                if score > self.config.bias_threshold:
                    violations.append(f"Unfair {metric}: {score:.3f}")
        
        if violations:
            self.violations.extend(violations)
            logger.warning(f"Bias/fairness violations: {violations}")
            return False
        
        logger.info("Bias and fairness checks passed")
        return True
    
    def check_compliance(self, experiment_config: Dict[str, Any]) -> bool:
        """Check regulatory compliance.
        
        Args:
            experiment_config: Experiment configuration
            
        Returns:
            True if compliance checks pass
        """
        violations = []
        
        # GDPR compliance
        if self.config.gdpr_compliant:
            if not self._check_gdpr_compliance(experiment_config):
                violations.append("GDPR compliance issues detected")
        
        # CCPA compliance
        if self.config.ccpa_compliant:
            if not self._check_ccpa_compliance(experiment_config):
                violations.append("CCPA compliance issues detected")
        
        # Audit logging
        if self.config.audit_logging and not self._has_audit_logging():
            violations.append("Audit logging not enabled")
        
        if violations:
            self.violations.extend(violations)
            logger.warning(f"Compliance violations: {violations}")
            return False
        
        logger.info("Compliance checks passed")
        return True
    
    def get_safety_report(self) -> Dict[str, Any]:
        """Generate comprehensive safety report.
        
        Returns:
            Safety report dictionary
        """
        return {
            'config': self.config.__dict__,
            'violations': self.violations,
            'status': 'PASS' if not self.violations else 'FAIL',
            'recommendations': self._generate_recommendations()
        }
    
    def _contains_pii(self, X: np.ndarray) -> bool:
        """Check if data contains potential PII."""
        # Simple heuristic checks
        if X.dtype.kind in ['U', 'S']:  # String data
            return True
        
        # Check for patterns that might indicate PII
        if X.shape[1] > 50:  # Too many features might indicate raw data
            return True
        
        return False
    
    def _is_anonymized(self, X: np.ndarray) -> bool:
        """Check if data appears to be anonymized."""
        # Check for obvious identifiers
        if X.dtype.kind in ['U', 'S']:
            return False
        
        # Check for sequential patterns
        if np.allclose(np.diff(X[:, 0]), 1):  # Sequential IDs
            return False
        
        return True
    
    def _estimate_model_size(self, model: Any) -> float:
        """Estimate model size in MB."""
        # Simple estimation based on model parameters
        if hasattr(model, 'coef_'):
            return len(model.coef_.flatten()) * 4 / (1024 * 1024)  # 4 bytes per float
        return 1.0  # Default estimate
    
    def _has_human_oversight(self) -> bool:
        """Check if human oversight is in place."""
        # This would integrate with actual oversight systems
        return True  # Placeholder
    
    def _calculate_bias_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate bias score."""
        # Simple bias calculation
        accuracy_by_class = []
        for class_label in np.unique(y_true):
            mask = y_true == class_label
            if np.sum(mask) > 0:
                class_accuracy = np.mean(y_pred[mask] == y_true[mask])
                accuracy_by_class.append(class_accuracy)
        
        if len(accuracy_by_class) > 1:
            return np.std(accuracy_by_class)
        return 0.0
    
    def _calculate_fairness_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   sensitive_features: np.ndarray) -> Dict[str, float]:
        """Calculate fairness metrics."""
        metrics = {}
        
        for group in np.unique(sensitive_features):
            mask = sensitive_features == group
            if np.sum(mask) > 0:
                group_accuracy = np.mean(y_pred[mask] == y_true[mask])
                metrics[f'accuracy_group_{group}'] = group_accuracy
        
        return metrics
    
    def _check_gdpr_compliance(self, config: Dict[str, Any]) -> bool:
        """Check GDPR compliance."""
        # Check for data minimization, purpose limitation, etc.
        return True  # Placeholder
    
    def _check_ccpa_compliance(self, config: Dict[str, Any]) -> bool:
        """Check CCPA compliance."""
        # Check for consumer rights, data transparency, etc.
        return True  # Placeholder
    
    def _has_audit_logging(self) -> bool:
        """Check if audit logging is enabled."""
        return logging.getLogger().level <= logging.INFO
    
    def _generate_recommendations(self) -> List[str]:
        """Generate safety recommendations."""
        recommendations = []
        
        if self.violations:
            recommendations.append("Address all safety violations before deployment")
        
        recommendations.extend([
            "Implement regular bias audits",
            "Maintain data privacy documentation",
            "Enable comprehensive logging",
            "Conduct regular security reviews",
            "Implement model versioning and rollback capabilities"
        ])
        
        return recommendations


def create_safety_disclaimer(task_type: str) -> str:
    """Create appropriate safety disclaimer for the task type.
    
    Args:
        task_type: Type of machine learning task
        
    Returns:
        Safety disclaimer text
    """
    base_disclaimer = """
    ⚠️ IMPORTANT SAFETY DISCLAIMERS ⚠️
    
    This is a RESEARCH AND EDUCATION focused implementation.
    
    NOT FOR PRODUCTION USE:
    - This implementation is for educational and research purposes only
    - No performance guarantees for production applications
    - Models may not perform well on real-world data
    - Synthetic data may not reflect real-world complexity
    
    ETHICAL CONSIDERATIONS:
    - Online learning can amplify biases in streaming data
    - Ensure compliance with data protection regulations
    - Maintain human oversight for critical decisions
    - Document model decisions and limitations
    
    RESPONSIBLE USE:
    - Academic research and education
    - Prototype development and proof-of-concept
    - Algorithm comparison and benchmarking
    - Learning tool for understanding online learning concepts
    
    Author: kryptologyst | GitHub: https://github.com/kryptologyst
    License: MIT | Research/Education Use Only
    """
    
    task_specific_disclaimers = {
        'classification': """
    CLASSIFICATION-SPECIFIC WARNINGS:
    - Classification decisions should not be used for critical life decisions
    - Ensure proper validation on diverse datasets
    - Consider fairness and bias implications across different groups
    - Implement proper confidence calibration
        """,
        'regression': """
    REGRESSION-SPECIFIC WARNINGS:
    - Regression predictions should not be used for financial or medical decisions
    - Ensure proper uncertainty quantification
    - Validate predictions on out-of-distribution data
    - Consider the impact of prediction errors
        """,
        'recommendation': """
    RECOMMENDATION-SPECIFIC WARNINGS:
    - Recommendations should not be used for critical decision-making
    - Ensure diversity and fairness in recommendations
    - Implement proper filtering for inappropriate content
    - Consider user privacy and data protection
        """
    }
    
    disclaimer = base_disclaimer
    if task_type in task_specific_disclaimers:
        disclaimer += task_specific_disclaimers[task_type]
    
    return disclaimer


def log_safety_check(safety_report: Dict[str, Any]) -> None:
    """Log safety check results.
    
    Args:
        safety_report: Safety report from SafetyChecker
    """
    logger.info("="*50)
    logger.info("SAFETY CHECK REPORT")
    logger.info("="*50)
    
    logger.info(f"Status: {safety_report['status']}")
    
    if safety_report['violations']:
        logger.warning("VIOLATIONS DETECTED:")
        for violation in safety_report['violations']:
            logger.warning(f"  - {violation}")
    
    logger.info("RECOMMENDATIONS:")
    for recommendation in safety_report['recommendations']:
        logger.info(f"  - {recommendation}")
    
    logger.info("="*50)
