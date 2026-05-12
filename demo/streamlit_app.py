"""Streamlit demo application for online learning."""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from typing import Dict, Any, List

# Import our modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import (
    OnlineSGDClassifier, OnlineSGDRegressor,
    OnlinePerceptron, OnlinePassiveAggressive,
    AdaptiveOnlineLearner
)
from src.data import OnlineDataGenerator, preprocess_data
from src.train import run_online_learning_experiment
from src.viz import plot_learning_curves, plot_algorithm_comparison


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    st.session_state.random_seed = seed


def generate_data(task_type: str, n_samples: int, n_features: int, 
                 n_classes: int, noise: float, random_state: int) -> tuple:
    """Generate synthetic data for the experiment."""
    data_gen = OnlineDataGenerator(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        batch_size=10,
        noise=noise,
        random_state=random_state
    )
    
    if task_type == "Classification":
        X, y = data_gen.generate_classification_data()
    else:
        X, y = data_gen.generate_regression_data()
    
    # Preprocess data
    X_scaled, y_scaled, scaler = preprocess_data(X, y)
    
    # Split data
    split_idx = int(0.8 * len(X_scaled))
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]
    
    return X_train, X_test, y_train, y_test


def run_experiment(task_type: str, algorithm: str, batch_size: int,
                  X_train: np.ndarray, y_train: np.ndarray,
                  X_test: np.ndarray, y_test: np.ndarray,
                  random_state: int) -> Dict[str, Any]:
    """Run online learning experiment."""
    
    # Define algorithms
    algorithms = {
        "SGD Classifier": (OnlineSGDClassifier, {"loss": "log", "random_state": random_state}),
        "SGD Regressor": (OnlineSGDRegressor, {"loss": "squared_error", "random_state": random_state}),
        "Perceptron": (OnlinePerceptron, {"random_state": random_state}),
        "Passive-Aggressive": (OnlinePassiveAggressive, {"random_state": random_state}),
        "Adaptive Learner": (AdaptiveOnlineLearner, {
            "n_features": X_train.shape[1],
            "n_classes": len(np.unique(y_train)) if task_type == "Classification" else 1,
            "random_state": random_state
        })
    }
    
    if algorithm not in algorithms:
        st.error(f"Algorithm {algorithm} not supported for {task_type}")
        return {}
    
    model_class, params = algorithms[algorithm]
    
    try:
        results = run_online_learning_experiment(
            model_class=model_class,
            model_params=params,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            batch_size=batch_size,
            task_type=task_type.lower(),
            random_state=random_state
        )
        return results
    except Exception as e:
        st.error(f"Error running experiment: {str(e)}")
        return {}


def plot_learning_curve_plotly(results: Dict[str, Any], metric: str) -> go.Figure:
    """Create interactive learning curve plot."""
    fig = go.Figure()
    
    learning_curve = results.get('learning_curve', [])
    if not learning_curve:
        return fig
    
    samples_seen = [point['samples_seen'] for point in learning_curve]
    metric_values = [point[metric] for point in learning_curve]
    
    fig.add_trace(go.Scatter(
        x=samples_seen,
        y=metric_values,
        mode='lines+markers',
        name=f'{metric.title()}',
        line=dict(width=2),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title=f'Online Learning Curve - {metric.title()}',
        xaxis_title='Samples Seen',
        yaxis_title=metric.title(),
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Online Learning Demo",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 Online Learning Implementation Demo")
    st.markdown("**Author:** [kryptologyst](https://github.com/kryptologyst)")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Task type selection
    task_type = st.sidebar.selectbox(
        "Task Type",
        ["Classification", "Regression"],
        help="Choose between classification or regression task"
    )
    
    # Data parameters
    st.sidebar.subheader("Data Parameters")
    n_samples = st.sidebar.slider("Number of Samples", 100, 2000, 1000)
    n_features = st.sidebar.slider("Number of Features", 2, 10, 4)
    
    if task_type == "Classification":
        n_classes = st.sidebar.slider("Number of Classes", 2, 5, 3)
    else:
        n_classes = 1
    
    noise = st.sidebar.slider("Noise Level", 0.0, 0.5, 0.1, 0.01)
    
    # Algorithm selection
    st.sidebar.subheader("Algorithm Selection")
    if task_type == "Classification":
        algorithm = st.sidebar.selectbox(
            "Algorithm",
            ["SGD Classifier", "Perceptron", "Passive-Aggressive", "Adaptive Learner"]
        )
    else:
        algorithm = st.sidebar.selectbox(
            "Algorithm",
            ["SGD Regressor"]
        )
    
    # Training parameters
    st.sidebar.subheader("Training Parameters")
    batch_size = st.sidebar.slider("Batch Size", 5, 50, 10)
    random_seed = st.sidebar.number_input("Random Seed", 0, 10000, 42)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Experiment Results")
        
        if st.button("🚀 Run Experiment", type="primary"):
            with st.spinner("Running experiment..."):
                # Generate data
                X_train, X_test, y_train, y_test = generate_data(
                    task_type, n_samples, n_features, n_classes, noise, random_seed
                )
                
                # Run experiment
                results = run_experiment(
                    task_type, algorithm, batch_size,
                    X_train, y_train, X_test, y_test, random_seed
                )
                
                if results:
                    st.session_state.results = results
                    st.session_state.data_info = {
                        'X_train': X_train, 'X_test': X_test,
                        'y_train': y_train, 'y_test': y_test
                    }
                    st.success("Experiment completed successfully!")
                else:
                    st.error("Experiment failed!")
    
    with col2:
        st.header("Data Info")
        if 'data_info' in st.session_state:
            data_info = st.session_state.data_info
            st.metric("Training Samples", len(data_info['X_train']))
            st.metric("Test Samples", len(data_info['X_test']))
            st.metric("Features", data_info['X_train'].shape[1])
            if task_type == "Classification":
                st.metric("Classes", len(np.unique(data_info['y_train'])))
    
    # Display results
    if 'results' in st.session_state:
        results = st.session_state.results
        
        # Final metrics
        st.subheader("Final Performance")
        final_metrics = results.get('final_metrics', {})
        
        if task_type == "Classification":
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{final_metrics.get('final_accuracy', 0):.4f}")
            with col2:
                st.metric("Precision", f"{final_metrics.get('final_precision', 0):.4f}")
            with col3:
                st.metric("Recall", f"{final_metrics.get('final_recall', 0):.4f}")
            with col4:
                st.metric("F1 Score", f"{final_metrics.get('final_f1', 0):.4f}")
        else:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("MSE", f"{final_metrics.get('final_mse', 0):.4f}")
            with col2:
                st.metric("RMSE", f"{final_metrics.get('final_rmse', 0):.4f}")
            with col3:
                st.metric("MAE", f"{final_metrics.get('final_mae', 0):.4f}")
            with col4:
                st.metric("R²", f"{final_metrics.get('final_r2', 0):.4f}")
        
        # Learning curve
        st.subheader("Learning Curve")
        
        if task_type == "Classification":
            metric = "accuracy"
        else:
            metric = "mse"
        
        fig = plot_learning_curve_plotly(results, metric)
        st.plotly_chart(fig, use_container_width=True)
        
        # Model parameters
        st.subheader("Model Parameters")
        model_params = results.get('model_params', {})
        st.json(model_params)
    
    # Safety disclaimer
    st.markdown("---")
    st.markdown("""
    ### ⚠️ Important Disclaimers
    
    **This is a research and education focused demonstration.**
    
    - **Not for Production Use**: This demo is for educational purposes only
    - **Synthetic Data**: Results are based on synthetic data and may not reflect real-world performance
    - **No Guarantees**: No performance guarantees for production applications
    - **Research Only**: Suitable for academic research and learning purposes
    
    **Author:** [kryptologyst](https://github.com/kryptologyst) | **License:** MIT
    """)


if __name__ == "__main__":
    main()
