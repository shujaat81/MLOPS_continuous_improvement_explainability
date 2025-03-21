"""
Module for model monitoring and performance tracking.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import mlflow
import mlflow.sklearn
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.stats import ks_2samp
import warnings


def load_config(config_path='../configs/monitoring_config.yaml'):
    """
    Load configuration from YAML file.
    
    Parameters
    ----------
    config_path : str, default='../configs/monitoring_config.yaml'
        Path to the configuration file.
        
    Returns
    -------
    dict
        Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_mlflow(tracking_uri=None, experiment_name='fashion-mnist', run_name=None, tags=None):
    """
    Set up MLflow tracking.
    
    Parameters
    ----------
    tracking_uri : str, default=None
        MLflow tracking URI. If None, use local filesystem.
    experiment_name : str, default='fashion-mnist'
        Name of the MLflow experiment.
    run_name : str, default=None
        Name of the MLflow run. If None, use current timestamp.
    tags : dict, default=None
        Tags to set on the MLflow run.
        
    Returns
    -------
    str
        MLflow run ID.
    """
    # Set tracking URI
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    
    # Set experiment
    mlflow.set_experiment(experiment_name)
    
    # Generate run name if not provided
    if run_name is None:
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Start run
    mlflow.start_run(run_name=run_name)
    
    # Set tags
    if tags:
        mlflow.set_tags(tags)
    
    # Get run ID
    run_id = mlflow.active_run().info.run_id
    
    print(f"MLflow run started: {run_id}")
    print(f"Experiment: {experiment_name}")
    print(f"Run name: {run_name}")
    
    return run_id


def log_model_params(params):
    """
    Log model parameters to MLflow.
    
    Parameters
    ----------
    params : dict
        Model parameters.
    """
    for key, value in params.items():
        mlflow.log_param(key, value)


def log_model_metrics(metrics):
    """
    Log model metrics to MLflow.
    
    Parameters
    ----------
    metrics : dict
        Model metrics.
    """
    for key, value in metrics.items():
        mlflow.log_metric(key, value)


def log_model(model, model_name='model'):
    """
    Log model to MLflow.
    
    Parameters
    ----------
    model : object
        Trained model.
    model_name : str, default='model'
        Name of the model.
    """
    mlflow.sklearn.log_model(model, model_name)


def log_artifact(artifact_path):
    """
    Log artifact to MLflow.
    
    Parameters
    ----------
    artifact_path : str
        Path to the artifact.
    """
    mlflow.log_artifact(artifact_path)


def detect_feature_drift(reference_data, current_data, method='ks_test', threshold=0.05):
    """
    Detect feature drift between reference and current data.
    
    Parameters
    ----------
    reference_data : numpy.ndarray
        Reference data (e.g., training data).
    current_data : numpy.ndarray
        Current data (e.g., new data to monitor).
    method : str, default='ks_test'
        Method to use for drift detection. Options: 'ks_test'.
    threshold : float, default=0.05
        Threshold for drift detection.
        
    Returns
    -------
    tuple
        (drift_detected, drift_scores) whether drift was detected and drift scores for each feature.
    """
    # Flatten data if needed
    if len(reference_data.shape) > 2:
        reference_data = reference_data.reshape(reference_data.shape[0], -1)
    if len(current_data.shape) > 2:
        current_data = current_data.reshape(current_data.shape[0], -1)
    
    # Initialize drift scores
    n_features = reference_data.shape[1]
    drift_scores = np.zeros(n_features)
    
    # Detect drift for each feature
    for i in range(n_features):
        if method == 'ks_test':
            # Kolmogorov-Smirnov test
            statistic, p_value = ks_2samp(reference_data[:, i], current_data[:, i])
            drift_scores[i] = p_value
        else:
            raise ValueError(f"Unknown drift detection method: {method}")
    
    # Determine if drift was detected
    drift_detected = np.any(drift_scores < threshold)
    
    return drift_detected, drift_scores


def detect_performance_drift(reference_metrics, current_metrics, threshold=0.05):
    """
    Detect performance drift between reference and current metrics.
    
    Parameters
    ----------
    reference_metrics : dict
        Reference metrics (e.g., from validation).
    current_metrics : dict
        Current metrics (e.g., from new data).
    threshold : float, default=0.05
        Threshold for drift detection.
        
    Returns
    -------
    tuple
        (drift_detected, drift_scores) whether drift was detected and drift scores for each metric.
    """
    # Initialize drift scores
    drift_scores = {}
    
    # Detect drift for each metric
    for metric in reference_metrics:
        if metric in current_metrics:
            # Calculate relative change
            ref_value = reference_metrics[metric]
            curr_value = current_metrics[metric]
            
            if ref_value != 0:
                rel_change = abs((curr_value - ref_value) / ref_value)
            else:
                rel_change = abs(curr_value - ref_value)
            
            drift_scores[metric] = rel_change
    
    # Determine if drift was detected
    drift_detected = any(score > threshold for score in drift_scores.values())
    
    return drift_detected, drift_scores


def plot_drift_detection(drift_scores, feature_names=None, threshold=0.05, figsize=(12, 8), 
                        output_file=None):
    """
    Plot drift detection results.
    
    Parameters
    ----------
    drift_scores : numpy.ndarray
        Drift scores for each feature.
    feature_names : list, default=None
        List of feature names.
    threshold : float, default=0.05
        Threshold for drift detection.
    figsize : tuple, default=(12, 8)
        Figure size.
    output_file : str, default=None
        Path to save the figure.
    """
    # Create feature names if not provided
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(drift_scores))]
    
    # Create DataFrame for plotting
    drift_df = pd.DataFrame({
        'Feature': feature_names,
        'Drift Score': drift_scores,
        'Drift Detected': drift_scores < threshold
    })
    
    # Sort by drift score
    drift_df = drift_df.sort_values('Drift Score')
    
    # Plot drift scores
    plt.figure(figsize=figsize)
    bars = plt.barh(drift_df['Feature'], drift_df['Drift Score'], 
                   color=drift_df['Drift Detected'].map({True: 'red', False: 'green'}))
    plt.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold ({threshold})')
    plt.xlabel('Drift Score (p-value)')
    plt.ylabel('Feature')
    plt.title('Feature Drift Detection')
    plt.legend()
    plt.grid(axis='x')
    plt.tight_layout()
    
    # Save the figure if an output file is provided
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Drift detection plot saved to {output_file}")
    
    plt.show()
    
    return drift_df


def plot_metrics_over_time(metrics_history, figsize=(12, 8), output_file=None):
    """
    Plot metrics over time.
    
    Parameters
    ----------
    metrics_history : list of dict
        List of metrics dictionaries, each representing a time point.
    figsize : tuple, default=(12, 8)
        Figure size.
    output_file : str, default=None
        Path to save the figure.
    """
    # Convert to DataFrame
    metrics_df = pd.DataFrame(metrics_history)
    
    # Add timestamp if not present
    if 'timestamp' not in metrics_df.columns:
        metrics_df['timestamp'] = pd.date_range(start='today', periods=len(metrics_df), freq='D')
    
    # Set timestamp as index
    metrics_df = metrics_df.set_index('timestamp')
    
    # Plot metrics
    plt.figure(figsize=figsize)
    for column in metrics_df.columns:
        plt.plot(metrics_df.index, metrics_df[column], marker='o', label=column)
    
    plt.xlabel('Time')
    plt.ylabel('Metric Value')
    plt.title('Metrics Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save the figure if an output file is provided
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Metrics over time plot saved to {output_file}")
    
    plt.show()
    
    return metrics_df


def evaluate_and_log_metrics(model, X_test, y_test, run_id=None):
    """
    Evaluate model and log metrics to MLflow.
    
    Parameters
    ----------
    model : object
        Trained model.
    X_test : numpy.ndarray
        Test features.
    y_test : numpy.ndarray
        Test labels.
    run_id : str, default=None
        MLflow run ID. If None, use active run.
        
    Returns
    -------
    dict
        Evaluation metrics.
    """
    # Convert one-hot encoded labels back to integers if needed
    if len(y_test.shape) > 1:
        y_test = np.argmax(y_test, axis=1)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_test, y_pred)
    metrics['precision'] = precision_score(y_test, y_pred, average='weighted')
    metrics['recall'] = recall_score(y_test, y_pred, average='weighted')
    metrics['f1'] = f1_score(y_test, y_pred, average='weighted')
    
    # Calculate ROC AUC if possible
    try:
        y_score = model.predict_proba(X_test)
        # For multi-class, calculate macro-average ROC AUC
        n_classes = y_score.shape[1]
        metrics['roc_auc'] = roc_auc_score(
            np.eye(n_classes)[y_test], y_score, multi_class='ovr', average='macro'
        )
    except (AttributeError, ValueError):
        # Model doesn't support predict_proba or ROC AUC not applicable
        warnings.warn("ROC AUC could not be calculated.")
    
    # Log metrics to MLflow
    # Check if we're already in an active run
    active_run = mlflow.active_run()
    if active_run is not None:
        # If we're already in an active run, just log the metrics
        log_model_metrics(metrics)
    elif run_id:
        # If we're not in an active run but have a run_id, start a new run with that ID
        with mlflow.start_run(run_id=run_id):
            log_model_metrics(metrics)
    else:
        # If we're not in an active run and don't have a run_id, just log the metrics
        # (this will fail if there's no active run)
        log_model_metrics(metrics)
    
    return metrics


if __name__ == "__main__":
    # Example usage
    from src.data.load_data import load_fashion_mnist
    from src.data.preprocess import prepare_data_for_modeling
    from src.models.train_model import train_model, load_saved_model
    
    # Load configuration
    config = load_config()
    
    # Load the data
    (x_train, y_train), (x_test, y_test), class_names = load_fashion_mnist(save_to_disk=False)
    
    # Prepare the data
    x_train_prep, y_train_prep, x_test_prep, y_test_prep = prepare_data_for_modeling(
        x_train, y_train, x_test, y_test, normalize=True
    )
    
    # Flatten the data
    x_train_flat = x_train_prep.reshape(x_train_prep.shape[0], -1)
    x_test_flat = x_test_prep.reshape(x_test_prep.shape[0], -1)
    
    # Set up MLflow
    run_id = setup_mlflow(
        tracking_uri=config['mlflow']['tracking_uri'],
        experiment_name=config['mlflow']['experiment_name'],
        run_name=config['mlflow']['run_name'],
        tags=config['mlflow']['tags']
    )
    
    # Try to load a saved model, or train a new one if not found
    try:
        model = load_saved_model('random_forest_model.pkl')
        log_model_params(model.get_params())
    except FileNotFoundError:
        print("No saved model found. Training a new model...")
        model = train_model(x_train_flat, y_train_prep, model_type='random_forest')
        log_model_params(model.get_params())
    
    # Evaluate and log metrics
    metrics = evaluate_and_log_metrics(model, x_test_flat, y_test_prep, run_id)
    
    # Log model
    log_model(model, 'random_forest')
    
    # Detect feature drift
    # Simulate drift by adding noise to test data
    np.random.seed(42)
    x_test_drift = x_test_flat + np.random.normal(0, 0.1, x_test_flat.shape)
    
    drift_detected, drift_scores = detect_feature_drift(
        x_train_flat, x_test_drift, 
        method=config['drift_detection']['feature_drift']['method'],
        threshold=config['drift_detection']['feature_drift']['threshold']
    )
    
    print(f"Feature drift detected: {drift_detected}")
    
    # Plot drift detection results
    feature_names = [f"pixel_{i}" for i in range(min(20, x_train_flat.shape[1]))]
    drift_df = plot_drift_detection(
        drift_scores[:20], feature_names, 
        threshold=config['drift_detection']['feature_drift']['threshold'],
        output_file='../reports/figures/feature_drift.png'
    )
    
    # Log drift detection plot
    log_artifact('../reports/figures/feature_drift.png')
    
    # End MLflow run
    mlflow.end_run() 