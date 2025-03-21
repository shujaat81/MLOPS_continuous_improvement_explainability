"""
Module for training models on the Fashion MNIST dataset.
"""
import os
import numpy as np
import pandas as pd
import pickle
import yaml
import time
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from tpot import TPOTClassifier
import optuna


def load_config(config_path="../configs/model_config.yaml"):
    """
    Load configuration from YAML file.

    Parameters
    ----------
    config_path : str, default='../configs/model_config.yaml'
        Path to the configuration file.

    Returns
    -------
    dict
        Configuration dictionary.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def train_model(X_train, y_train, model_type="random_forest", params=None):
    """
    Train a model on the given data.

    Parameters
    ----------
    X_train : numpy.ndarray
        Training features.
    y_train : numpy.ndarray
        Training labels.
    model_type : str, default='random_forest'
        Type of model to train. Options: 'logistic_regression', 'random_forest', 'svm', 'neural_network'.
    params : dict, default=None
        Model parameters. If None, default parameters are used.

    Returns
    -------
    object
        Trained model.
    """
    # Convert one-hot encoded labels back to integers if needed
    if len(y_train.shape) > 1:
        y_train = np.argmax(y_train, axis=1)

    # Initialize model based on type
    if model_type == "logistic_regression":
        if params is None:
            params = {"C": 1.0, "solver": "liblinear", "max_iter": 1000}
        model = LogisticRegression(**params, random_state=42)

    elif model_type == "random_forest":
        if params is None:
            params = {"n_estimators": 100, "max_depth": None}
        model = RandomForestClassifier(**params, random_state=42)

    elif model_type == "svm":
        if params is None:
            params = {"C": 1.0, "kernel": "rbf", "gamma": "scale"}
        model = SVC(**params, random_state=42, probability=True)

    elif model_type == "neural_network":
        if params is None:
            params = {
                "hidden_layer_sizes": (100,),
                "activation": "relu",
                "solver": "adam",
                "max_iter": 200,
            }
        model = MLPClassifier(**params, random_state=42)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Train the model
    model.fit(X_train, y_train)

    return model


def train_with_automl(
    X_train,
    y_train,
    X_test=None,
    y_test=None,
    time_budget=3600,
    metric="accuracy",
    cv=5,
    random_state=42,
):
    """
    Train a model using TPOT AutoML with memory-efficient settings.
    """
    print(
        f"Starting lightweight AutoML with TPOT (time budget: {time_budget} seconds)..."
    )

    # Initialize TPOT with memory-efficient settings
    tpot = TPOTClassifier(
        generations=10,  # Reduced from 100
        population_size=20,  # Reduced from 100
        cv=cv,
        scoring=metric,
        random_state=random_state,
        verbosity=2,
        max_time_mins=time_budget // 60,
        n_jobs=1,  # Use single core to reduce memory usage
        memory="auto",  # Use disk for caching
        use_dask=False,  # Disable dask to reduce complexity
        config_dict="TPOT light",  # Use lightweight pipeline options
        template="Classifier",  # Restrict to simple classifiers
        early_stop=5,  # Stop if no improvement for 5 generations
    )

    start_time = time.time()

    try:
        # Sample data if too large (optional)
        if len(X_train) > 10000:
            from sklearn.model_selection import train_test_split

            print("Sampling training data to reduce memory usage...")
            X_train_sample, _, y_train_sample, _ = train_test_split(
                X_train,
                y_train,
                train_size=10000,
                random_state=random_state,
                stratify=y_train,
            )
        else:
            X_train_sample, y_train_sample = X_train, y_train

        # Fit TPOT
        if X_test is not None and y_test is not None:
            tpot.fit(X_train_sample, y_train_sample)
            test_score = tpot.score(X_test, y_test)
            print(f"\nTest set {metric}: {test_score:.4f}")
        else:
            tpot.fit(X_train_sample, y_train_sample)

        training_time = time.time() - start_time
        print(f"\nAutoML completed in {training_time:.2f} seconds")

        # Get the best model
        best_model = tpot.fitted_pipeline_

        # Print the best pipeline
        print("\nBest pipeline:")
        print(tpot.fitted_pipeline_)

        return best_model, tpot

    except MemoryError as e:
        print(f"Memory error occurred: {str(e)}")
        print("Try reducing the data size or increasing system swap space.")
        return None, None
    except Exception as e:
        print(f"Error during AutoML: {str(e)}")
        return None, None


def optimize_hyperparameters(
    X_train,
    y_train,
    model_type="random_forest",
    n_trials=100,
    timeout=3600,
    metric="accuracy",
    direction="maximize",
    cv=5,
    random_state=42,
):
    """
    Optimize hyperparameters using Optuna.

    Parameters
    ----------
    X_train : numpy.ndarray
        Training features.
    y_train : numpy.ndarray
        Training labels.
    model_type : str, default='random_forest'
        Type of model to optimize. Options: 'logistic_regression', 'random_forest', 'svm', 'neural_network'.
    n_trials : int, default=100
        Number of optimization trials.
    timeout : int, default=3600
        Timeout in seconds.
    metric : str, default='accuracy'
        Metric to optimize.
    direction : str, default='maximize'
        Direction of optimization. Options: 'maximize', 'minimize'.
    cv : int, default=5
        Number of cross-validation folds.
    random_state : int, default=42
        Random state for reproducibility.

    Returns
    -------
    tuple
        (best_params, best_score, study) best parameters, best score, and Optuna study.
    """
    # Convert one-hot encoded labels back to integers if needed
    if len(y_train.shape) > 1:
        y_train = np.argmax(y_train, axis=1)

    # Define the objective function for Optuna
    def objective(trial):
        # Define hyperparameters based on model type
        if model_type == "logistic_regression":
            params = {
                "C": trial.suggest_float("C", 0.001, 10.0, log=True),
                "solver": trial.suggest_categorical("solver", ["liblinear", "saga"]),
                "max_iter": trial.suggest_int("max_iter", 100, 2000),
            }
            model = LogisticRegression(**params, random_state=random_state)

        elif model_type == "random_forest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 10, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 30)
                if trial.suggest_categorical("use_max_depth", [True, False])
                else None,
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            }
            model = RandomForestClassifier(**params, random_state=random_state)

        elif model_type == "svm":
            params = {
                "C": trial.suggest_float("C", 0.1, 100.0, log=True),
                "kernel": trial.suggest_categorical(
                    "kernel", ["linear", "rbf", "poly"]
                ),
                "gamma": trial.suggest_categorical("gamma", ["scale", "auto"])
                if trial.suggest_categorical("use_gamma", [True, False])
                else "scale",
            }
            model = SVC(**params, random_state=random_state, probability=True)

        elif model_type == "neural_network":
            params = {
                "hidden_layer_sizes": (
                    trial.suggest_int("hidden_layer_size_1", 10, 200),
                ),
                "activation": trial.suggest_categorical(
                    "activation", ["relu", "tanh", "logistic"]
                ),
                "solver": trial.suggest_categorical("solver", ["adam", "sgd"]),
                "alpha": trial.suggest_float("alpha", 0.0001, 0.01, log=True),
                "learning_rate_init": trial.suggest_float(
                    "learning_rate_init", 0.0001, 0.1, log=True
                ),
                "max_iter": trial.suggest_int("max_iter", 100, 500),
            }
            model = MLPClassifier(**params, random_state=random_state)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Evaluate the model using cross-validation
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=metric)
        return scores.mean()

    # Create a study object and optimize the objective function
    study = optuna.create_study(
        direction=direction, sampler=optuna.samplers.TPESampler(seed=random_state)
    )

    print(
        f"Starting hyperparameter optimization for {model_type} (n_trials={n_trials}, timeout={timeout}s)..."
    )
    start_time = time.time()

    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    elapsed_time = time.time() - start_time
    print(f"Optimization completed in {elapsed_time:.2f} seconds.")

    # Get the best parameters and score
    best_params = study.best_params
    best_score = study.best_value

    print(f"Best {metric}: {best_score:.4f}")
    print(f"Best parameters: {best_params}")

    return best_params, best_score, study


def save_model(model, filename, output_dir="../models"):
    """
    Save a trained model to disk.

    Parameters
    ----------
    model : object
        Trained model to save.
    filename : str
        Name of the file to save.
    output_dir : str, default='../models'
        Directory to save the model.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    with open(output_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to {output_path}")


def load_saved_model(filename, input_dir="../models"):
    """
    Load a saved model from disk.

    Parameters
    ----------
    filename : str
        Name of the file to load.
    input_dir : str, default='../models'
        Directory containing the model.

    Returns
    -------
    object
        Loaded model.
    """
    input_path = os.path.join(input_dir, filename)

    with open(input_path, "rb") as f:
        model = pickle.load(f)

    print(f"Model loaded from {input_path}")

    return model


if __name__ == "__main__":
    # Example usage
    from src.data.load_data import load_fashion_mnist
    from src.data.preprocess import prepare_data_for_modeling

    # Load configuration
    config = load_config()

    # Load the data
    (x_train, y_train), (x_test, y_test), _ = load_fashion_mnist(save_to_disk=False)

    # Prepare the data
    x_train_prep, y_train_prep, x_test_prep, y_test_prep = prepare_data_for_modeling(
        x_train, y_train, x_test, y_test, normalize=True
    )

    # Flatten the data
    x_train_flat = x_train_prep.reshape(x_train_prep.shape[0], -1)
    x_test_flat = x_test_prep.reshape(x_test_prep.shape[0], -1)

    # Train a simple model
    model = train_model(x_train_flat, y_train_prep, model_type="random_forest")

    # Save the model
    save_model(model, "random_forest_model.pkl")
