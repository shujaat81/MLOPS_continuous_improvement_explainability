"""
Module for evaluating models on the Fashion MNIST dataset.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)


def evaluate_model(model, X_test, y_test, class_names=None):
    """
    Evaluate a model on test data.

    Parameters
    ----------
    model : object
        Trained model with predict and predict_proba methods.
    X_test : numpy.ndarray
        Test features.
    y_test : numpy.ndarray
        Test labels.
    class_names : list, default=None
        List of class names.

    Returns
    -------
    dict
        Dictionary of evaluation metrics.
    """
    # Convert one-hot encoded labels back to integers if needed
    if len(y_test.shape) > 1:
        y_test = np.argmax(y_test, axis=1)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    # Create metrics dictionary
    metrics = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Generate classification report
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(np.unique(y_test)))]

    report = classification_report(y_test, y_pred, target_names=class_names)
    print(f"\nClassification Report:\n{report}")

    return metrics


def plot_confusion_matrix(
    model, X_test, y_test, class_names=None, figsize=(10, 8), output_file=None
):
    """
    Plot confusion matrix for model evaluation.

    Parameters
    ----------
    model : object
        Trained model with predict method.
    X_test : numpy.ndarray
        Test features.
    y_test : numpy.ndarray
        Test labels.
    class_names : list, default=None
        List of class names.
    figsize : tuple, default=(10, 8)
        Figure size.
    output_file : str, default=None
        Path to save the figure.

    Returns
    -------
    numpy.ndarray
        Confusion matrix.
    """
    # Convert one-hot encoded labels back to integers if needed
    if len(y_test.shape) > 1:
        y_test = np.argmax(y_test, axis=1)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Create class names if not provided
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(np.unique(y_test)))]

    # Plot confusion matrix
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    # Save the figure if an output file is provided
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Confusion matrix saved to {output_file}")

    plt.show()

    return cm


def plot_roc_curve(
    model, X_test, y_test, class_names=None, figsize=(10, 8), output_file=None
):
    """
    Plot ROC curve for model evaluation.

    Parameters
    ----------
    model : object
        Trained model with predict_proba method.
    X_test : numpy.ndarray
        Test features.
    y_test : numpy.ndarray
        Test labels.
    class_names : list, default=None
        List of class names.
    figsize : tuple, default=(10, 8)
        Figure size.
    output_file : str, default=None
        Path to save the figure.

    Returns
    -------
    dict
        Dictionary of ROC AUC scores for each class.
    """
    # Convert one-hot encoded labels to one-hot if needed
    if len(y_test.shape) == 1:
        n_classes = len(np.unique(y_test))
        y_test_onehot = np.zeros((len(y_test), n_classes))
        for i in range(n_classes):
            y_test_onehot[:, i] = y_test == i
    else:
        y_test_onehot = y_test
        n_classes = y_test.shape[1]

    # Create class names if not provided
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]

    # Get predicted probabilities
    y_score = model.predict_proba(X_test)

    # Compute ROC curve and ROC area for each class
    fpr = {}
    tpr = {}
    roc_auc = {}

    plt.figure(figsize=figsize)

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_onehot[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        plt.plot(
            fpr[i], tpr[i], lw=2, label=f"{class_names[i]} (AUC = {roc_auc[i]:.2f})"
        )

    # Plot diagonal line
    plt.plot([0, 1], [0, 1], "k--", lw=2)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()

    # Save the figure if an output file is provided
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"ROC curve saved to {output_file}")

    plt.show()

    # Calculate macro-average ROC AUC
    macro_roc_auc = np.mean(list(roc_auc.values()))
    print(f"Macro-average ROC AUC: {macro_roc_auc:.4f}")

    return roc_auc


def plot_feature_importance(
    model, feature_names=None, top_n=20, figsize=(10, 8), output_file=None
):
    """
    Plot feature importance for tree-based models.

    Parameters
    ----------
    model : object
        Trained model with feature_importances_ attribute.
    feature_names : list, default=None
        List of feature names.
    top_n : int, default=20
        Number of top features to show.
    figsize : tuple, default=(10, 8)
        Figure size.
    output_file : str, default=None
        Path to save the figure.

    Returns
    -------
    pandas.DataFrame
        DataFrame of feature importances.
    """
    # Check if model has feature_importances_ attribute
    if not hasattr(model, "feature_importances_"):
        print(
            "Model does not have feature_importances_ attribute. Skipping feature importance plot."
        )
        return None

    # Get feature importances
    importances = model.feature_importances_

    # Create feature names if not provided
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(importances))]

    # Create DataFrame of feature importances
    feature_importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    )

    # Sort by importance
    feature_importance_df = feature_importance_df.sort_values(
        "Importance", ascending=False
    )

    # Select top N features
    if top_n is not None and top_n < len(feature_importance_df):
        feature_importance_df = feature_importance_df.head(top_n)

    # Plot feature importances
    plt.figure(figsize=figsize)
    sns.barplot(x="Importance", y="Feature", data=feature_importance_df)
    plt.title("Feature Importance")
    plt.tight_layout()

    # Save the figure if an output file is provided
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Feature importance plot saved to {output_file}")

    plt.show()

    return feature_importance_df


if __name__ == "__main__":
    # Example usage
    from src.data.load_data import load_fashion_mnist
    from src.data.preprocess import prepare_data_for_modeling
    from src.models.train_model import train_model, load_saved_model

    # Load the data
    (x_train, y_train), (x_test, y_test), class_names = load_fashion_mnist(
        save_to_disk=False
    )

    # Prepare the data
    x_train_prep, y_train_prep, x_test_prep, y_test_prep = prepare_data_for_modeling(
        x_train, y_train, x_test, y_test, normalize=True
    )

    # Flatten the data
    x_train_flat = x_train_prep.reshape(x_train_prep.shape[0], -1)
    x_test_flat = x_test_prep.reshape(x_test_prep.shape[0], -1)

    # Try to load a saved model, or train a new one if not found
    try:
        model = load_saved_model("random_forest_model.pkl")
    except FileNotFoundError:
        print("No saved model found. Training a new model...")
        model = train_model(x_train_flat, y_train_prep, model_type="random_forest")

    # Evaluate the model
    metrics = evaluate_model(model, x_test_flat, y_test_prep, class_names)

    # Plot confusion matrix
    plot_confusion_matrix(
        model,
        x_test_flat,
        y_test_prep,
        class_names,
        output_file="../reports/figures/confusion_matrix.png",
    )

    # Plot ROC curve
    plot_roc_curve(
        model,
        x_test_flat,
        y_test_prep,
        class_names,
        output_file="../reports/figures/roc_curve.png",
    )

    # Plot feature importance
    feature_names = [f"pixel_{i}" for i in range(x_train_flat.shape[1])]
    plot_feature_importance(
        model,
        feature_names,
        top_n=20,
        output_file="../reports/figures/feature_importance.png",
    )
