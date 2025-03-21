"""
Module for model explainability on the Fashion MNIST dataset.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import shap
import lime
import lime.lime_image
from sklearn.pipeline import Pipeline
import yaml
from skimage.segmentation import mark_boundaries


def load_config(config_path="../configs/feature_config.yaml"):
    """
    Load configuration from YAML file.

    Parameters
    ----------
    config_path : str, default='../configs/feature_config.yaml'
        Path to the configuration file.

    Returns
    -------
    dict
        Configuration dictionary.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def explain_with_shap(
    model, X_background, X_explain, class_names=None, n_samples=100, output_file=None
):
    """
    Explain model predictions using SHAP.

    Parameters
    ----------
    model : object
        Trained model with a predict method.
    X_background : numpy.ndarray
        Background data for the explainer.
    X_explain : numpy.ndarray
        Data to explain.
    class_names : list, default=None
        List of class names.
    n_samples : int, default=100
        Number of samples to use for explanation.
    output_file : str, default=None
        Path to save the SHAP plot.

    Returns
    -------
    object
        SHAP explainer object.
    """
    # Create a small sample of background data
    if len(X_background) > 100:
        background_sample = X_background[:100]
    else:
        background_sample = X_background

    # Create a small sample of data to explain
    if len(X_explain) > n_samples:
        explain_sample = X_explain[:n_samples]
    else:
        explain_sample = X_explain

    # Initialize the SHAP explainer
    if hasattr(model, "predict_proba"):
        explainer = shap.KernelExplainer(model.predict_proba, background_sample)
    else:
        explainer = shap.KernelExplainer(model.predict, background_sample)

    # Calculate SHAP values
    shap_values = explainer.shap_values(explain_sample)

    # Create SHAP summary plot
    plt.figure(figsize=(12, 8))
    if isinstance(shap_values, list):
        # For multi-class models
        if class_names is None:
            class_names = [f"Class {i}" for i in range(len(shap_values))]

        # Plot for a specific class (e.g., the first class)
        shap.summary_plot(
            shap_values[0],
            explain_sample,
            feature_names=[f"pixel_{i}" for i in range(explain_sample.shape[1])],
            show=False,
        )
        plt.title(f"SHAP Feature Importance for {class_names[0]}")
    else:
        # For binary or regression models
        shap.summary_plot(
            shap_values,
            explain_sample,
            feature_names=[f"pixel_{i}" for i in range(explain_sample.shape[1])],
            show=False,
        )
        plt.title("SHAP Feature Importance")

    # Save the plot if an output file is provided
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"SHAP plot saved to {output_file}")

    plt.close()

    return explainer, shap_values


def explain_with_lime(
    model, X_train, X_explain, y_explain, class_names, n_samples, output_file
):
    """
    Generate LIME explanations for image classification predictions.
    """
    from lime import lime_image
    from skimage.segmentation import mark_boundaries
    import os

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Create the LIME image explainer
    explainer = lime_image.LimeImageExplainer()
    explanations = []

    # Function to predict probabilities
    def predict_fn(x):
        # Convert RGB image back to grayscale and flatten
        x_gray = np.mean(x, axis=3)  # Average across RGB channels
        x_flat = x_gray.reshape(
            x_gray.shape[0], 28 * 28
        )  # Flatten to match model input
        return model.predict_proba(x_flat)

    # Number of images to explain
    n_images = min(len(X_explain), 5)  # Limit to 5 images

    # Create figure
    fig, axes = plt.subplots(n_images, 2, figsize=(12, 4 * n_images))
    if n_images == 1:
        axes = axes.reshape(1, -1)

    # Reshape images for explanation
    X_explain_images = X_explain[:n_images].reshape(-1, 28, 28)

    for i, image in enumerate(X_explain_images):
        # Reshape image for LIME (needs to be 3D: height, width, channels)
        image_3d = np.expand_dims(image, axis=2)  # Add channel dimension
        image_3d = np.repeat(image_3d, 3, axis=2)  # Repeat to create RGB

        # Generate explanation
        explanation = explainer.explain_instance(
            image_3d, predict_fn, top_labels=5, hide_color=0, num_samples=n_samples
        )
        explanations.append(explanation)

        # Get the predicted label
        pred_label = model.predict(image.reshape(1, -1))[0]

        # Get the explanation image
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=5,
            hide_rest=True,
        )

        # Plot original image
        axes[i, 0].imshow(image, cmap="gray")
        axes[i, 0].set_title(f"Original (True: {class_names[y_explain[i]]})")
        axes[i, 0].axis("off")

        # Plot explanation
        axes[i, 1].imshow(mark_boundaries(temp, mask))
        axes[i, 1].set_title(f"Explanation (Pred: {class_names[pred_label]})")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    return explanations


def mark_boundaries(image, mask, color=(1, 0, 0), alpha=0.7):
    """
    Mark the boundaries of the mask on the image.

    Parameters
    ----------
    image : numpy.ndarray
        Image to mark.
    mask : numpy.ndarray
        Mask to mark on the image.
    color : tuple, default=(1, 0, 0)
        Color of the boundary.
    alpha : float, default=0.7
        Transparency of the boundary.

    Returns
    -------
    numpy.ndarray
        Image with marked boundaries.
    """
    # Convert grayscale to RGB if needed
    if len(image.shape) == 2 or image.shape[2] == 1:
        image_rgb = np.stack([image] * 3, axis=2)
        if len(image_rgb.shape) == 4:
            image_rgb = image_rgb.squeeze(axis=3)
    else:
        image_rgb = image.copy()

    # Create a colored mask
    colored_mask = np.zeros_like(image_rgb)
    for i in range(3):
        colored_mask[:, :, i] = mask * color[i]

    # Blend the image and the mask
    marked_image = (1 - alpha) * image_rgb + alpha * colored_mask

    return marked_image


if __name__ == "__main__":
    # Example usage
    from src.data.load_data import load_fashion_mnist
    from src.data.preprocess import prepare_data_for_modeling
    from sklearn.ensemble import RandomForestClassifier

    # Load configuration
    config = load_config()

    # Load the data
    (x_train, y_train), (x_test, y_test), class_names = load_fashion_mnist(
        save_to_disk=False
    )

    # Prepare the data
    x_train_prep, y_train_prep, x_test_prep, y_test_prep = prepare_data_for_modeling(
        x_train, y_train, x_test, y_test, normalize=True
    )

    # Flatten the images
    x_train_flat = x_train_prep.reshape(x_train_prep.shape[0], -1)
    x_test_flat = x_test_prep.reshape(x_test_prep.shape[0], -1)

    # Train a simple model
    print("Training a Random Forest classifier...")
    model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    model.fit(x_train_flat, y_train_prep)

    # Explain with SHAP if enabled
    if config["explainability"]["shap"]["apply"]:
        print("Generating SHAP explanations...")
        explainer, shap_values = explain_with_shap(
            model,
            x_train_flat,
            x_test_flat,
            class_names=class_names,
            n_samples=config["explainability"]["shap"]["n_samples"],
            output_file=config["explainability"]["shap"]["output_file"],
        )

    # Explain with LIME if enabled
    if config["explainability"]["lime"]["apply"]:
        print("Generating LIME explanations...")
        explanations = explain_with_lime(
            model,
            x_train_flat,
            x_test_flat,
            y_explain=y_test_prep,
            class_names=class_names,
            n_samples=config["explainability"]["lime"]["n_samples"],
            output_file=config["explainability"]["lime"]["output_file"],
        )
