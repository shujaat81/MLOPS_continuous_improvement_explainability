"""
Module for making predictions with trained models on the Fashion MNIST dataset.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from src.models.train_model import load_saved_model


def predict_single_image(model, image, class_names=None):
    """
    Make a prediction for a single image.

    Parameters
    ----------
    model : object
        Trained model with predict and predict_proba methods.
    image : numpy.ndarray
        Image to predict, shape (28, 28) or (784,).
    class_names : list, default=None
        List of class names.

    Returns
    -------
    tuple
        (predicted_class, predicted_proba) predicted class and probability.
    """
    # Reshape image if needed
    if image.shape == (28, 28):
        image_flat = image.reshape(1, -1)
    elif image.shape == (28, 28, 1):
        image_flat = image.reshape(1, -1)
    elif len(image.shape) == 1 and image.shape[0] == 784:
        image_flat = image.reshape(1, -1)
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")

    # Make prediction
    predicted_class = model.predict(image_flat)[0]
    predicted_proba = model.predict_proba(image_flat)[0]

    # Get class name if provided
    if class_names is not None:
        class_name = class_names[predicted_class]
    else:
        class_name = f"Class {predicted_class}"

    # Get probability of predicted class
    predicted_prob = predicted_proba[predicted_class]

    return predicted_class, class_name, predicted_prob, predicted_proba


def predict_batch(model, images, class_names=None):
    """
    Make predictions for a batch of images.

    Parameters
    ----------
    model : object
        Trained model with predict and predict_proba methods.
    images : numpy.ndarray
        Images to predict, shape (n_samples, 28, 28) or (n_samples, 784).
    class_names : list, default=None
        List of class names.

    Returns
    -------
    tuple
        (predicted_classes, predicted_probas) predicted classes and probabilities.
    """
    # Reshape images if needed
    if len(images.shape) == 3 and images.shape[1:] == (28, 28):
        images_flat = images.reshape(images.shape[0], -1)
    elif len(images.shape) == 4 and images.shape[1:3] == (28, 28):
        images_flat = images.reshape(images.shape[0], -1)
    elif len(images.shape) == 2 and images.shape[1] == 784:
        images_flat = images
    else:
        raise ValueError(f"Unexpected images shape: {images.shape}")

    # Make predictions
    predicted_classes = model.predict(images_flat)
    predicted_probas = model.predict_proba(images_flat)

    # Get class names if provided
    if class_names is not None:
        class_names_pred = [class_names[i] for i in predicted_classes]
    else:
        class_names_pred = [f"Class {i}" for i in predicted_classes]

    return predicted_classes, class_names_pred, predicted_probas


def visualize_prediction(
    image,
    predicted_class,
    predicted_proba,
    class_names=None,
    figsize=(8, 6),
    output_file=None,
):
    """
    Visualize an image and its prediction.

    Parameters
    ----------
    image : numpy.ndarray
        Image to visualize, shape (28, 28) or (784,).
    predicted_class : int
        Predicted class.
    predicted_proba : numpy.ndarray
        Predicted probabilities for each class.
    class_names : list, default=None
        List of class names.
    figsize : tuple, default=(8, 6)
        Figure size.
    output_file : str, default=None
        Path to save the figure.
    """
    # Reshape image if needed
    if len(image.shape) == 1 and image.shape[0] == 784:
        image = image.reshape(28, 28)

    # Create class names if not provided
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(predicted_proba))]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1, 1.5]}
    )

    # Plot image
    ax1.imshow(image, cmap="gray")
    ax1.set_title(f"Prediction: {class_names[predicted_class]}")
    ax1.axis("off")

    # Plot prediction probabilities
    sorted_indices = np.argsort(predicted_proba)[::-1]
    sorted_probs = predicted_proba[sorted_indices]
    sorted_names = [class_names[i] for i in sorted_indices]

    # Plot top 5 probabilities
    top_n = min(5, len(sorted_names))
    ax2.barh(range(top_n), sorted_probs[:top_n], color="skyblue")
    ax2.set_yticks(range(top_n))
    ax2.set_yticklabels(sorted_names[:top_n])
    ax2.set_xlabel("Probability")
    ax2.set_title("Top Predictions")
    ax2.set_xlim(0, 1)
    ax2.grid(axis="x")

    # Add probability values to the bars
    for i, prob in enumerate(sorted_probs[:top_n]):
        ax2.text(prob + 0.02, i, f"{prob:.4f}", va="center")

    plt.tight_layout()

    # Save the figure if an output file is provided
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Prediction visualization saved to {output_file}")

    plt.show()


def visualize_batch_predictions(
    images,
    predicted_classes,
    predicted_probas,
    class_names=None,
    n_samples=5,
    figsize=(15, 10),
    output_file=None,
):
    """
    Visualize predictions for a batch of images.

    Parameters
    ----------
    images : numpy.ndarray
        Images to visualize, shape (n_samples, 28, 28) or (n_samples, 784).
    predicted_classes : numpy.ndarray
        Predicted classes.
    predicted_probas : numpy.ndarray
        Predicted probabilities.
    class_names : list, default=None
        List of class names.
    n_samples : int, default=5
        Number of samples to visualize.
    figsize : tuple, default=(15, 10)
        Figure size.
    output_file : str, default=None
        Path to save the figure.
    """
    # Reshape images if needed
    if len(images.shape) == 2 and images.shape[1] == 784:
        images = images.reshape(-1, 28, 28)

    # Create class names if not provided
    if class_names is None:
        class_names = [f"Class {i}" for i in range(predicted_probas.shape[1])]

    # Limit the number of samples to visualize
    n_samples = min(n_samples, len(images))

    # Create figure with subplots
    fig, axes = plt.subplots(
        n_samples, 2, figsize=figsize, gridspec_kw={"width_ratios": [1, 2]}
    )

    for i in range(n_samples):
        # Get image and prediction
        image = images[i]
        predicted_class = predicted_classes[i]
        predicted_proba = predicted_probas[i]

        # Plot image
        axes[i, 0].imshow(image, cmap="gray")
        axes[i, 0].set_title(f"Prediction: {class_names[predicted_class]}")
        axes[i, 0].axis("off")

        # Plot prediction probabilities
        sorted_indices = np.argsort(predicted_proba)[::-1]
        sorted_probs = predicted_proba[sorted_indices]
        sorted_names = [class_names[i] for i in sorted_indices]

        # Plot top 5 probabilities
        top_n = min(5, len(sorted_names))
        axes[i, 1].barh(range(top_n), sorted_probs[:top_n], color="skyblue")
        axes[i, 1].set_yticks(range(top_n))
        axes[i, 1].set_yticklabels(sorted_names[:top_n])
        axes[i, 1].set_xlim(0, 1)
        axes[i, 1].grid(axis="x")

        # Add probability values to the bars
        for j, prob in enumerate(sorted_probs[:top_n]):
            axes[i, 1].text(prob + 0.02, j, f"{prob:.2f}", va="center")

    plt.tight_layout()

    # Save the figure if an output file is provided
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Batch prediction visualization saved to {output_file}")

    plt.show()


if __name__ == "__main__":
    # Example usage
    from src.data.load_data import load_fashion_mnist
    from src.data.preprocess import prepare_data_for_modeling

    # Load the data
    (x_train, y_train), (x_test, y_test), class_names = load_fashion_mnist(
        save_to_disk=False
    )

    # Prepare the data
    x_train_prep, y_train_prep, x_test_prep, y_test_prep = prepare_data_for_modeling(
        x_train, y_train, x_test, y_test, normalize=True
    )

    # Try to load a saved model
    try:
        model = load_saved_model("random_forest_model.pkl")

        # Make a prediction for a single image
        sample_idx = 42  # Choose a sample index
        sample_image = x_test_prep[sample_idx]
        true_class = y_test[sample_idx]

        (
            predicted_class,
            class_name,
            predicted_prob,
            predicted_proba,
        ) = predict_single_image(model, sample_image, class_names)

        print(f"True class: {class_names[true_class]}")
        print(f"Predicted class: {class_name}")
        print(f"Prediction probability: {predicted_prob:.4f}")

        # Visualize the prediction
        visualize_prediction(
            sample_image,
            predicted_class,
            predicted_proba,
            class_names,
            output_file="../reports/figures/sample_prediction.png",
        )

        # Make predictions for a batch of images
        batch_size = 5
        batch_images = x_test_prep[:batch_size]
        batch_true_classes = y_test[:batch_size]

        predicted_classes, class_names_pred, predicted_probas = predict_batch(
            model, batch_images, class_names
        )

        # Visualize batch predictions
        visualize_batch_predictions(
            batch_images,
            predicted_classes,
            predicted_probas,
            class_names,
            output_file="../reports/figures/batch_predictions.png",
        )

    except FileNotFoundError:
        print("No saved model found. Train a model first using train_model.py.")
