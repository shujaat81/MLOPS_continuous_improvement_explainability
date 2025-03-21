"""
Module for visualization utilities.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import yaml


def load_config(config_path="../configs/eda_config.yaml"):
    """
    Load configuration from YAML file.

    Parameters
    ----------
    config_path : str, default='../configs/eda_config.yaml'
        Path to the configuration file.

    Returns
    -------
    dict
        Configuration dictionary.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def plot_class_distribution(
    y, class_names=None, title=None, figsize=(10, 6), save_path=None
):
    """
    Plot the distribution of classes in the dataset.

    Parameters
    ----------
    y : numpy.ndarray
        Labels array.
    class_names : list, default=None
        List of class names.
    title : str, default=None
        Plot title.
    figsize : tuple, default=(10, 6)
        Figure size.
    save_path : str, default=None
        Path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.
    """
    plt.figure(figsize=figsize)

    # Count the occurrences of each class
    unique_classes, counts = np.unique(y, return_counts=True)

    # Use class names if provided, otherwise use class indices
    if class_names is None:
        class_names = [str(i) for i in unique_classes]
    else:
        class_names = [class_names[i] for i in unique_classes]

    # Create the bar plot
    ax = sns.barplot(x=unique_classes, y=counts)

    # Add count labels on top of each bar
    for i, count in enumerate(counts):
        ax.text(i, count + 50, str(count), ha="center")

    # Set labels and title
    plt.xlabel("Class")
    plt.ylabel("Count")
    if title:
        plt.title(title)
    else:
        plt.title("Class Distribution")

    # Set x-tick labels to class names
    plt.xticks(unique_classes, class_names, rotation=45, ha="right")

    plt.tight_layout()

    # Save the figure if a path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    return plt.gcf()


def plot_sample_images(
    x, y, class_names=None, samples_per_class=5, figsize=(15, 10), save_path=None
):
    """
    Plot sample images from each class.

    Parameters
    ----------
    x : numpy.ndarray
        Image data with shape (n_samples, height, width).
    y : numpy.ndarray
        Labels array.
    class_names : list, default=None
        List of class names.
    samples_per_class : int, default=5
        Number of samples to show for each class.
    figsize : tuple, default=(15, 10)
        Figure size.
    save_path : str, default=None
        Path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.
    """
    # Get unique classes
    unique_classes = np.unique(y)
    n_classes = len(unique_classes)

    # Create figure
    fig, axes = plt.subplots(n_classes, samples_per_class, figsize=figsize)

    # Use class names if provided, otherwise use class indices
    if class_names is None:
        class_names = [str(i) for i in unique_classes]

    # Plot samples for each class
    for i, class_idx in enumerate(unique_classes):
        # Get indices of samples in this class
        indices = np.where(y == class_idx)[0]

        # Select random samples
        selected_indices = np.random.choice(
            indices, size=min(samples_per_class, len(indices)), replace=False
        )

        # Plot each sample
        for j, idx in enumerate(selected_indices):
            if n_classes == 1:
                ax = axes[j]
            else:
                ax = axes[i, j]

            ax.imshow(x[idx], cmap="gray")
            ax.axis("off")

            if j == 0:
                if n_classes == 1:
                    ax.set_title(f"{class_names[i]}")
                else:
                    ax.set_ylabel(f"{class_names[i]}", rotation=90, size="large")

    plt.tight_layout()
    plt.suptitle("Sample Images from Each Class", y=1.02, fontsize=16)

    # Save the figure if a path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    return fig


def plot_pixel_intensity_distribution(x, figsize=(12, 8), save_path=None):
    """
    Plot the distribution of pixel intensities.

    Parameters
    ----------
    x : numpy.ndarray
        Image data.
    figsize : tuple, default=(12, 8)
        Figure size.
    save_path : str, default=None
        Path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.
    """
    plt.figure(figsize=figsize)

    # Flatten the image data
    x_flat = x.flatten()

    # Plot histogram
    sns.histplot(x_flat, bins=50, kde=True)

    # Set labels and title
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.title("Distribution of Pixel Intensities")

    plt.tight_layout()

    # Save the figure if a path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    return plt.gcf()


def plot_correlation_matrix(
    df, method="spearman", figsize=(14, 12), max_features=100, save_path=None
):
    """
    Plot correlation matrix of features.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the features.
    method : str, default='spearman'
        Correlation method. Options: 'pearson', 'spearman', 'kendall'.
    figsize : tuple, default=(14, 12)
        Figure size.
    max_features : int, default=100
        Maximum number of features to include in the correlation matrix.
    save_path : str, default=None
        Path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])

    # Limit the number of features if needed
    if numeric_df.shape[1] > max_features:
        # Select a subset of features
        numeric_df = numeric_df.iloc[:, :max_features]
        print(f"Limiting correlation matrix to first {max_features} features.")

    # Calculate correlation matrix
    corr = numeric_df.corr(method=method)

    # Create figure
    plt.figure(figsize=figsize)

    # Plot heatmap
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        cmap="coolwarm",
        annot=False,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
    )

    # Set title
    plt.title(f"Feature Correlation Matrix ({method.capitalize()})", fontsize=16)

    plt.tight_layout()

    # Save the figure if a path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    return plt.gcf()


if __name__ == "__main__":
    # Example usage
    from src.data.load_data import load_fashion_mnist

    # Load the data
    (x_train, y_train), (x_test, y_test), class_names = load_fashion_mnist(
        save_to_disk=False
    )

    # Load configuration
    config = load_config()

    # Plot class distribution
    if config["visualizations"]["class_distribution"]["enabled"]:
        plot_class_distribution(
            y_train,
            class_names=class_names,
            title=config["visualizations"]["class_distribution"]["title"],
            figsize=tuple(config["visualizations"]["class_distribution"]["figsize"]),
            save_path=config["visualizations"]["class_distribution"]["output_file"],
        )

    # Plot sample images
    if config["visualizations"]["sample_images"]["enabled"]:
        plot_sample_images(
            x_train,
            y_train,
            class_names=class_names,
            samples_per_class=config["visualizations"]["sample_images"][
                "samples_per_class"
            ],
            figsize=tuple(config["visualizations"]["sample_images"]["figsize"]),
            save_path=config["visualizations"]["sample_images"]["output_file"],
        )

    # Plot pixel intensity distribution
    if config["visualizations"]["pixel_intensity"]["enabled"]:
        plot_pixel_intensity_distribution(
            x_train,
            figsize=tuple(config["visualizations"]["pixel_intensity"]["figsize"]),
            save_path=config["visualizations"]["pixel_intensity"]["output_file"],
        )
