"""
Module for loading the Fashion MNIST dataset.
"""
import os
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
import pandas as pd


def load_fashion_mnist(save_to_disk=True, data_dir="../data/raw"):
    """
    Load the Fashion MNIST dataset.

    Parameters
    ----------
    save_to_disk : bool, default=True
        Whether to save the dataset to disk.
    data_dir : str, default='../data/raw'
        Directory to save the dataset.

    Returns
    -------
    tuple
        (x_train, y_train), (x_test, y_test) where x is the image data and y is the labels.
    """
    # Load the Fashion MNIST dataset
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # Create class names
    class_names = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    if save_to_disk:
        os.makedirs(data_dir, exist_ok=True)

        # Save as numpy arrays
        np.save(os.path.join(data_dir, "x_train.npy"), x_train)
        np.save(os.path.join(data_dir, "y_train.npy"), y_train)
        np.save(os.path.join(data_dir, "x_test.npy"), x_test)
        np.save(os.path.join(data_dir, "y_test.npy"), y_test)

        # Save class names
        with open(os.path.join(data_dir, "class_names.txt"), "w") as f:
            for name in class_names:
                f.write(f"{name}\n")

        print(f"Dataset saved to {data_dir}")

    return (x_train, y_train), (x_test, y_test), class_names


def load_as_dataframe(flatten=True):
    """
    Load the Fashion MNIST dataset as pandas DataFrames.

    Parameters
    ----------
    flatten : bool, default=True
        Whether to flatten the images into 1D arrays.

    Returns
    -------
    tuple
        (train_df, test_df) where each is a pandas DataFrame.
    """
    # Load the dataset
    (x_train, y_train), (x_test, y_test), class_names = load_fashion_mnist(
        save_to_disk=False
    )

    # Flatten the images if requested
    if flatten:
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)

    # Create column names
    if flatten:
        feature_cols = [f"pixel_{i}" for i in range(x_train.shape[1])]
    else:
        feature_cols = [f"pixel_{i}_{j}" for i in range(28) for j in range(28)]

    # Create DataFrames
    train_df = pd.DataFrame(x_train, columns=feature_cols)
    train_df["label"] = y_train
    train_df["label_name"] = [class_names[y] for y in y_train]

    test_df = pd.DataFrame(x_test, columns=feature_cols)
    test_df["label"] = y_test
    test_df["label_name"] = [class_names[y] for y in y_test]

    return train_df, test_df


if __name__ == "__main__":
    # Example usage
    (x_train, y_train), (x_test, y_test), class_names = load_fashion_mnist()
    print(f"Training data shape: {x_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    print(f"Class names: {class_names}")
