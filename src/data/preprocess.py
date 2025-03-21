"""
Module for preprocessing the Fashion MNIST dataset.
"""
import os
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd


def normalize_images(x_train, x_test, method="standard"):
    """
    Normalize the image data.

    Parameters
    ----------
    x_train : numpy.ndarray
        Training image data.
    x_test : numpy.ndarray
        Test image data.
    method : str, default='standard'
        Normalization method. Options: 'standard', 'minmax', 'simple'.

    Returns
    -------
    tuple
        (x_train_norm, x_test_norm) normalized image data.
    """
    # Reshape to 2D for sklearn preprocessing
    original_shape = x_train.shape
    x_train_2d = x_train.reshape(x_train.shape[0], -1)
    x_test_2d = x_test.reshape(x_test.shape[0], -1)

    if method == "standard":
        # Standardize to mean=0, std=1
        scaler = StandardScaler()
        x_train_2d = scaler.fit_transform(x_train_2d)
        x_test_2d = scaler.transform(x_test_2d)
    elif method == "minmax":
        # Scale to range [0, 1]
        scaler = MinMaxScaler()
        x_train_2d = scaler.fit_transform(x_train_2d)
        x_test_2d = scaler.transform(x_test_2d)
    elif method == "simple":
        # Simple scaling to [0, 1]
        x_train_2d = x_train_2d / 255.0
        x_test_2d = x_test_2d / 255.0
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    # Reshape back to original shape
    x_train_norm = x_train_2d.reshape(original_shape)
    x_test_norm = x_test_2d.reshape(x_test.shape)

    return x_train_norm, x_test_norm


def prepare_data_for_modeling(
    x_train,
    y_train,
    x_test,
    y_test,
    normalize=True,
    reshape_for_cnn=False,
    one_hot_encode=False,
):
    """
    Prepare data for modeling by applying normalization and reshaping.

    Parameters
    ----------
    x_train : numpy.ndarray
        Training image data.
    y_train : numpy.ndarray
        Training labels.
    x_test : numpy.ndarray
        Test image data.
    y_test : numpy.ndarray
        Test labels.
    normalize : bool, default=True
        Whether to normalize the data.
    reshape_for_cnn : bool, default=False
        Whether to reshape the data for CNN (add channel dimension).
    one_hot_encode : bool, default=False
        Whether to one-hot encode the labels.

    Returns
    -------
    tuple
        (x_train_prep, y_train_prep, x_test_prep, y_test_prep) prepared data.
    """
    # Make copies to avoid modifying the original data
    x_train_prep = x_train.copy()
    y_train_prep = y_train.copy()
    x_test_prep = x_test.copy()
    y_test_prep = y_test.copy()

    # Normalize the data
    if normalize:
        x_train_prep, x_test_prep = normalize_images(
            x_train_prep, x_test_prep, method="simple"
        )

    # Reshape for CNN if requested
    if reshape_for_cnn:
        x_train_prep = x_train_prep.reshape(x_train_prep.shape[0], 28, 28, 1)
        x_test_prep = x_test_prep.reshape(x_test_prep.shape[0], 28, 28, 1)

    # One-hot encode the labels if requested
    if one_hot_encode:
        from tensorflow.keras.utils import to_categorical

        y_train_prep = to_categorical(y_train_prep, 10)
        y_test_prep = to_categorical(y_test_prep, 10)

    return x_train_prep, y_train_prep, x_test_prep, y_test_prep


def save_processed_data(x_train, y_train, x_test, y_test, data_dir="../data/processed"):
    """
    Save processed data to disk.

    Parameters
    ----------
    x_train : numpy.ndarray
        Processed training image data.
    y_train : numpy.ndarray
        Training labels.
    x_test : numpy.ndarray
        Processed test image data.
    y_test : numpy.ndarray
        Test labels.
    data_dir : str, default='../data/processed'
        Directory to save the processed data.
    """
    os.makedirs(data_dir, exist_ok=True)

    # Save as numpy arrays
    np.save(os.path.join(data_dir, "x_train_processed.npy"), x_train)
    np.save(os.path.join(data_dir, "y_train.npy"), y_train)
    np.save(os.path.join(data_dir, "x_test_processed.npy"), x_test)
    np.save(os.path.join(data_dir, "y_test.npy"), y_test)

    print(f"Processed data saved to {data_dir}")


if __name__ == "__main__":
    # Example usage
    from load_data import load_fashion_mnist

    # Load the data
    (x_train, y_train), (x_test, y_test), _ = load_fashion_mnist(save_to_disk=False)

    # Prepare the data for modeling
    x_train_prep, y_train_prep, x_test_prep, y_test_prep = prepare_data_for_modeling(
        x_train,
        y_train,
        x_test,
        y_test,
        normalize=True,
        reshape_for_cnn=True,
        one_hot_encode=True,
    )

    # Save the processed data
    save_processed_data(x_train_prep, y_train_prep, x_test_prep, y_test_prep)

    print(f"Original training data shape: {x_train.shape}")
    print(f"Processed training data shape: {x_train_prep.shape}")
    print(f"Original training labels shape: {y_train.shape}")
    print(f"Processed training labels shape: {y_train_prep.shape}")
