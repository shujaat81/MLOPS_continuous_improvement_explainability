"""
Module for feature engineering on the Fashion MNIST dataset.
"""
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import yaml


def load_config(config_path='../configs/feature_config.yaml'):
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
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def apply_pca(x_train, x_test, n_components=100, random_state=42):
    """
    Apply PCA dimensionality reduction.
    
    Parameters
    ----------
    x_train : numpy.ndarray
        Training data.
    x_test : numpy.ndarray
        Test data.
    n_components : int, default=100
        Number of principal components to keep.
    random_state : int, default=42
        Random state for reproducibility.
        
    Returns
    -------
    tuple
        (x_train_pca, x_test_pca, pca_model) transformed data and PCA model.
    """
    # Flatten the data if it's not already flattened
    if len(x_train.shape) > 2:
        x_train_flat = x_train.reshape(x_train.shape[0], -1)
        x_test_flat = x_test.reshape(x_test.shape[0], -1)
    else:
        x_train_flat = x_train
        x_test_flat = x_test
    
    # Apply PCA
    pca = PCA(n_components=n_components, random_state=random_state)
    x_train_pca = pca.fit_transform(x_train_flat)
    x_test_pca = pca.transform(x_test_flat)
    
    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_.sum()
    print(f"PCA with {n_components} components explains {explained_variance:.2%} of variance")
    
    return x_train_pca, x_test_pca, pca


def apply_tsne(x_train, x_test=None, n_components=2, perplexity=30, n_iter=1000, random_state=42):
    """
    Apply t-SNE dimensionality reduction.
    
    Parameters
    ----------
    x_train : numpy.ndarray
        Training data.
    x_test : numpy.ndarray, default=None
        Test data. If provided, will be transformed using the same parameters.
    n_components : int, default=2
        Number of dimensions in the embedded space.
    perplexity : float, default=30
        Related to the number of nearest neighbors used in the algorithm.
    n_iter : int, default=1000
        Number of iterations for optimization.
    random_state : int, default=42
        Random state for reproducibility.
        
    Returns
    -------
    tuple
        (x_train_tsne, x_test_tsne) transformed data.
    """
    # Flatten the data if it's not already flattened
    if len(x_train.shape) > 2:
        x_train_flat = x_train.reshape(x_train.shape[0], -1)
        if x_test is not None:
            x_test_flat = x_test.reshape(x_test.shape[0], -1)
    else:
        x_train_flat = x_train
        if x_test is not None:
            x_test_flat = x_test
    
    # Apply t-SNE to training data
    print(f"Applying t-SNE with perplexity={perplexity}, n_iter={n_iter}...")
    tsne = TSNE(n_components=n_components, perplexity=perplexity, 
                n_iter=n_iter, random_state=random_state)
    x_train_tsne = tsne.fit_transform(x_train_flat)
    
    # If test data is provided, transform it as well
    # Note: t-SNE doesn't have a transform method, so we need to fit it again
    if x_test is not None:
        # For visualization purposes, we can use a sample of test data
        sample_size = min(1000, x_test_flat.shape[0])
        x_test_sample = x_test_flat[:sample_size]
        
        # Combine with some training data to maintain structure
        combined_data = np.vstack([x_train_flat[:sample_size], x_test_sample])
        combined_tsne = TSNE(n_components=n_components, perplexity=perplexity, 
                            n_iter=n_iter, random_state=random_state).fit_transform(combined_data)
        
        # Split back into train and test
        x_test_tsne = combined_tsne[sample_size:]
        return x_train_tsne, x_test_tsne
    
    return x_train_tsne, None


def build_autoencoder(input_shape, architecture):
    try:
        # Set mixed precision policy for better performance
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
        # Input layer
        input_layer = Input(shape=input_shape)
        
        # Encoder
        x = input_layer
        for i in range(1, len(architecture) // 2 + 1):
            x = Dense(architecture[i], activation='relu', 
                     kernel_initializer='he_normal')(x)
        
        # Get the bottleneck layer
        bottleneck = x
        
        # Decoder
        for i in range(len(architecture) // 2 + 1, len(architecture)):
            x = Dense(architecture[i], activation='relu', 
                     kernel_initializer='he_normal')(x)
        
        # Output layer
        output_layer = Dense(architecture[0], activation='sigmoid')(x)
        
        # Create models
        autoencoder = Model(input_layer, output_layer)
        encoder = Model(input_layer, bottleneck)
        
        # Compile
        autoencoder.compile(optimizer='adam', loss='mse')
        
        return autoencoder, encoder
        
    except Exception as e:
        print(f"Error building autoencoder: {str(e)}")
        print("Falling back to simpler architecture...")
        
        # Fallback to a simpler architecture
        input_layer = Input(shape=input_shape)
        encoded = Dense(128, activation='relu')(input_layer)
        encoded = Dense(64, activation='relu')(encoded)
        decoded = Dense(128, activation='relu')(encoded)
        decoded = Dense(architecture[0], activation='sigmoid')(decoded)
        
        autoencoder = Model(input_layer, decoded)
        encoder = Model(input_layer, encoded)
        
        autoencoder.compile(optimizer='adam', loss='mse')
        
        return autoencoder, encoder


def train_autoencoder(x_train, x_test, architecture=[784, 128, 64, 128, 784], 
                     epochs=50, batch_size=256, validation_split=0.1):
    """
    Train an autoencoder for feature extraction.
    
    Parameters
    ----------
    x_train : numpy.ndarray
        Training data.
    x_test : numpy.ndarray
        Test data.
    architecture : list, default=[784, 128, 64, 128, 784]
        List of layer sizes for the autoencoder.
    epochs : int, default=50
        Number of epochs for training.
    batch_size : int, default=256
        Batch size for training.
    validation_split : float, default=0.1
        Fraction of training data to use for validation.
        
    Returns
    -------
    tuple
        (autoencoder, encoder, history) trained models and training history.
    """
    # Flatten the data if it's not already flattened
    if len(x_train.shape) > 2:
        x_train_flat = x_train.reshape(x_train.shape[0], -1)
        x_test_flat = x_test.reshape(x_test.shape[0], -1)
    else:
        x_train_flat = x_train
        x_test_flat = x_test
    
    # Normalize the data if it's not already normalized
    if x_train_flat.max() > 1.0:
        x_train_flat = x_train_flat / 255.0
        x_test_flat = x_test_flat / 255.0
    
    # Build the autoencoder
    autoencoder, encoder = build_autoencoder(x_train_flat.shape[1:], architecture)
    
    # Train the autoencoder
    history = autoencoder.fit(
        x_train_flat, x_train_flat,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_split=validation_split,
        verbose=1
    )
    
    return autoencoder, encoder, history


def extract_autoencoder_features(x_train, x_test, encoder):
    """
    Extract features using a trained encoder.
    
    Parameters
    ----------
    x_train : numpy.ndarray
        Training data.
    x_test : numpy.ndarray
        Test data.
    encoder : tensorflow.keras.Model
        Trained encoder model.
        
    Returns
    -------
    tuple
        (x_train_encoded, x_test_encoded) encoded features.
    """
    # Flatten the data if it's not already flattened
    if len(x_train.shape) > 2:
        x_train_flat = x_train.reshape(x_train.shape[0], -1)
        x_test_flat = x_test.reshape(x_test.shape[0], -1)
    else:
        x_train_flat = x_train
        x_test_flat = x_test
    
    # Normalize the data if it's not already normalized
    if x_train_flat.max() > 1.0:
        x_train_flat = x_train_flat / 255.0
        x_test_flat = x_test_flat / 255.0
    
    # Extract features
    x_train_encoded = encoder.predict(x_train_flat)
    x_test_encoded = encoder.predict(x_test_flat)
    
    return x_train_encoded, x_test_encoded


def save_features(features, filename, output_dir='../data/processed'):
    """
    Save extracted features to disk.
    
    Parameters
    ----------
    features : numpy.ndarray
        Extracted features.
    filename : str
        Name of the file to save.
    output_dir : str, default='../data/processed'
        Directory to save the features.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    np.save(output_path, features)
    print(f"Features saved to {output_path}")


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
    
    # Apply PCA if enabled
    if config['feature_extraction']['pca']['apply']:
        x_train_pca, x_test_pca, pca_model = apply_pca(
            x_train_prep, 
            x_test_prep,
            n_components=config['feature_extraction']['pca']['n_components'],
            random_state=config['feature_extraction']['pca']['random_state']
        )
        
        # Save PCA features
        save_features(
            x_train_pca, 
            'x_train_pca.npy', 
            output_dir=os.path.dirname(config['feature_extraction']['pca']['output_file'])
        )
        save_features(
            x_test_pca, 
            'x_test_pca.npy', 
            output_dir=os.path.dirname(config['feature_extraction']['pca']['output_file'])
        )
    
    # Apply t-SNE if enabled
    if config['feature_extraction']['tsne']['apply']:
        x_train_tsne, x_test_tsne = apply_tsne(
            x_train_prep, 
            x_test_prep,
            n_components=config['feature_extraction']['tsne']['n_components'],
            perplexity=config['feature_extraction']['tsne']['perplexity'],
            n_iter=config['feature_extraction']['tsne']['n_iter'],
            random_state=config['feature_extraction']['tsne']['random_state']
        )
        
        # Save t-SNE features
        save_features(
            x_train_tsne, 
            'x_train_tsne.npy', 
            output_dir=os.path.dirname(config['feature_extraction']['tsne']['output_file'])
        )
        if x_test_tsne is not None:
            save_features(
                x_test_tsne, 
                'x_test_tsne.npy', 
                output_dir=os.path.dirname(config['feature_extraction']['tsne']['output_file'])
            )
    
    # Apply autoencoder if enabled
    if config['feature_extraction']['autoencoder']['apply']:
        autoencoder, encoder, history = train_autoencoder(
            x_train_prep, 
            x_test_prep,
            architecture=config['feature_extraction']['autoencoder']['architecture'],
            epochs=config['feature_extraction']['autoencoder']['epochs'],
            batch_size=config['feature_extraction']['autoencoder']['batch_size'],
            validation_split=config['feature_extraction']['autoencoder']['validation_split']
        )
        
        # Extract and save autoencoder features
        x_train_encoded, x_test_encoded = extract_autoencoder_features(
            x_train_prep, x_test_prep, encoder
        )
        
        save_features(
            x_train_encoded, 
            'x_train_autoencoder.npy', 
            output_dir=os.path.dirname(config['feature_extraction']['autoencoder']['output_file'])
        )
        save_features(
            x_test_encoded, 
            'x_test_autoencoder.npy', 
            output_dir=os.path.dirname(config['feature_extraction']['autoencoder']['output_file'])
        )