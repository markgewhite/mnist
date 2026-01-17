"""Utility functions for MNIST GAN."""

import os
import platform
from typing import Optional

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def get_device_info() -> dict:
    """Detect available compute devices and return info.

    Returns:
        Dictionary with device information.
    """
    devices = tf.config.list_physical_devices()
    gpus = tf.config.list_physical_devices('GPU')

    info = {
        'platform': platform.system(),
        'machine': platform.machine(),
        'tensorflow_version': tf.__version__,
        'devices': [d.name for d in devices],
        'gpu_available': len(gpus) > 0,
        'gpu_count': len(gpus),
    }

    # Detect GPU type
    if info['gpu_available']:
        if info['platform'] == 'Darwin' and info['machine'] == 'arm64':
            info['gpu_type'] = 'Apple Metal'
        else:
            info['gpu_type'] = 'CUDA'
    else:
        info['gpu_type'] = None

    return info


def print_device_info() -> None:
    """Print available compute devices."""
    info = get_device_info()

    print(f"Platform: {info['platform']} ({info['machine']})")
    print(f"TensorFlow: {info['tensorflow_version']}")

    if info['gpu_available']:
        print(f"GPU: {info['gpu_type']} ({info['gpu_count']} device(s))")
        print("Training will use GPU acceleration")
    else:
        print("GPU: Not available")
        print("Training will use CPU")

        # Platform-specific hints
        if info['platform'] == 'Darwin' and info['machine'] == 'arm64':
            print("  Hint: Install tensorflow-metal for GPU support")
        elif info['platform'] == 'Windows':
            print("  Hint: Install CUDA toolkit for GPU support")


def sample_latent_vectors(
    num_samples: int,
    latent_dim: int = 100,
    seed: Optional[int] = None
) -> tf.Tensor:
    """Sample random latent vectors from standard normal distribution.

    Args:
        num_samples: Number of latent vectors to sample.
        latent_dim: Dimension of each latent vector.
        seed: Random seed for reproducibility.

    Returns:
        Tensor of shape (num_samples, latent_dim).
    """
    if seed is not None:
        tf.random.set_seed(seed)

    return tf.random.normal(shape=(num_samples, latent_dim))


def save_image_grid(
    images: tf.Tensor,
    filepath: str,
    grid_size: Optional[tuple] = None,
    figsize: tuple = (10, 10)
) -> None:
    """Save a grid of generated images to file.

    Args:
        images: Tensor of shape (N, H, W, 1) with values in [-1, 1].
        filepath: Path to save the image.
        grid_size: Tuple of (rows, cols). If None, uses square grid.
        figsize: Figure size in inches.
    """
    # Convert to numpy and rescale from [-1, 1] to [0, 1]
    images = images.numpy() if hasattr(images, 'numpy') else images
    images = (images + 1) / 2.0
    images = np.clip(images, 0, 1)

    num_images = images.shape[0]

    # Determine grid size
    if grid_size is None:
        cols = int(np.ceil(np.sqrt(num_images)))
        rows = int(np.ceil(num_images / cols))
    else:
        rows, cols = grid_size

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).flatten()

    for i, ax in enumerate(axes):
        ax.axis('off')
        if i < num_images:
            ax.imshow(images[i, :, :, 0], cmap='gray', vmin=0, vmax=1)

    plt.tight_layout()

    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    plt.savefig(filepath, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


def display_image_grid(
    images: tf.Tensor,
    grid_size: Optional[tuple] = None,
    figsize: tuple = (10, 10),
    title: Optional[str] = None
) -> plt.Figure:
    """Display a grid of generated images.

    Args:
        images: Tensor of shape (N, H, W, 1) with values in [-1, 1].
        grid_size: Tuple of (rows, cols). If None, uses square grid.
        figsize: Figure size in inches.
        title: Optional title for the figure.

    Returns:
        Matplotlib figure.
    """
    # Convert to numpy and rescale from [-1, 1] to [0, 1]
    images = images.numpy() if hasattr(images, 'numpy') else images
    images = (images + 1) / 2.0
    images = np.clip(images, 0, 1)

    num_images = images.shape[0]

    # Determine grid size
    if grid_size is None:
        cols = int(np.ceil(np.sqrt(num_images)))
        rows = int(np.ceil(num_images / cols))
    else:
        rows, cols = grid_size

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).flatten()

    for i, ax in enumerate(axes):
        ax.axis('off')
        if i < num_images:
            ax.imshow(images[i, :, :, 0], cmap='gray', vmin=0, vmax=1)

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    return fig


def normalize_images(images: np.ndarray) -> np.ndarray:
    """Normalize images from [0, 255] to [-1, 1] range.

    Args:
        images: Array of images with values in [0, 255].

    Returns:
        Normalized images in [-1, 1] range.
    """
    return (images.astype(np.float32) / 127.5) - 1.0


def denormalize_images(images: np.ndarray) -> np.ndarray:
    """Denormalize images from [-1, 1] to [0, 255] range.

    Args:
        images: Array of images with values in [-1, 1].

    Returns:
        Denormalized images in [0, 255] range as uint8.
    """
    return ((images + 1.0) * 127.5).astype(np.uint8)
