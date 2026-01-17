"""MNIST GAN TensorFlow/Keras implementation.

A DCGAN that generates handwritten digit images, ported from MATLAB.
"""

from .models import Generator, Discriminator
from .gan import MNISTGAN
from .trainer import GANTrainer, train_gan
from .callbacks import SampleGeneratorCallback, ProgressCallback
from .utils import (
    get_device_info,
    print_device_info,
    sample_latent_vectors,
    save_image_grid,
    display_image_grid,
    normalize_images,
    denormalize_images
)

__all__ = [
    # Models
    "Generator",
    "Discriminator",
    "MNISTGAN",
    # Training
    "GANTrainer",
    "train_gan",
    # Callbacks
    "SampleGeneratorCallback",
    "ProgressCallback",
    # Utilities
    "get_device_info",
    "print_device_info",
    "sample_latent_vectors",
    "save_image_grid",
    "display_image_grid",
    "normalize_images",
    "denormalize_images",
]

__version__ = "1.0.0"
