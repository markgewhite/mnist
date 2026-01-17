"""Keras callbacks for GAN training."""

import os
from typing import Optional

import tensorflow as tf
from tensorflow import keras

from .utils import save_image_grid, sample_latent_vectors


class SampleGeneratorCallback(keras.callbacks.Callback):
    """Callback to generate and save sample images during training.

    Generates images from fixed latent vectors at specified intervals
    to visualize training progress.
    """

    def __init__(
        self,
        output_dir: str = "outputs/generated_samples",
        num_samples: int = 25,
        latent_dim: int = 100,
        frequency: int = 100,
        seed: int = 42
    ):
        """Initialize the callback.

        Args:
            output_dir: Directory to save generated images.
            num_samples: Number of images to generate.
            latent_dim: Dimension of latent vectors.
            frequency: Generate samples every N batches.
            seed: Random seed for consistent latent vectors.
        """
        super().__init__()
        self.output_dir = output_dir
        self.num_samples = num_samples
        self.latent_dim = latent_dim
        self.frequency = frequency

        # Create fixed latent vectors for consistent comparison
        self.fixed_latent = sample_latent_vectors(
            num_samples, latent_dim, seed=seed
        )

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        self._batch_count = 0
        self._epoch = 0

    def on_epoch_begin(self, epoch: int, logs: Optional[dict] = None) -> None:
        """Record current epoch."""
        self._epoch = epoch

    def on_train_batch_end(self, batch: int, logs: Optional[dict] = None) -> None:
        """Generate samples at specified frequency.

        Args:
            batch: Current batch index within epoch.
            logs: Dictionary of metric values.
        """
        self._batch_count += 1

        # Generate at specified frequency or first batch
        if self._batch_count == 1 or self._batch_count % self.frequency == 0:
            self._generate_and_save()

    def _generate_and_save(self) -> None:
        """Generate images from fixed latent vectors and save."""
        # Generate images
        generated = self.model.generate_from_latent(self.fixed_latent)

        # Save with epoch and batch info
        filename = f"epoch_{self._epoch + 1:03d}_batch_{self._batch_count:05d}.png"
        filepath = os.path.join(self.output_dir, filename)

        save_image_grid(generated, filepath)

    def on_train_end(self, logs: Optional[dict] = None) -> None:
        """Generate final samples at end of training."""
        self._generate_and_save()


class ProgressCallback(keras.callbacks.Callback):
    """Callback to display training progress."""

    def __init__(self, print_frequency: int = 100):
        """Initialize the callback.

        Args:
            print_frequency: Print progress every N batches.
        """
        super().__init__()
        self.print_frequency = print_frequency
        self._batch_count = 0
        self._epoch = 0

    def on_epoch_begin(self, epoch: int, logs: Optional[dict] = None) -> None:
        """Record current epoch."""
        self._epoch = epoch
        print(f"\nEpoch {epoch + 1}")
        print("-" * 50)

    def on_train_batch_end(self, batch: int, logs: Optional[dict] = None) -> None:
        """Print progress at specified frequency.

        Args:
            batch: Current batch index within epoch.
            logs: Dictionary of metric values.
        """
        self._batch_count += 1

        if self._batch_count % self.print_frequency == 0:
            if logs:
                g_loss = logs.get('g_loss', 0)
                d_loss = logs.get('d_loss', 0)
                g_score = logs.get('g_score', 0)
                d_score = logs.get('d_score', 0)

                print(
                    f"Batch {self._batch_count:5d} | "
                    f"G Loss: {g_loss:.4f} | D Loss: {d_loss:.4f} | "
                    f"G Score: {g_score:.4f} | D Score: {d_score:.4f}"
                )

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        """Print epoch summary."""
        if logs:
            g_loss = logs.get('g_loss', 0)
            d_loss = logs.get('d_loss', 0)
            print(f"\nEpoch {epoch + 1} Summary: G Loss={g_loss:.4f}, D Loss={d_loss:.4f}")
