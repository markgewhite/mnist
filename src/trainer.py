"""Training orchestration for MNIST GAN."""

from typing import List, Optional

import tensorflow as tf
from tensorflow import keras

from .gan import MNISTGAN
from .callbacks import SampleGeneratorCallback, ProgressCallback


class GANTrainer:
    """Orchestrates GAN training with MATLAB-matched hyperparameters.

    Handles:
    - MNIST data loading and preprocessing
    - GAN model creation and compilation
    - Training with callbacks
    """

    # MATLAB-matched default hyperparameters
    DEFAULT_BATCH_SIZE = 60
    DEFAULT_EPOCHS = 500
    DEFAULT_LEARNING_RATE = 0.0002
    DEFAULT_BETA1 = 0.5
    DEFAULT_BETA2 = 0.999
    DEFAULT_FLIP_FACTOR = 0.3
    DEFAULT_LABEL_SMOOTHING = 0.1
    DEFAULT_LATENT_DIM = 100

    def __init__(
        self,
        batch_size: int = DEFAULT_BATCH_SIZE,
        epochs: int = DEFAULT_EPOCHS,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        beta1: float = DEFAULT_BETA1,
        beta2: float = DEFAULT_BETA2,
        flip_factor: float = DEFAULT_FLIP_FACTOR,
        label_smoothing: float = DEFAULT_LABEL_SMOOTHING,
        latent_dim: int = DEFAULT_LATENT_DIM,
        output_dir: str = "outputs/generated_samples"
    ):
        """Initialize the trainer.

        Args:
            batch_size: Training batch size.
            epochs: Number of training epochs.
            learning_rate: Learning rate for Adam optimizers.
            beta1: Beta1 parameter for Adam.
            beta2: Beta2 parameter for Adam.
            flip_factor: Fraction of real labels to flip.
            label_smoothing: One-sided label smoothing (real labels = 1 - this).
            latent_dim: Dimension of latent vectors.
            output_dir: Directory for generated sample outputs.
        """
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.flip_factor = flip_factor
        self.label_smoothing = label_smoothing
        self.latent_dim = latent_dim
        self.output_dir = output_dir

        self.model: Optional[MNISTGAN] = None
        self.train_dataset: Optional[tf.data.Dataset] = None
        self.history = None

    def load_data(self) -> tf.data.Dataset:
        """Load and preprocess MNIST training data.

        Returns:
            TensorFlow Dataset ready for training.
        """
        # Load MNIST
        (train_images, _), (_, _) = keras.datasets.mnist.load_data()

        # Reshape to (N, 28, 28, 1) and normalize to [-1, 1]
        train_images = train_images.reshape(-1, 28, 28, 1).astype("float32")
        train_images = (train_images - 127.5) / 127.5

        # Create dataset with shuffling and batching
        # Match MATLAB: discard partial batches
        dataset = tf.data.Dataset.from_tensor_slices(train_images)
        dataset = dataset.shuffle(buffer_size=60000)
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        self.train_dataset = dataset
        return dataset

    def build_model(self) -> MNISTGAN:
        """Build and compile the GAN model.

        Returns:
            Compiled MNISTGAN model.
        """
        # Create GAN
        self.model = MNISTGAN(
            latent_dim=self.latent_dim,
            flip_factor=self.flip_factor,
            label_smoothing=self.label_smoothing
        )

        # Create optimizers with MATLAB-matched parameters
        g_optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta1,
            beta_2=self.beta2
        )
        d_optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta1,
            beta_2=self.beta2
        )

        # Compile
        self.model.compile(
            g_optimizer=g_optimizer,
            d_optimizer=d_optimizer
        )

        return self.model

    def get_default_callbacks(
        self,
        sample_frequency: int = 100,
        print_frequency: int = 100
    ) -> List[keras.callbacks.Callback]:
        """Get default training callbacks.

        Args:
            sample_frequency: Generate samples every N batches.
            print_frequency: Print progress every N batches.

        Returns:
            List of Keras callbacks.
        """
        return [
            SampleGeneratorCallback(
                output_dir=self.output_dir,
                num_samples=25,
                latent_dim=self.latent_dim,
                frequency=sample_frequency
            ),
            ProgressCallback(print_frequency=print_frequency)
        ]

    def train(
        self,
        callbacks: Optional[List[keras.callbacks.Callback]] = None,
        verbose: int = 0
    ) -> keras.callbacks.History:
        """Train the GAN.

        Args:
            callbacks: List of Keras callbacks. If None, uses defaults.
            verbose: Keras fit verbosity level.

        Returns:
            Training history.
        """
        # Ensure data and model are ready
        if self.train_dataset is None:
            self.load_data()

        if self.model is None:
            self.build_model()

        # Use default callbacks if none provided
        if callbacks is None:
            callbacks = self.get_default_callbacks()

        # Train
        self.history = self.model.fit(
            self.train_dataset,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=verbose
        )

        return self.history

    def generate_samples(
        self,
        num_samples: int = 25,
        seed: Optional[int] = None
    ) -> tf.Tensor:
        """Generate sample images from trained model.

        Args:
            num_samples: Number of images to generate.
            seed: Random seed for reproducibility.

        Returns:
            Generated images tensor.

        Raises:
            ValueError: If model hasn't been built yet.
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        return self.model.generate(num_samples, seed=seed)


def train_gan(
    epochs: int = 500,
    batch_size: int = 60,
    output_dir: str = "outputs/generated_samples"
) -> GANTrainer:
    """Convenience function to train the MNIST GAN.

    Args:
        epochs: Number of training epochs.
        batch_size: Training batch size.
        output_dir: Directory for generated samples.

    Returns:
        Trained GANTrainer instance.
    """
    trainer = GANTrainer(
        epochs=epochs,
        batch_size=batch_size,
        output_dir=output_dir
    )
    trainer.load_data()
    trainer.build_model()
    trainer.train()

    return trainer
