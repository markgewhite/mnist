"""MNIST GAN composite model with custom training step.

Combines Generator and Discriminator with custom train_step implementing
the training logic from the MATLAB modelGradients.m:
- Binary cross-entropy loss (from_logits=True)
- Label flipping for regularization
- Separate optimizer updates for G and D
"""

import tensorflow as tf
from tensorflow import keras

from .models import Generator, Discriminator


class MNISTGAN(keras.Model):
    """GAN model combining Generator and Discriminator with custom training.

    Implements the MATLAB training logic:
    1. Generate fake images from latent vectors
    2. Compute discriminator predictions on real and fake images
    3. Apply label flipping to real labels for regularization
    4. Compute BCE losses for both networks
    5. Update networks with separate optimizers
    """

    def __init__(
        self,
        latent_dim: int = 100,
        flip_factor: float = 0.3,
        label_smoothing: float = 0.1,
        name: str = "MNISTGAN"
    ):
        """Initialize the MNIST GAN.

        Args:
            latent_dim: Dimension of latent input vector for generator.
            flip_factor: Fraction of real labels to flip (regularization).
            label_smoothing: Smooth real labels from 1.0 to (1.0 - smoothing).
            name: Name identifier for the model.
        """
        super().__init__(name=name)
        self.latent_dim = latent_dim
        self.flip_factor = flip_factor
        self.label_smoothing = label_smoothing

        # Build component networks
        self.generator = Generator(latent_dim=latent_dim)
        self.discriminator = Discriminator()

        # Loss function - BCE from logits for numerical stability
        self.bce_loss = keras.losses.BinaryCrossentropy(from_logits=True)

        # Metrics
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_score_metric = keras.metrics.Mean(name="g_score")
        self.d_score_metric = keras.metrics.Mean(name="d_score")

    def compile(
        self,
        g_optimizer: keras.optimizers.Optimizer = None,
        d_optimizer: keras.optimizers.Optimizer = None,
        **kwargs
    ):
        """Compile the GAN with separate optimizers for G and D.

        Args:
            g_optimizer: Optimizer for generator. Defaults to Adam(0.0002, 0.5).
            d_optimizer: Optimizer for discriminator. Defaults to Adam(0.0002, 0.5).
            **kwargs: Additional arguments passed to parent compile.
        """
        super().compile(**kwargs)

        # Default MATLAB-matched optimizers: Adam(lr=0.0002, beta1=0.5, beta2=0.999)
        if g_optimizer is None:
            g_optimizer = keras.optimizers.Adam(
                learning_rate=0.0002,
                beta_1=0.5,
                beta_2=0.999
            )
        if d_optimizer is None:
            d_optimizer = keras.optimizers.Adam(
                learning_rate=0.0002,
                beta_1=0.5,
                beta_2=0.999
            )

        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

    @property
    def metrics(self):
        """Return list of metrics to track."""
        return [
            self.g_loss_metric,
            self.d_loss_metric,
            self.g_score_metric,
            self.d_score_metric
        ]

    def _flip_labels(self, labels: tf.Tensor, batch_size: int) -> tf.Tensor:
        """Flip a fraction of labels for regularization.

        Matches MATLAB logic: randomly select flip_factor fraction of samples
        and flip their labels (1->0 for real images).

        Args:
            labels: Original labels tensor.
            batch_size: Number of samples in batch.

        Returns:
            Labels with some flipped.
        """
        # Number of labels to flip
        num_flip = tf.cast(
            tf.cast(batch_size, tf.float32) * self.flip_factor,
            tf.int32
        )

        # Random indices to flip
        indices = tf.random.shuffle(tf.range(batch_size))[:num_flip]

        # Create flip mask
        flip_mask = tf.scatter_nd(
            indices=tf.expand_dims(indices, 1),
            updates=tf.ones(num_flip),
            shape=[batch_size]
        )
        flip_mask = tf.expand_dims(flip_mask, 1)

        # Flip: where mask is 1, labels become (1 - labels)
        flipped_labels = labels * (1 - flip_mask) + (1 - labels) * flip_mask

        return flipped_labels

    def train_step(self, real_images: tf.Tensor) -> dict:
        """Custom training step implementing MATLAB GAN training logic.

        Matches MATLAB modelGradients.m: both G and D gradients are computed
        from the SAME fake images in a single forward pass, then both networks
        are updated.

        Args:
            real_images: Batch of real images from dataset.

        Returns:
            Dictionary of metric values.
        """
        batch_size = tf.shape(real_images)[0]

        # Sample random latent vectors (same z used for both G and D training)
        z = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Labels for real and fake images
        # One-sided label smoothing: real labels are (1 - smoothing) instead of 1.0
        real_labels = tf.ones((batch_size, 1)) * (1.0 - self.label_smoothing)
        fake_labels = tf.zeros((batch_size, 1))

        # For generator loss, use unsmoothed labels (we want D to output 1.0)
        real_labels_unsmoothed = tf.ones((batch_size, 1))

        # Apply label flipping to real labels (regularization)
        flipped_real_labels = self._flip_labels(real_labels, batch_size)

        # === Compute both gradients from same fake images (matches MATLAB) ===
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            # Generate fake images ONCE
            fake_images = self.generator(z, training=True)

            # Get discriminator predictions on real and fake
            real_logits = self.discriminator(real_images, training=True)
            fake_logits = self.discriminator(fake_images, training=True)

            # Discriminator loss (with flipped real labels)
            d_loss_real = self.bce_loss(flipped_real_labels, real_logits)
            d_loss_fake = self.bce_loss(fake_labels, fake_logits)
            d_loss = d_loss_real + d_loss_fake

            # Generator loss: wants discriminator to think fakes are real
            g_loss = self.bce_loss(real_labels_unsmoothed, fake_logits)

        # Compute gradients
        d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)

        # Update both networks (matches MATLAB order: D first, then G)
        self.d_optimizer.apply_gradients(
            zip(d_grads, self.discriminator.trainable_variables)
        )
        self.g_optimizer.apply_gradients(
            zip(g_grads, self.generator.trainable_variables)
        )

        # === Compute scores (probabilities) ===
        prob_real = tf.sigmoid(real_logits)
        prob_fake = tf.sigmoid(fake_logits)

        # Discriminator score: average of (prob_real + (1 - prob_fake)) / 2
        d_score = (tf.reduce_mean(prob_real) + tf.reduce_mean(1 - prob_fake)) / 2

        # Generator score: average prob_fake (how well it fools discriminator)
        g_score = tf.reduce_mean(prob_fake)

        # Update metrics
        self.g_loss_metric.update_state(g_loss)
        self.d_loss_metric.update_state(d_loss)
        self.g_score_metric.update_state(g_score)
        self.d_score_metric.update_state(d_score)

        return {
            "g_loss": self.g_loss_metric.result(),
            "d_loss": self.d_loss_metric.result(),
            "g_score": self.g_score_metric.result(),
            "d_score": self.d_score_metric.result()
        }

    def generate(self, num_images: int = 25, seed: int = None) -> tf.Tensor:
        """Generate images from random latent vectors.

        Args:
            num_images: Number of images to generate.
            seed: Random seed for reproducibility.

        Returns:
            Generated images of shape (num_images, 28, 28, 1) in range [-1, 1].
        """
        if seed is not None:
            tf.random.set_seed(seed)

        z = tf.random.normal(shape=(num_images, self.latent_dim))
        return self.generator(z, training=False)

    def generate_from_latent(self, z: tf.Tensor) -> tf.Tensor:
        """Generate images from specific latent vectors.

        Args:
            z: Latent vectors of shape (num_images, latent_dim).

        Returns:
            Generated images of shape (num_images, 28, 28, 1) in range [-1, 1].
        """
        return self.generator(z, training=False)
