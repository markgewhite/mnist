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
        name: str = "MNISTGAN"
    ):
        """Initialize the MNIST GAN.

        Args:
            latent_dim: Dimension of latent input vector for generator.
            flip_factor: Fraction of real labels to flip (regularization).
            name: Name identifier for the model.
        """
        super().__init__(name=name)
        self.latent_dim = latent_dim
        self.flip_factor = flip_factor

        # Build component networks
        self.generator = Generator(latent_dim=latent_dim)
        self.discriminator = Discriminator()

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

    def train_step(self, real_images: tf.Tensor) -> dict:
        """Custom training step - EXACT MATLAB match.

        MATLAB modelGradients.m:
            1. Forward pass through both networks
            2. Compute probabilities with sigmoid
            3. Flip some real probabilities (not labels)
            4. Compute log-based losses manually
            5. Compute gradients and update
        """
        batch_size = tf.shape(real_images)[0]
        eps = 1e-7  # For numerical stability in log

        # Sample random latent vectors
        z = tf.random.normal(shape=(batch_size, self.latent_dim))

        # === Compute both gradients from same fake images (matches MATLAB) ===
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            # Generate fake images
            fake_images = self.generator(z, training=True)

            # Get discriminator logits
            real_logits = self.discriminator(real_images, training=True)
            fake_logits = self.discriminator(fake_images, training=True)

            # Convert to probabilities (MATLAB: sigmoid)
            prob_real = tf.sigmoid(real_logits)
            prob_fake = tf.sigmoid(fake_logits)

            # Flip some real probabilities (MATLAB: probReal(:,:,:,idx) = 1 - probReal(:,:,:,idx))
            num_flip = tf.cast(tf.cast(batch_size, tf.float32) * self.flip_factor, tf.int32)
            flip_indices = tf.random.shuffle(tf.range(batch_size))[:num_flip]
            flip_mask = tf.scatter_nd(
                tf.expand_dims(flip_indices, 1),
                tf.ones(num_flip),
                [batch_size]
            )
            flip_mask = tf.reshape(flip_mask, [-1, 1])

            # Flip: where mask=1, prob becomes (1-prob)
            prob_real_flipped = prob_real * (1 - flip_mask) + (1 - prob_real) * flip_mask

            # MATLAB loss: lossDiscriminator = -mean(log(probReal)) - mean(log(1-probGenerated))
            d_loss = -tf.reduce_mean(tf.math.log(prob_real_flipped + eps)) \
                     -tf.reduce_mean(tf.math.log(1 - prob_fake + eps))

            # MATLAB loss: lossGenerator = -mean(log(probGenerated))
            g_loss = -tf.reduce_mean(tf.math.log(prob_fake + eps))

        # Compute gradients
        d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)

        # Update both networks (MATLAB order: D first, then G)
        self.d_optimizer.apply_gradients(
            zip(d_grads, self.discriminator.trainable_variables)
        )
        self.g_optimizer.apply_gradients(
            zip(g_grads, self.generator.trainable_variables)
        )

        # MATLAB scores
        d_score = (tf.reduce_mean(prob_real) + tf.reduce_mean(1 - prob_fake)) / 2
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
