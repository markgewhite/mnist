"""Generator network for MNIST GAN.

Architecture matches the MATLAB implementation:
- 100-dim latent input
- Project to 3x3x112 (not 7x7x256)
- TransposedConv with filter sequence: 56, 28, 1
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .base import BaseNetwork


class Generator(BaseNetwork):
    """Generator network that transforms latent vectors into images.

    Architecture:
        z (100) -> Dense(3*3*112) -> Reshape(3,3,112)
        -> TransConv(56, 5x5, valid) -> BN -> ReLU -> 7x7x56
        -> TransConv(28, 5x5, stride=2, same) -> BN -> ReLU -> 14x14x28
        -> TransConv(1, 5x5, stride=2, same) -> BN -> tanh -> 28x28x1
    """

    def __init__(
        self,
        latent_dim: int = 100,
        projection_shape: tuple = (3, 3, 112),
        base_filters: int = 28,
        kernel_size: int = 5,
        name: str = "Generator"
    ):
        """Initialize the Generator.

        Args:
            latent_dim: Dimension of latent input vector.
            projection_shape: Shape to project latent vector to (H, W, C).
            base_filters: Base number of filters (doubled for first layer).
            kernel_size: Kernel size for transposed convolutions.
            name: Name identifier for the network.
        """
        super().__init__(name=name)
        self.latent_dim = latent_dim
        self.projection_shape = projection_shape
        self.base_filters = base_filters
        self.kernel_size = kernel_size
        self.build_layers()

    def build_layers(self) -> None:
        """Build the generator layers."""
        # Project and reshape: 100 -> 3*3*112 -> (3, 3, 112)
        projection_units = (
            self.projection_shape[0] *
            self.projection_shape[1] *
            self.projection_shape[2]
        )

        self.dense_proj = layers.Dense(
            projection_units,
            use_bias=False,
            name="projection"
        )
        self.reshape = layers.Reshape(self.projection_shape, name="reshape")

        # TransConv1: 3x3x112 -> 7x7x56 (valid padding, no stride)
        self.tconv1 = layers.Conv2DTranspose(
            filters=2 * self.base_filters,  # 56
            kernel_size=self.kernel_size,
            strides=1,
            padding="valid",
            use_bias=False,
            name="tconv1"
        )
        self.bn1 = layers.BatchNormalization(name="bn1")

        # TransConv2: 7x7x56 -> 14x14x28 (same padding, stride 2)
        self.tconv2 = layers.Conv2DTranspose(
            filters=self.base_filters,  # 28
            kernel_size=self.kernel_size,
            strides=2,
            padding="same",
            use_bias=False,
            name="tconv2"
        )
        self.bn2 = layers.BatchNormalization(name="bn2")

        # TransConv3: 14x14x28 -> 28x28x1 (same padding, stride 2)
        self.tconv3 = layers.Conv2DTranspose(
            filters=1,
            kernel_size=self.kernel_size,
            strides=2,
            padding="same",
            use_bias=False,
            name="tconv3"
        )
        self.bn3 = layers.BatchNormalization(name="bn3")

        self._layers_built = True

    def call(self, z: tf.Tensor, training: bool = True) -> tf.Tensor:
        """Forward pass through the generator.

        Args:
            z: Latent vector of shape (batch_size, latent_dim).
            training: Whether in training mode (affects BatchNorm).

        Returns:
            Generated images of shape (batch_size, 28, 28, 1) in range [-1, 1].
        """
        # Project + Reshape
        x = self.dense_proj(z)
        x = self.reshape(x)

        # TransConv1 -> BN -> ReLU
        x = self.tconv1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)

        # TransConv2 -> BN -> ReLU
        x = self.tconv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)

        # TransConv3 -> BN -> tanh
        x = self.tconv3(x)
        x = self.bn3(x, training=training)
        x = tf.nn.tanh(x)

        return x
