"""Discriminator network for MNIST GAN.

Architecture matches the MATLAB implementation:
- Input dropout 0.5
- Conv filter sequence: 28, 56, 112, 224, 1
- Final 2x2 valid conv to single logit
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .base import BaseNetwork


class Discriminator(BaseNetwork):
    """Discriminator network that classifies images as real or fake.

    Architecture:
        28x28x1 -> Dropout(0.5)
        -> Conv(28, 5x5, stride=2, same) + LeakyReLU(0.2) -> 14x14x28
        -> Conv(56, 5x5, stride=2, same) + BN + LeakyReLU(0.2) -> 7x7x56
        -> Conv(112, 5x5, stride=2, same) + BN + LeakyReLU(0.2) -> 4x4x112
        -> Conv(224, 5x5, stride=2, same) + BN + LeakyReLU(0.2) -> 2x2x224
        -> Conv(1, 2x2, valid) -> 1x1x1 logit
    """

    def __init__(
        self,
        input_shape: tuple = (28, 28, 1),
        base_filters: int = 28,
        kernel_size: int = 5,
        dropout_rate: float = 0.5,
        leaky_relu_alpha: float = 0.2,
        name: str = "Discriminator"
    ):
        """Initialize the Discriminator.

        Args:
            input_shape: Shape of input images (H, W, C).
            base_filters: Base number of filters (multiplied for deeper layers).
            kernel_size: Kernel size for convolutions.
            dropout_rate: Dropout probability at input.
            leaky_relu_alpha: Negative slope for LeakyReLU.
            name: Name identifier for the network.
        """
        super().__init__(name=name)
        self.input_image_shape = input_shape
        self.base_filters = base_filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.leaky_relu_alpha = leaky_relu_alpha
        self.build_layers()

    def build_layers(self) -> None:
        """Build the discriminator layers."""
        # Input dropout
        self.input_dropout = layers.Dropout(self.dropout_rate, name="input_dropout")

        # Conv1: 28x28x1 -> 14x14x28 (no BN on first layer)
        self.conv1 = layers.Conv2D(
            filters=self.base_filters,  # 28
            kernel_size=self.kernel_size,
            strides=2,
            padding="same",
            name="conv1"
        )

        # Conv2: 14x14x28 -> 7x7x56
        self.conv2 = layers.Conv2D(
            filters=2 * self.base_filters,  # 56
            kernel_size=self.kernel_size,
            strides=2,
            padding="same",
            name="conv2"
        )
        self.bn2 = layers.BatchNormalization(name="bn2")

        # Conv3: 7x7x56 -> 4x4x112
        self.conv3 = layers.Conv2D(
            filters=4 * self.base_filters,  # 112
            kernel_size=self.kernel_size,
            strides=2,
            padding="same",
            name="conv3"
        )
        self.bn3 = layers.BatchNormalization(name="bn3")

        # Conv4: 4x4x112 -> 2x2x224
        self.conv4 = layers.Conv2D(
            filters=8 * self.base_filters,  # 224
            kernel_size=self.kernel_size,
            strides=2,
            padding="same",
            name="conv4"
        )
        self.bn4 = layers.BatchNormalization(name="bn4")

        # Conv5: 2x2x224 -> 1x1x1 (logit output)
        self.conv5 = layers.Conv2D(
            filters=1,
            kernel_size=2,
            strides=1,
            padding="valid",
            name="conv5_logits"
        )

        self._layers_built = True

    def call(self, x: tf.Tensor, training: bool = True) -> tf.Tensor:
        """Forward pass through the discriminator.

        Args:
            x: Input images of shape (batch_size, 28, 28, 1).
            training: Whether in training mode (affects Dropout and BatchNorm).

        Returns:
            Logits of shape (batch_size, 1) for real/fake classification.
        """
        # Input dropout
        x = self.input_dropout(x, training=training)

        # Conv1 + LeakyReLU (no BN)
        x = self.conv1(x)
        x = tf.nn.leaky_relu(x, alpha=self.leaky_relu_alpha)

        # Conv2 + BN + LeakyReLU
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=self.leaky_relu_alpha)

        # Conv3 + BN + LeakyReLU
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=self.leaky_relu_alpha)

        # Conv4 + BN + LeakyReLU
        x = self.conv4(x)
        x = self.bn4(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=self.leaky_relu_alpha)

        # Conv5 -> logits
        x = self.conv5(x)

        # Flatten to (batch_size, 1)
        x = tf.reshape(x, [-1, 1])

        return x
