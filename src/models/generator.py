"""Generator network for MNIST GAN - TensorFlow DCGAN tutorial architecture."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .base import BaseNetwork

# DCGAN weight initialization: N(0, 0.02)
WEIGHT_INIT = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)


class Generator(BaseNetwork):
    """Generator network matching TensorFlow DCGAN tutorial.

    Architecture: latent -> project -> reshape -> tconv1 -> tconv2 -> tconv3 -> output
    Larger projection (7x7x256) for more capacity.
    """

    # Network configuration (TensorFlow tutorial)
    filter_size = 5
    latent_dim = 100
    projection_size = (7, 7, 256)
    bn_momentum = 0.9
    leaky_alpha = 0.2

    def __init__(self, latent_dim: int = None, name: str = "Generator"):
        super().__init__(name=name)
        if latent_dim is not None:
            self.latent_dim = latent_dim

        proj_units = self.projection_size[0] * self.projection_size[1] * self.projection_size[2]

        # Project and reshape: 100 -> 7*7*256
        self.dense = layers.Dense(proj_units, use_bias=False, kernel_initializer=WEIGHT_INIT, name="proj_dense")
        self.bn0 = layers.BatchNormalization(momentum=self.bn_momentum, name="bnorm0")
        self.reshape = layers.Reshape(self.projection_size, name="proj_reshape")

        # tconv1: 7x7x256 -> 7x7x128 (stride 1, same)
        self.tconv1 = layers.Conv2DTranspose(
            128, self.filter_size, strides=1, padding="same",
            use_bias=False, kernel_initializer=WEIGHT_INIT, name="tconv1"
        )
        self.bn1 = layers.BatchNormalization(momentum=self.bn_momentum, name="bnorm1")

        # tconv2: 7x7x128 -> 14x14x64 (stride 2, same)
        self.tconv2 = layers.Conv2DTranspose(
            64, self.filter_size, strides=2, padding="same",
            use_bias=False, kernel_initializer=WEIGHT_INIT, name="tconv2"
        )
        self.bn2 = layers.BatchNormalization(momentum=self.bn_momentum, name="bnorm2")

        # tconv3: 14x14x64 -> 28x28x1 (stride 2, same)
        self.tconv3 = layers.Conv2DTranspose(
            1, self.filter_size, strides=2, padding="same",
            use_bias=False, kernel_initializer=WEIGHT_INIT, name="tconv3"
        )

    def build_layers(self):
        pass  # Layers built in __init__

    def call(self, z, training=True):
        # Project: 100 -> 12544, then BatchNorm + LeakyReLU
        x = self.dense(z)
        x = self.bn0(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=self.leaky_alpha)
        x = self.reshape(x)

        # tconv1: 7x7x256 -> 7x7x128
        x = self.tconv1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=self.leaky_alpha)

        # tconv2: 7x7x128 -> 14x14x64
        x = self.tconv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=self.leaky_alpha)

        # tconv3: 14x14x64 -> 28x28x1
        x = self.tconv3(x)
        x = tf.nn.tanh(x)

        return x
