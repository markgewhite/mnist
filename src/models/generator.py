"""Generator network for MNIST GAN - exact MATLAB match."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .base import BaseNetwork

# DCGAN weight initialization: N(0, 0.02)
WEIGHT_INIT = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)


class Generator(BaseNetwork):
    """Generator network matching MATLAB architecture.

    Architecture: latent -> project -> tconv1 -> tconv2 -> tconv3 -> output
    Filter progression: 4*n_filters -> 2*n_filters -> n_filters -> 1
    """

    # Network configuration (matching MATLAB)
    n_filters = 28
    filter_size = 5
    latent_dim = 100
    projection_size = (3, 3, 112)  # 4 * n_filters
    bn_momentum = 0.9
    leaky_alpha = 0.2

    def __init__(self, latent_dim: int = None, name: str = "Generator"):
        super().__init__(name=name)
        if latent_dim is not None:
            self.latent_dim = latent_dim

        proj_units = self.projection_size[0] * self.projection_size[1] * self.projection_size[2]

        # projectAndReshapeLayer
        self.dense = layers.Dense(proj_units, use_bias=False, kernel_initializer=WEIGHT_INIT, name="proj_dense")
        self.reshape = layers.Reshape(self.projection_size, name="proj_reshape")

        # transposedConv2dLayer -> 2*n_filters (valid padding, stride 1)
        self.tconv1 = layers.Conv2DTranspose(
            2 * self.n_filters, self.filter_size, strides=1, padding="valid",
            use_bias=False, kernel_initializer=WEIGHT_INIT, name="tconv1"
        )
        self.bn1 = layers.BatchNormalization(momentum=self.bn_momentum, name="bnorm1")

        # transposedConv2dLayer -> n_filters (stride 2, same)
        self.tconv2 = layers.Conv2DTranspose(
            self.n_filters, self.filter_size, strides=2, padding="same",
            use_bias=False, kernel_initializer=WEIGHT_INIT, name="tconv2"
        )
        self.bn2 = layers.BatchNormalization(momentum=self.bn_momentum, name="bnorm2")

        # transposedConv2dLayer -> 1 channel output (stride 2, same)
        self.tconv3 = layers.Conv2DTranspose(
            1, self.filter_size, strides=2, padding="same",
            use_bias=False, kernel_initializer=WEIGHT_INIT, name="tconv3"
        )

    def build_layers(self):
        pass  # Layers built in __init__

    def call(self, z, training=True):
        # projectAndReshapeLayer
        x = self.dense(z)
        x = self.reshape(x)

        # tconv1 -> bnorm1 -> leaky_relu1
        x = self.tconv1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=self.leaky_alpha)

        # tconv2 -> bnorm2 -> leaky_relu2
        x = self.tconv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=self.leaky_alpha)

        # tconv3 -> tanh
        x = self.tconv3(x)
        x = tf.nn.tanh(x)

        return x
