"""Discriminator network for MNIST GAN - exact MATLAB match."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .base import BaseNetwork

# DCGAN weight initialization: N(0, 0.02)
WEIGHT_INIT = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)


class Discriminator(BaseNetwork):
    """Discriminator network matching MATLAB architecture.

    Architecture: input -> dropout -> conv1 -> conv2 -> conv3 -> conv4 -> conv5 -> output
    Filter progression: n_filters -> 2*n_filters -> 4*n_filters -> 8*n_filters -> 1
    """

    # Network configuration (matching MATLAB)
    n_filters = 28
    filter_size = 5
    dropout_rate = 0.5
    leaky_alpha = 0.2
    bn_momentum = 0.9

    def __init__(self, name: str = "Discriminator"):
        super().__init__(name=name)

        # dropoutLayer
        self.dropout = layers.Dropout(self.dropout_rate, name="dropout")

        # conv2dLayer -> n_filters (no BN on first conv)
        self.conv1 = layers.Conv2D(
            self.n_filters, self.filter_size, strides=2, padding="same",
            kernel_initializer=WEIGHT_INIT, name="conv1"
        )

        # conv2dLayer -> 2*n_filters + BN
        self.conv2 = layers.Conv2D(
            2 * self.n_filters, self.filter_size, strides=2, padding="same",
            kernel_initializer=WEIGHT_INIT, name="conv2"
        )
        self.bn2 = layers.BatchNormalization(momentum=self.bn_momentum, name="bn2")

        # conv2dLayer -> 4*n_filters + BN
        self.conv3 = layers.Conv2D(
            4 * self.n_filters, self.filter_size, strides=2, padding="same",
            kernel_initializer=WEIGHT_INIT, name="conv3"
        )
        self.bn3 = layers.BatchNormalization(momentum=self.bn_momentum, name="bn3")

        # conv2dLayer -> 8*n_filters + BN
        self.conv4 = layers.Conv2D(
            8 * self.n_filters, self.filter_size, strides=2, padding="same",
            kernel_initializer=WEIGHT_INIT, name="conv4"
        )
        self.bn4 = layers.BatchNormalization(momentum=self.bn_momentum, name="bn4")

        # conv2dLayer -> 1 (final logit, 2x2 kernel)
        self.conv5 = layers.Conv2D(
            1, 2, strides=1, padding="valid",
            kernel_initializer=WEIGHT_INIT, name="conv5"
        )

    def build_layers(self):
        pass  # Layers built in __init__

    def call(self, x, training=True):
        # dropout
        x = self.dropout(x, training=training)

        # conv1 -> lrelu1 (no BN)
        x = self.conv1(x)
        x = tf.nn.leaky_relu(x, alpha=self.leaky_alpha)

        # conv2 -> bn2 -> lrelu2
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=self.leaky_alpha)

        # conv3 -> bn3 -> lrelu3
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=self.leaky_alpha)

        # conv4 -> bn4 -> lrelu4
        x = self.conv4(x)
        x = self.bn4(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=self.leaky_alpha)

        # conv5 -> logit
        x = self.conv5(x)
        x = tf.reshape(x, [-1, 1])

        return x
