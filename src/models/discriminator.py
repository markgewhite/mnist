"""Discriminator network for MNIST GAN - exact MATLAB match."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .base import BaseNetwork

# DCGAN weight initialization: N(0, 0.02)
WEIGHT_INIT = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)


class Discriminator(BaseNetwork):
    """Discriminator network matching TensorFlow DCGAN tutorial.

    Architecture: input -> conv1 -> dropout -> conv2 -> dropout -> flatten -> dense
    Simpler than MATLAB version - 2 conv layers, no BatchNorm, dropout after each conv.
    """

    # Network configuration
    n_filters = 64
    filter_size = 5
    dropout_rate = 0.3
    leaky_alpha = 0.2

    def __init__(self, name: str = "Discriminator"):
        super().__init__(name=name)

        # conv2dLayer -> n_filters
        self.conv1 = layers.Conv2D(
            self.n_filters, self.filter_size, strides=2, padding="same",
            kernel_initializer=WEIGHT_INIT, name="conv1"
        )
        self.dropout1 = layers.Dropout(self.dropout_rate, name="dropout1")

        # conv2dLayer -> 2*n_filters
        self.conv2 = layers.Conv2D(
            2 * self.n_filters, self.filter_size, strides=2, padding="same",
            kernel_initializer=WEIGHT_INIT, name="conv2"
        )
        self.dropout2 = layers.Dropout(self.dropout_rate, name="dropout2")

        # Flatten and dense output
        self.flatten = layers.Flatten(name="flatten")
        self.dense = layers.Dense(1, kernel_initializer=WEIGHT_INIT, name="output")

    def build_layers(self):
        pass  # Layers built in __init__

    def call(self, x, training=True):
        # conv1 -> lrelu -> dropout
        x = self.conv1(x)
        x = tf.nn.leaky_relu(x, alpha=self.leaky_alpha)
        x = self.dropout1(x, training=training)

        # conv2 -> lrelu -> dropout
        x = self.conv2(x)
        x = tf.nn.leaky_relu(x, alpha=self.leaky_alpha)
        x = self.dropout2(x, training=training)

        # flatten -> dense -> logit
        x = self.flatten(x)
        x = self.dense(x)

        return x
