"""Discriminator network for MNIST GAN - exact MATLAB match."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .base import BaseNetwork

# DCGAN weight initialization: N(0, 0.02)
WEIGHT_INIT = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)


class Discriminator(BaseNetwork):
    """Discriminator - reduced capacity version.

    Removed conv4 (224 filters) to weaken discriminator.
    Original was overpowering the generator.

    Architecture:
        imageInputLayer([28,28,1], Normalization='none')
        dropoutLayer(0.5)
        conv2dLayer(5, 28, stride=2, same)  -> lrelu1        [output: 14x14x28]
        conv2dLayer(5, 56, stride=2, same)  -> bn2 -> lrelu2 [output: 7x7x56]
        conv2dLayer(5, 112, stride=2, same) -> bn3 -> lrelu3 [output: 4x4x112]
        conv2dLayer(4, 1)                                    [output: 1x1x1]
    """

    def __init__(self, dropout_rate: float = 0.5, leaky_alpha: float = 0.2, name: str = "Discriminator"):
        super().__init__(name=name)
        self.leaky_alpha = leaky_alpha

        # dropoutLayer(0.5)
        self.dropout = layers.Dropout(dropout_rate, name="dropout")

        # conv2dLayer(5, 28, stride=2, same) - no BN on first conv
        self.conv1 = layers.Conv2D(28, 5, strides=2, padding="same", kernel_initializer=WEIGHT_INIT, name="conv1")

        # conv2dLayer(5, 56, stride=2, same) -> bn2
        self.conv2 = layers.Conv2D(56, 5, strides=2, padding="same", kernel_initializer=WEIGHT_INIT, name="conv2")
        self.bn2 = layers.BatchNormalization(momentum=0.9, name="bn2")

        # conv2dLayer(5, 112, stride=2, same) -> bn3
        self.conv3 = layers.Conv2D(112, 5, strides=2, padding="same", kernel_initializer=WEIGHT_INIT, name="conv3")
        self.bn3 = layers.BatchNormalization(momentum=0.9, name="bn3")

        # conv2dLayer(4, 1) - final logit (4x4 kernel to go from 4x4 to 1x1)
        self.conv4 = layers.Conv2D(1, 4, strides=1, padding="valid", kernel_initializer=WEIGHT_INIT, name="conv4")

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

        # conv4 -> logit (4x4 kernel, no activation)
        x = self.conv4(x)
        x = tf.reshape(x, [-1, 1])

        return x
