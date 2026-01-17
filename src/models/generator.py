"""Generator network for MNIST GAN - exact MATLAB match."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .base import BaseNetwork

# DCGAN weight initialization: N(0, 0.02)
WEIGHT_INIT = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)


class Generator(BaseNetwork):
    """Generator with boosted capacity.

    Doubled filter counts from original MATLAB to strengthen generator.
    Original: 112 -> 56 -> 28 -> 1
    Boosted:  224 -> 112 -> 56 -> 1
    """

    def __init__(self, latent_dim: int = 100, name: str = "Generator"):
        super().__init__(name=name)
        self.latent_dim = latent_dim

        # projectAndReshapeLayer: 100 -> 3*3*224 -> reshape to (3,3,224)
        self.dense = layers.Dense(3 * 3 * 224, use_bias=False, kernel_initializer=WEIGHT_INIT, name="proj_dense")
        self.reshape = layers.Reshape((3, 3, 224), name="proj_reshape")

        # transposedConv2dLayer(5, 112) - valid padding, stride 1
        self.tconv1 = layers.Conv2DTranspose(112, 5, strides=1, padding="valid", use_bias=False, kernel_initializer=WEIGHT_INIT, name="tconv1")
        self.bn1 = layers.BatchNormalization(momentum=0.9, name="bnorm1")

        # transposedConv2dLayer(5, 56, stride=2, same)
        self.tconv2 = layers.Conv2DTranspose(56, 5, strides=2, padding="same", use_bias=False, kernel_initializer=WEIGHT_INIT, name="tconv2")
        self.bn2 = layers.BatchNormalization(momentum=0.9, name="bnorm2")

        # transposedConv2dLayer(5, 1, stride=2, same)
        self.tconv3 = layers.Conv2DTranspose(1, 5, strides=2, padding="same", use_bias=False, kernel_initializer=WEIGHT_INIT, name="tconv3")

    def build_layers(self):
        pass  # Layers built in __init__

    def call(self, z, training=True):
        # projectAndReshapeLayer
        x = self.dense(z)
        x = self.reshape(x)

        # tconv1 -> bnorm1 -> relu1
        x = self.tconv1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)

        # tconv2 -> bnorm2 -> relu2
        x = self.tconv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)

        # tconv3 -> tanh
        x = self.tconv3(x)
        x = tf.nn.tanh(x)

        return x
