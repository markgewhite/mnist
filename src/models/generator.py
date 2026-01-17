"""Generator network for MNIST GAN - exact MATLAB match."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .base import BaseNetwork


class Generator(BaseNetwork):
    """Generator matching MATLAB exactly:

    MATLAB:
        featureInputLayer(100)
        projectAndReshapeLayer([3,3,112], 100)
        transposedConv2dLayer(5, 56)           -> bnorm1 -> relu1
        transposedConv2dLayer(5, 28, stride=2, same) -> bnorm2 -> relu2
        transposedConv2dLayer(5, 1, stride=2, same)  -> tanh
    """

    def __init__(self, latent_dim: int = 100, name: str = "Generator"):
        super().__init__(name=name)
        self.latent_dim = latent_dim

        # projectAndReshapeLayer: 100 -> 3*3*112 -> reshape to (3,3,112)
        self.dense = layers.Dense(3 * 3 * 112, use_bias=False, name="proj_dense")
        self.reshape = layers.Reshape((3, 3, 112), name="proj_reshape")

        # transposedConv2dLayer(5, 56) - valid padding, stride 1
        self.tconv1 = layers.Conv2DTranspose(56, 5, strides=1, padding="valid", use_bias=False, name="tconv1")
        self.bn1 = layers.BatchNormalization(name="bnorm1")

        # transposedConv2dLayer(5, 28, stride=2, same)
        self.tconv2 = layers.Conv2DTranspose(28, 5, strides=2, padding="same", use_bias=False, name="tconv2")
        self.bn2 = layers.BatchNormalization(name="bnorm2")

        # transposedConv2dLayer(5, 1, stride=2, same)
        self.tconv3 = layers.Conv2DTranspose(1, 5, strides=2, padding="same", use_bias=False, name="tconv3")

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
