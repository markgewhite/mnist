"""Abstract base class for GAN networks."""

from abc import ABC, abstractmethod

import tensorflow as tf
from tensorflow import keras


class BaseNetwork(keras.Model, ABC):
    """Abstract base class for Generator and Discriminator networks.

    Provides a common interface for building and summarizing networks.
    """

    def __init__(self, name: str = "BaseNetwork"):
        """Initialize the base network.

        Args:
            name: Name identifier for the network.
        """
        super().__init__(name=name)
        self._layers_built = False

    @abstractmethod
    def build_layers(self) -> None:
        """Build the network layers. Must be implemented by subclasses."""
        pass

    def summary_with_shapes(self, input_shape: tuple) -> None:
        """Print model summary with input/output shapes.

        Args:
            input_shape: Shape of input tensor (excluding batch dimension).
        """
        # Build model by calling with dummy input
        dummy_input = tf.zeros((1,) + input_shape)
        _ = self(dummy_input, training=False)
        self.summary()
