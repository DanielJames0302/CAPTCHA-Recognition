import tensorflow as tf
from tensorflow import keras
from keras import layers
import typing

def activation_layer(layer, activation: str="relu", alpha: float=0.1) -> tf.Tensor:
    """ Activation layer wrapper for LeakyReLU and ReLU activation functions
    Args:
        layer: tf.Tensor
        activation: str, activation function name (default: 'relu')
        alpha: float (LeakyReLU activation function parameter)
    Returns:
        tf.Tensor
    """
    if activation == "relu":
        layer = layers.ReLU()(layer)
    elif activation == "leaky_relu":
        layer = layers.LeakyReLU(alpha=alpha)(layer)

    return layer


def residual_block(
        x: tf.Tensor,
        filter_num: int,
        strides: typing.Union[int, list] = 2,
        kernel_size: typing.Union[int, list] = 3,
        skip_conv: bool = True,
        padding: str = "same",
        kernel_initializer: str = "he_uniform",
        activation: str = "relu",
        dropout: float = 0.2):
    # Create skip connection tensor
    x_skip = x

    # Perform 1-st convolution
    x = layers.Conv2D(filter_num, kernel_size, padding = padding, strides = strides, kernel_initializer=kernel_initializer)(x)
    x = layers.BatchNormalization()(x)
    x = activation_layer(x, activation=activation)

    # Perform 2-nd convoluti
    x = layers.Conv2D(filter_num, kernel_size, padding = padding, kernel_initializer=kernel_initializer)(x)
    x = layers.BatchNormalization()(x)

    # Perform 3-rd convolution if skip_conv is True, matchin the number of filters and the shape of the skip connection tensor
    if skip_conv:
        x_skip = layers.Conv2D(filter_num, 1, padding = padding, strides = strides, kernel_initializer=kernel_initializer)(x_skip)

    # Add x and skip connection and apply activation function
    x = layers.Add()([x, x_skip])
    x = activation_layer(x, activation=activation)

    # Apply dropout
    if dropout:
        x = layers.Dropout(dropout)(x)

    return x