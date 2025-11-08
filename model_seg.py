from keras import layers
from keras.models import Model

def train_model(input_dim, output_dim, activation="leaky_relu", dropout=0.4):
    inputs = layers.Input(shape=input_dim, name="input")
    x = layers.Rescaling(1./255)(inputs)

    # Block 1
    x = layers.SeparableConv2D(32, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.SeparableConv2D(32, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)

    # Block 2
    x = layers.SeparableConv2D(64, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.SeparableConv2D(64, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)

    # Block 3
    x = layers.SeparableConv2D(128, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(dropout)(x)

    outputs = layers.Dense(output_dim, activation="softmax", name="output")(x)
    return Model(inputs=inputs, outputs=outputs)
