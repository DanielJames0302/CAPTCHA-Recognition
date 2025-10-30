from keras import layers
from keras.models import Model

from mltu.tensorflow.model_utils import residual_block

def train_model(input_dim, output_dim, activation="leaky_relu", dropout=0.25):
    
    inputs = layers.Input(shape=input_dim, name="input")

    # normalize to [0,1]
    x = layers.Lambda(lambda x: x / 255.0)(inputs)

    # Conv block 1
    x = layers.Conv2D(32, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(32, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Conv block 2
    x = layers.Conv2D(64, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Global pooling + dense head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation=None, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(dropout)(x)

    outputs = layers.Dense(output_dim + 1, activation="softmax", name="output")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model
