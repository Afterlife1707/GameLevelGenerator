from keras import layers
#!pip install keras --upgrade # We need Keras 3 for the PixelCNN Model, certain things will not work otherwise i.e. ops
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import ops
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import keras
from keras.saving import load_model

num_classes = 1
input_shape = (8, 8, 1)
n_residual_blocks = 4

class PixelConvLayer(layers.Layer):
    def __init__(self, mask_type, **kwargs):
        super().__init__()
        self.mask_type = mask_type
        self.conv = layers.Conv2D(**kwargs)

    def build(self, input_shape):
        # Build the conv2d layer to initialize kernel variables
        self.conv.build(input_shape)
        # Use the initialized kernel to create the mask
        kernel_shape = ops.shape(self.conv.kernel)
        self.mask = np.zeros(shape=kernel_shape)
        self.mask[: kernel_shape[0] // 2, ...] = 1.0
        self.mask[kernel_shape[0] // 2, : kernel_shape[1] // 2, ...] = 1.0
        if self.mask_type == "B":
            self.mask[kernel_shape[0] // 2, kernel_shape[1] // 2, ...] = 1.0

    def call(self, inputs):
        self.conv.kernel.assign(self.conv.kernel * self.mask)
        return self.conv(inputs)

class ResidualBlock(keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = keras.layers.Conv2D(
            filters=filters, kernel_size=1, activation="relu"
        )
        self.pixel_conv = PixelConvLayer(
            mask_type="B",
            filters=filters // 2,
            kernel_size=3,
            activation="relu",
            padding="same",
        )
        self.conv2 = keras.layers.Conv2D(
            filters=filters, kernel_size=1, activation="relu"
        )

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pixel_conv(x)
        x = self.conv2(x)
        return keras.layers.add([inputs, x])

inputs = keras.Input(shape=input_shape)
x = PixelConvLayer(
    mask_type="A", filters=128, kernel_size=7, activation="relu", padding="same"
)(inputs)

out = keras.layers.Conv2D(
    filters=1, kernel_size=1, strides=1, activation="sigmoid", padding="valid"
)(x)

pixel_cnn = keras.Model(inputs, out)

loaded_pixel_cnn = load_model('PixelCNN_Dungeon_Generator_Model_3618.keras', custom_objects={'PixelConvLayer': PixelConvLayer, 'ResidualBlock': ResidualBlock})
batch = 9
pixels = np.zeros(shape=(batch,) + (pixel_cnn.input_shape)[1:])
batch, rows, cols, channels = pixels.shape
for row in tqdm(range(rows)):
    for col in range(cols):
        for channel in range(channels):
            probs = loaded_pixel_cnn.predict(pixels)[:, row, col, channel]

            pixels[:, row, col, channel] = tf.math.ceil(
                probs - tf.random.uniform(probs.shape)
            )

def deprocess_image(x):
    x *= 256
    x = np.clip(x, 0, 255).astype("uint8")
    return x

img = deprocess_image(np.squeeze(pixels[0], -1))
print(img)
plt.imshow(img, cmap='gray_r')
plt.show()