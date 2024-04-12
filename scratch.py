from keras.models import load_model
from matplotlib import pyplot
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
import numpy as np
import keras
from keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape, LeakyReLU, BatchNormalization, \
    RandomFlip, RandomRotation
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

img_width  = 16
img_height = 16
num_channels = 1
input_shape = (img_height, img_width, num_channels)

latent_dim = 150 # Number of latent dim parameters

model_path = 'VAE_Room_Generator_Decoder_1352.h5'

input_img = Input(shape=input_shape, name='encoder_input')
#x = RandomFlip(mode='horizontal_and_vertical')(input_img) # Already flipped images in data aug so this isn't really necessary
x = Conv2D(filters=128, kernel_size=3, strides=(2,2), padding='same', activation='relu')(input_img)
x = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
x = Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')(x)
x = Conv2D(filters=16, kernel_size=3, strides=(2,2), padding='same', activation='relu')(x)

conv_shape = K.int_shape(x) #Shape of conv to be provided to decoder
# Flatten
x = Flatten()(x)
x = Dense(20, activation='relu')(x)

z_mu = Dense(latent_dim, name='latent_mu')(x)  # Mean values of encoded input
z_sigma = Dense(latent_dim, name='latent_sigma')(x)  # Std dev. (variance) of encoded input


class CustomLayer(keras.layers.Layer):
    def vae_loss(self, x, z_decoded):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)

        # Reconstruction loss (as we used sigmoid activation we can use binarycrossentropy)
        recon_loss = keras.metrics.binary_crossentropy(x, z_decoded) * img_width * img_height

        # KL divergence
        # kl_loss = -5e-4 * K.mean(1 + z_sigma - K.square(z_mu) - K.exp(z_sigma), axis=-1)

        # KL divergence loss
        kl_loss = 1 + z_sigma - K.square(z_mu) - K.exp(z_sigma)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5

        return K.mean(recon_loss + kl_loss)

    # add custom loss to the class
    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        loss = self.vae_loss(x, z_decoded)
        self.add_loss(loss, inputs=inputs)
        return x

model = load_model(model_path)
# model = load_model('C:/Users/aquil/Downloads/PixelCNN_Room_Generator_Model.keras', custom_objects={'CustomLayer': CustomLayer()}, safe_mode=False, compile=True)

# model.summary()
vector = np.random.normal(0,1,size=(1,50))

X = model.predict(vector)

pyplot.imshow(X[0, :, :, 0], cmap='gray_r')
pyplot.show()