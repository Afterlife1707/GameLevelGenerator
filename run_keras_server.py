import os
import flask
import numpy as np
import tensorflow as tf
import keras
from keras import backend as K

print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)

# Initialize Flask application
app = flask.Flask(__name__)

# Set CPU as available physical device
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Load the pre-trained Keras model
model_path = 'VAE_Room_Generator_Decoder_1352.h5'
model = None
decoder = None

img_width = 16
img_height = 16
num_channels = 1
input_shape = (img_height, img_width, num_channels)

latent_dim = 50  # Number of latent dim parameters

input_img = tf.keras.layers.Input(shape=input_shape, name='encoder_input')
x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=(2, 2), padding='same', activation='relu')(input_img)
x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')(x)
x = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=(2, 2), padding='same', activation='relu')(x)

conv_shape = tf.keras.backend.int_shape(x)  # Shape of conv to be provided to decoder
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(20, activation='relu')(x)

z_mu = tf.keras.layers.Dense(latent_dim, name='latent_mu')(x)  # Mean values of encoded input
z_sigma = tf.keras.layers.Dense(latent_dim, name='latent_sigma')(x)

class CustomLayer(keras.layers.Layer):
    def vae_loss(self, x, z_decoded):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)

        # Reconstruction loss (as we used sigmoid activation we can use binarycrossentropy)
        recon_loss = keras.metrics.binary_crossentropy(x, z_decoded) * img_width * img_height

        # KL divergence loss
        kl_loss = 1 + z_sigma - K.square(z_mu) - K.exp(z_sigma)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -.5

        return K.mean(recon_loss + (.75 * kl_loss))

    # add custom loss to the class
    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        loss = self.vae_loss(x, z_decoded)
        self.add_loss(loss, inputs=inputs)
        return x

def load_model():
    global model
    global decoder
    model = tf.keras.models.load_model(model_path)
    decoder = keras.saving.load_model(model_path, custom_objects={'CustomLayer': CustomLayer()}, safe_mode=False, compile=True)

# Load the model when the application starts
load_model()

# Preprocess input vectors
preprocessed_vectors = np.random.normal(0, 1, size=(100, latent_dim))  # Adjust size as needed

# Precompute and cache model predictions for preprocessed vectors
cached_predictions = model.predict(preprocessed_vectors)

def generate_image():
    # Randomly select a preprocessed vector
    random_index = np.random.randint(0, len(preprocessed_vectors))
    vector = preprocessed_vectors[random_index]

    # Use the cached prediction corresponding to the selected vector
    X = cached_predictions[random_index]

    return X

@app.route("/predict", methods=["POST"])
def predict():
     #data = {"success": False}

     if flask.request.method == "POST":
         image = generate_image()
         results = np.round(image[:,:,0], 3).tolist()

         return flask.jsonify(results)

     #return flask.jsonify(data)


if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    app.run(host='0.0.0.0')
