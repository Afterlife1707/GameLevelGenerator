import os
import flask
import numpy as np
import tensorflow as tf

# Set CPU as available physical device
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Disable eager execution
tf.compat.v1.disable_eager_execution()

# Initialize Flask application
app = flask.Flask(__name__)

# Load the pre-trained Keras model
model_path = 'VAE_Room_Generator_Decoder_1352.h5'
model = None

def load_model():
    global model
    model = tf.keras.models.load_model(model_path)

def generate_image():
    # Load the model
    if model is None:
        load_model()

    img_width = 16
    img_height = 16
    num_channels = 1
    input_shape = (img_height, img_width, num_channels)

    latent_dim = 150  # Number of latent dim parameters

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

    vector = np.random.normal(0, 1, size=(1, 50))
    X = model.predict(vector)

    return X

@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}

    if flask.request.method == "POST":
        image = generate_image()
        print("Image has been generated")

        results = np.asarray(image)
        data["results"] = results.tolist()

        data["success"] = True

    return flask.jsonify(data)

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    app.run()
