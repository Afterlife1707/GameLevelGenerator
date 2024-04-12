import os
import flask
import numpy as np
from numpy import asarray
import tensorflow as tf

# Set CPU as available physical device
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Initialize Flask application
app = flask.Flask(__name__)

# Load the pre-trained Keras model
model_path = 'VAE_Room_Generator_Decoder_1352.h5'
model = tf.keras.models.load_model(model_path)
latent_dim = 150

def generate_image():
    # Generate a random vector
    vector = np.random.normal(0, 1, size=(1, latent_dim))
    # Generate image using the model
    X = model.predict(vector)
    return X

@app.route("/predict", methods=["POST"])
def predict():
    # Initialize the data dictionary
    data = {"success": False}

    # Ensure an image was properly uploaded
    if flask.request.method == "POST":
        # Generate the image
        image = generate_image()
        # Prepare the image for response
        results = asarray(image)
        data["results"] = results.tolist()
        data["success"] = True

    # Return the data dictionary as JSON response
    return flask.jsonify(data)

if __name__ == "__main__":
    print("Loading Keras model and starting Flask server...")
    app.run()
