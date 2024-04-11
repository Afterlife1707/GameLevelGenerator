# import the necessary packages
from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import flask
from keras.models import load_model
from matplotlib import pyplot
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
import numpy as np
from keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape, LeakyReLU, BatchNormalization, \
    RandomFlip, RandomRotation
from keras import backend as K
from numpy import asarray
import tensorflow as tf

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None

def load_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global model
    model = tf.keras.models.load_model('C:/Users/aquil/Downloads/VAE_Room_Generator_Decoder_1352.h5')

def generate_image():
    model = tf.keras.models.load_model('C:/Users/aquil/Downloads/VAE_Room_Generator_Decoder_1352.h5')
    img_width = 16
    img_height = 16
    num_channels = 1
    input_shape = (img_height, img_width, num_channels)

    latent_dim = 150  # Number of latent dim parameters

    input_img = Input(shape=input_shape, name='encoder_input')
    # x = RandomFlip(mode='horizontal_and_vertical')(input_img) # Already flipped images in data aug so this isn't really necessary
    x = Conv2D(filters=128, kernel_size=3, strides=(2, 2), padding='same', activation='relu')(input_img)
    x = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(filters=16, kernel_size=3, strides=(2, 2), padding='same', activation='relu')(x)

    conv_shape = K.int_shape(x)  # Shape of conv to be provided to decoder
    # Flatten
    x = Flatten()(x)
    x = Dense(20, activation='relu')(x)

    z_mu = Dense(latent_dim, name='latent_mu')(x)  # Mean values of encoded input
    z_sigma = Dense(latent_dim, name='latent_sigma')(x)  # Std dev. (variance) of encoded input

    # model.summary()
    vector = np.random.normal(0, 1, size=(1, 50))

    X = model.predict(vector)

    # return the processed image
    return X

@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        # generate the image and prepare it for classification
        image = generate_image()
        print("Image has been generated")

        # classify the input image and then initialize the list
        # of predictions to return to the client
        results = asarray(image)
        data["results"] = []

        print(results.shape)
        print(results)

        # loop over the results and add them to the list of
        # returned predictions
        # for (imagenetID, label, prob) in results[0]:
        #     r = {"label": label, "probability": float(prob)}
        #     data["predictions"].append(r)

        # indicate that the request was a success
        data["success"] = True
        # if flask.request.files.get("image"):


    # return the data dictionary as a JSON response
    return flask.jsonify(data)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    # load_model()
    app.run()