import cv2
import numpy as np
import tensorflow as tf
import os
import logging
from keras.models import load_model
from keras import backend as K

from CNN.ottakshara_dict import ottakshara_mapping

# Get the absolute path to the project root
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))  # Adjusted for direct access

# Construct absolute paths for CNN models
CHAR_MODEL_PATH = os.path.join(PROJECT_ROOT, 'cnn_model.h5')
OTTAKSHARA_MODEL_PATH = os.path.join(PROJECT_ROOT, 'cnn_model_ottak.h5')

# Print paths for debugging
print("CHAR_MODEL_PATH:", CHAR_MODEL_PATH)
print("OTTAKSHARA_MODEL_PATH:", OTTAKSHARA_MODEL_PATH)


def keras_process_image(img):
    """Reshape the image for Keras input format."""
    image_x, image_y = img.shape
    img = np.reshape(img, (1, image_x, image_y, 1))
    return img


def keras_predict(image, model):
    """Predict the class of the given image using the provided model."""
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    processed = keras_process_image(image)
    pred_probab = list(model.predict(processed)[0])
    sorted_probab = sorted(pred_probab, reverse=True)
    return pred_probab.index(sorted_probab[0])


def recognize(dir):
    """Recognize characters from images in the specified directory."""
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    logging.getLogger("tensorflow").setLevel(logging.ERROR)

    # Ensure the model files exist
    if not os.path.exists(OTTAKSHARA_MODEL_PATH) or not os.path.exists(CHAR_MODEL_PATH):
        raise FileNotFoundError("One or both model files not found!")

    # Load models with absolute paths
    ottakshara_model = load_model(OTTAKSHARA_MODEL_PATH)
    char_model = load_model(CHAR_MODEL_PATH)

    # Dictionary to store predictions
    predictions = {}

    # Sort image names alphabetically
    flist = sorted(os.listdir(dir))

    # Iterate through images and predict
    for file in flist:
        file_path = os.path.abspath(os.path.join(dir, file))
        
        if '-1' in file:  # Ottakshara character
            predictions[file] = ottakshara_mapping[keras_predict(file_path, ottakshara_model)]
        else:  # Regular character
            predictions[file] = keras_predict(file_path, char_model)

    K.clear_session()
    return predictions
