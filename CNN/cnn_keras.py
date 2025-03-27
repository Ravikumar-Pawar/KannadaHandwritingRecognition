import numpy as np
import pickle
import cv2
import os
import sys
import glob
import logging
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set image format
K.set_image_data_format('channels_last')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Dataset directory
DEFAULT_DIRECTORY = "/workspaces/KannadaHandwritingRecognition/dataset/dataset/Kannada/Hnd"
directory = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIRECTORY

if not os.path.exists(directory):
    logging.error(f"Dataset path does not exist: {directory}")
    sys.exit(1)

logging.info(f"Using dataset directory: {directory}")

def get_all_images():
    image_paths = glob.glob(os.path.join(directory, "Img", "**", "*.png"), recursive=True)
    if not image_paths:
        logging.error("No PNG images found in dataset.")
        sys.exit(1)
    logging.info(f"Found {len(image_paths)} PNG images.")
    return image_paths

def get_image_size():
    image_paths = get_all_images()
    img = cv2.imread(image_paths[0], cv2.IMREAD_GRAYSCALE)
    if img is None:
        logging.error("Failed to read sample image.")
        sys.exit(1)
    logging.info(f"Sample image size: {img.shape}")
    return img.shape

def get_num_of_classes():
    class_dirs = [d for d in os.listdir(os.path.join(directory, "Img")) if os.path.isdir(os.path.join(directory, "Img", d))]
    logging.info(f"Number of classes found: {len(class_dirs)}")
    return len(class_dirs)

image_x, image_y = get_image_size()

def cnn_model():
    logging.info("Building CNN model...")
    num_of_classes = get_num_of_classes()
    model = Sequential([
        Conv2D(52, (5, 5), input_shape=(image_x, image_y, 1), activation='tanh'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(64, (5, 5), activation='tanh'),
        MaxPooling2D(pool_size=(5, 5), strides=(5, 5)),
        Flatten(),
        Dropout(0.5),
        Dense(num_of_classes, activation='softmax')
    ])
    
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    logging.info("Model compiled successfully.")
    checkpoint = ModelCheckpoint("cnn_model.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    return model, [checkpoint]

def load_dataset():
    logging.info("Loading dataset...")
    image_paths = get_all_images()
    labels = []
    images = []

    for img_path in image_paths:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(cv2.resize(img, (image_x, image_y)))
            label = int(img_path.split("/")[-2].replace("Sample", "")) - 1
            labels.append(label)

    images = np.array(images).reshape(-1, image_x, image_y, 1)
    labels = np.array(labels)

    with open("train_images.pkl", "wb") as f:
        pickle.dump(images, f)
    with open("train_labels.pkl", "wb") as f:
        pickle.dump(labels, f)

    logging.info(f"Dataset saved: {len(images)} images, {len(labels)} labels.")

def train():
    logging.info("Loading training data...")
    if not os.path.exists("train_images.pkl") or not os.path.exists("train_labels.pkl"):
        logging.error("Training data not found. Run preprocessing first.")
        sys.exit(1)

    with open("train_images.pkl", "rb") as f:
        train_images = np.array(pickle.load(f))
    with open("train_labels.pkl", "rb") as f:
        train_labels = np.array(pickle.load(f), dtype=np.int32)

    train_images = train_images.astype("float32") / 255.0
    train_labels = to_categorical(train_labels)

    logging.info(f"Training dataset: {train_images.shape}, Labels: {train_labels.shape}")

    model, callbacks_list = cnn_model()
    logging.info("Starting training...")
    history = model.fit(train_images, train_labels, validation_split=0.2, epochs=100, batch_size=100, callbacks=callbacks_list)
    
    logging.info("Saving training plots...")
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('accuracy.png')
    plt.clf()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('loss.png')
    
    logging.info("Training complete.")

load_dataset()
train()
K.clear_session()
logging.info("Finished.")
