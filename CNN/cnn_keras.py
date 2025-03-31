import numpy as np
import pickle
import cv2
import os
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import load_images  # Ensure this module exists and is correct

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

# Dataset Directory
directory = "/workspaces/KannadaHandwritingRecognition/dataset/dataset/Kannada/Hnd/Img"

# Get Image Dimensions
def get_image_size():
    img_path = os.path.join(directory, '1', '1.png')
    
    if not os.path.exists(img_path):
        print(f"[ERROR] Sample image '{img_path}' not found. Check dataset path!")
        exit(1)
    
    img = cv2.imread(img_path, 0)
    print(f"[INFO] Image dimensions: {img.shape}")
    return img.shape

# Get Number of Classes
def get_num_of_classes():
    num_classes = len(os.listdir(directory))
    print(f"[INFO] Number of classes: {num_classes}")
    return num_classes

# Model Architecture
def cnn_model():
    num_of_classes = get_num_of_classes()
    model = Sequential([
        Conv2D(52, (5, 5), input_shape=(image_x, image_y, 1), activation='tanh'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.Lambda(lambda x: tf.nn.local_response_normalization(x)),
        Conv2D(64, (5, 5), activation='tanh'),
        MaxPooling2D(pool_size=(5, 5), strides=(5, 5)),
        Flatten(),
        Dropout(0.5),
        Dense(num_of_classes, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(learning_rate=0.001),
                  metrics=['accuracy'])

    filepath = "cnn_model.h5"
    checkpoint1 = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    return model, [checkpoint1]

# Train Model
def train():
    print("[INFO] Loading dataset from pickle files...")

    # Check if pickle files exist
    for file in ["train_images", "train_labels", "test_images", "test_labels"]:
        if not os.path.exists(file):
            print(f"[ERROR] Missing pickle file: {file}. Run `load_images.create_pickle(directory)` first.")
            exit(1)

    # Load Pickle Data
    with open("train_images", "rb") as f:
        train_images = np.array(pickle.load(f))
    with open("train_labels", "rb") as f:
        train_labels = np.array(pickle.load(f), dtype=np.int32)
    with open("test_images", "rb") as f:
        test_images = np.array(pickle.load(f))
    with open("test_labels", "rb") as f:
        test_labels = np.array(pickle.load(f), dtype=np.int32)

    print(f"[INFO] Training Images: {train_images.shape}, Labels: {train_labels.shape}")
    print(f"[INFO] Test Images: {test_images.shape}, Labels: {test_labels.shape}")

    # Reshape for CNN
    train_images = np.reshape(train_images, (train_images.shape[0], image_x, image_y, 1))
    test_images = np.reshape(test_images, (test_images.shape[0], image_x, image_y, 1))

    # Convert Labels to Categorical
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    print(f"[INFO] Reshaped Train Images: {train_images.shape}")
    print(f"[INFO] Reshaped Test Images: {test_images.shape}")

    # Get Model
    model, callbacks_list = cnn_model()

    print("[INFO] Starting training...")
    history = model.fit(
        train_images, train_labels,
        validation_data=(test_images, test_labels),
        epochs=50,  # Reduced to 50
        batch_size=32,  # Reduced batch size
        callbacks=callbacks_list
    )

    # Evaluate Model
    scores = model.evaluate(test_images, test_labels, verbose=0)
    print(f"[INFO] CNN Test Accuracy: {scores[1] * 100:.2f}%")

    # Plot Accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('acc.png')
    plt.clf()

    # Plot Loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('loss.png')

    print("[INFO] Training complete. Model saved as 'cnn_model.h5'.")

# Run Dataset Processing
print("[INFO] Creating dataset pickle files...")
load_images.create_pickle(directory)

# Get Image Size
image_x, image_y = get_image_size()

# Start Training
train()

# Clear TensorFlow Session
tf.keras.backend.clear_session()
print("[INFO] Session cleared. Process complete!")
