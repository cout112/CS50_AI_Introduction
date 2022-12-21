import cv2
import numpy as np
import pandas as pd
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 20
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # We test all the combination of parameters a number of times to get the mean
    for i in range(1):
            

        # Split data into training and testing sets
        labels = tf.keras.utils.to_categorical(labels)       
        x_train, x_test, y_train, y_test = train_test_split(
            np.array(images), np.array(labels), test_size=TEST_SIZE
        )
        # x_train: images to train
        # y_train: labels to train
        # x_test: images to evaluate
        # y_test: labels to evaluate

        # Get a compiled neural network
        model = get_model()

        # Fit model on training data
        model.fit(x_train, y_train, epochs=EPOCHS)

        # Evaluate neural network performance
        model.evaluate(x_test,  y_test, verbose=2)

        print(model)

        # Save model to file
        if len(sys.argv) == 3:
            filename = sys.argv[2]
            model.save(filename)
            print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    folders = os.listdir(data_dir)
    dimension = ( IMG_WIDTH, IMG_HEIGHT )
    images = []
    labels = []
    for folder in folders:
        if folder.startswith('.'):
            continue
        image_names = os.listdir(os.path.join(data_dir, folder))
        for image_name in image_names:
            image = cv2.imread(os.path.join(data_dir, folder, image_name))
            image = cv2.resize(image, dimension, interpolation = cv2.INTER_AREA)
            images.append(image)
            labels.append(folder)
    return images, labels


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    

    layers = [  tf.keras.layers.Conv2D(100, 3, activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
                tf.keras.layers.MaxPooling2D((2,2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1000, activation='relu'),
                tf.keras.layers.Dense(1000, activation='relu'),
                tf.keras.layers.Dense(1000, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax')
                ]

    model = tf.keras.models.Sequential(layers)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )


    return model


if __name__ == "__main__":
    main()
