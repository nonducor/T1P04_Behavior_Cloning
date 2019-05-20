import random
from typing import Iterator, Tuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from util import load_data


def image_generator(data: pd.DataFrame,
                    batch_size: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Generator for image data.
    From a Pandas `DataFrame` containing the training data, reads the images and produce a stream of batches.
    Every line read might produce 4 images:
     - The original image
     - The original image flipped, matched with a flipped steering
     - The left camera image with a stronger right steering
     - The right camera image with a stronger left steering
    """
    mult_factor = 4            # For each line of data, we generate four cases
    steering_strength = 0.15   # Steering to add to the left and right images

    if batch_size % mult_factor != 0:
        raise Exception(f'Batch size has to be a multiple of {mult_factor}')

    images = []
    measurements = []
    index = 0
    while True:
        if index % len(data) == 0:
            data = data.sample(frac=1).reset_index(drop=True)  # Reshuffle
            index = 0
        line = data.loc[index]
        steering = line.steering

        # Process center image
        source_path_center = line['center']
        image = plt.imread(source_path_center)  # RGB
        images.append(image)
        measurements.append(steering)

        # Increase the training set with flipped images
        images.append(np.fliplr(image))
        measurements.append(-steering)

        # Increase the training set with left and right images
        source_path = line['right']
        image = plt.imread(source_path)  # RGB
        images.append(image)
        measurements.append(max(steering - steering_strength, -1.0))

        source_path = line['left']
        image = plt.imread(source_path)  # RGB
        images.append(image)
        measurements.append(min(steering + steering_strength, 1.0))

        index += 1

        if len(images) == batch_size:
            yield (np.array(images), np.array(measurements))
            images = []
            measurements = []


def split_validation(data: pd.DataFrame, split: float = 0.3, decimation: int = 5) -> Tuple[pd.DataFrame, Tuple[np.ndarray, np.ndarray]]:
    """Generate a training and validation set from the data (with a split of `split`).

    The `decimation` parameter is used to compensate for the number of straight driving examples. For example, if it is 5,
    only 1 in every 5 case where steering is between -0.25 and 0.25 will be selected. This impacts both the training
    and validation sets.
    """
    # Decimate the data near 0
    near_zero = (data.steering >= -0.25) & (data.steering <= 0.25)
    non_near_zero_data = data[~near_zero]
    near_zero_data = data[near_zero]
    data = pd.concat((non_near_zero_data, near_zero_data.sample(frac=1 / decimation)))

    # Split the two sets
    mask = np.random.randn(len(data)) < (1 - split)
    training_data = data[mask]
    validation_data = data[~mask]

    # Load the data for the validation set
    val_images = []
    val_steering = []
    for _, row in validation_data.iterrows():
        image = plt.imread(row.center)
        val_images.append(image)
        val_steering.append(row.steering)
    X_validation = np.array(val_images)
    y_validation = np.array(val_steering)
    return training_data, (X_validation, y_validation)


all_data = load_data()
training_data, (X_validation, y_validation) = split_validation(all_data)


print('\n')
print(f'Total number of images: {len(all_data)}')
print(f'Total number of original training : {len(training_data)}')
print(f'Number of validation samples: {len(X_validation)}')
print('\n')

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Cropping2D, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Conv2D(6, (5, 5), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(6, (5, 5), activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(1))

# Stop training early and save the best model
early_stop = EarlyStopping(monitor='val_loss', patience=8, verbose=0, mode='min')
mcp_save = ModelCheckpoint('model.h5', save_best_only=True, monitor='val_loss', mode='min')

model.compile(loss='mse', optimizer='adam')
BATCH_SIZE = 32
model.fit_generator(image_generator(training_data, BATCH_SIZE),
                    steps_per_epoch=np.ceil(len(training_data) / BATCH_SIZE),
                    nb_epoch=50,
                    validation_data=(X_validation, y_validation),
                    callbacks=[early_stop, mcp_save])
