from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

import numpy as np


def preprocessing(img):

    # Pad from 32x32 to 40x40
    padded_img = np.zeros((40, 40, 3), dtype=np.uint8)
    for i in range(3):
        padded_img[:, :, i] = np.pad(img[:, :, i], (4, 4), 'constant')
    img = padded_img

    # Crop random 32x32 segment
    x_offset = np.random.randint(8)
    y_offset = np.random.randint(8)
    img = img[x_offset:x_offset + 32, y_offset:y_offset + 32, :]

    # Apply cutout
    x_center = np.random.randint(32)
    y_center = np.random.randint(32)
    x_start = max(0, x_center - 16)
    x_end = min(32, x_center + 16)
    y_start = max(0, y_center - 16)
    y_end = min(32, y_center + 16)
    img[x_start:x_end, y_start:y_end, :] = np.zeros((x_end - x_start,
                                                    y_end - y_start, 3),
                                                    dtype=np.uint8)
    return img


def load_cifar10(val_size=0.1):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                      test_size=val_size,
                                                      random_state=0)

    generator_train = ImageDataGenerator(
        featurewise_center=True,
        samplewise_center=False,
        featurewise_std_normalization=True,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-6,
        rotation_range=20,
        width_shift_range=0.0,
        height_shift_range=0.0,
        shear_range=0.,
        zoom_range=0,
        channel_shift_range=0.,
        fill_mode='nearest',
        cval=0.,
        horizontal_flip=True,
        vertical_flip=False,
        rescale=None,
        preprocessing_function=preprocessing
    )

    generator_test = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True
    )

    generator_train.fit(x_train)
    generator_test.fit(x_train)

    return ((generator_train, generator_test), (x_train, y_train),
            (x_test, y_test), (x_val, y_val))


def multi_generator(generator, duplicate_y=2):

    for batch in generator:
        # dubplicate Y
        yield (batch[0], [batch[1]] * duplicate_y)
