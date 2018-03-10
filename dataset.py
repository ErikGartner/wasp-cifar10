from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split


def load_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=0)

    generator_train = ImageDataGenerator(
        featurewise_center=True,
        samplewise_center=False,
        featurewise_std_normalization=True,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-6,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.,
        zoom_range=0,
        channel_shift_range=0.,
        fill_mode='nearest',
        cval=0.,
        horizontal_flip=True,
        vertical_flip=False,
        rescale=None,
        preprocessing_function=None)

    generator_test = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True)

    generator_train.fit(x_train)
    generator_test.fit(x_train)
    return ((generator_train, generator_test), (x_train, y_train), (x_test, y_test), (x_val, y_val))
