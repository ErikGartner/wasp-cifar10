import sys
import time
import os

from keras.models import load_model
from keras.optimizers import *

from dataset import load_cifar10


def test_model(model_path=None):

    # Load the dataset with augmentations
    start_time = time.time()
    ((generator_train, generator_test),
     (x_train, y_train), (x_test, y_test),
     (x_val, y_val)) = load_cifar10()

    model = load_model(model_path)

    if not model.built:
        optimizer = SGD(lr=0.1, momentum=0.9, nesterov=True)
        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['acc'])

    loss = model.evaluate_generator(generator_test.flow(x_test, y_test))
    print('Loss was: %s' % loss)
    return loss


if __name__ == '__main__':

    path = sys.argv[1]

    if os.path.isdir(path):
        print("Testing all models in %s" % path)
        for model_file in os.listdir(path):

            try:
                print('Testing %s' % model_file)
                test_model(os.path.join(path, model_file))
            except:
                print('Some error occured!')

    else:
        test_model(path)