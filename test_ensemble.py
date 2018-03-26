import sys
import time
import os

import numpy as np

from keras.models import load_model
from keras.optimizers import *
from keras.metrics import *

from dataset import load_cifar10


def init_model(model_path):

    # Load the dataset with augmentations
    start_time = time.time()

    model = load_model(model_path)

    optimizer = SGD(lr=0.1, momentum=0.9, nesterov=True)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])
    return model


def predict_models(models, x):
    predictions = list(map(lambda model: model.predict(x)))

    for i, pred in enumerate(predictions):
        predictions[i] = np.asarray(pred).reshape(len(x), 10, 1)
    weighted_avg = np.mean(predictions, axis=2)
    votes = np.argmax(weighted_avg, axis=1)
    return Y


if __name__ == '__main__':

    path = sys.argv[1]
    print("Loading all models in %s" % path)

    models = []
    for model_file in os.listdir(path):

        try:
            print('Loading %s' % model_file)
            model = init_model(os.path.join(path, model_file))
            models.append(model)

            models = models * 2
            break
        except RuntimeError:
            print('Some error occured!')

    # Evaluate using ensemble:
    ((generator_train, generator_test),
     (x_train, y_train), (x_test, y_test),
     (x_val, y_val)) = load_cifar10()

    # predict using ensabmle:
    y = predict_models(models, x_test)

    print(sparse_categorical_accuracy(y_test, y))