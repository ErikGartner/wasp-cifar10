import sys
import time
import os

import numpy as np

from keras.models import load_model
from keras.optimizers import *
from keras.metrics import *

from dataset import load_cifar10


def init_model(model_path):

    model = load_model(model_path)

    optimizer = SGD(lr=0.1, momentum=0.9, nesterov=True)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])
    return model


def predict_models(models, x):
    predictions = list(map(lambda model:
                           model.predict_on_batch(x),
                           models))
    nb_preds = len(x)
    for i, pred in enumerate(predictions):
        pred = list(map(
            lambda probas: np.argmax(probas, axis=-1), pred
        ))
        predictions[i] = np.asarray(pred).reshape(nb_preds, 1)
    argmax_list = list(np.concatenate(predictions, axis=1))
    votes = np.asarray(list(
        map(lambda arr: max(set(arr)), argmax_list)
    ))
    return votes


if __name__ == '__main__':

    path = sys.argv[1]
    print("Loading all models in %s" % path)

    models = []
    for model_file in os.listdir(path):

        try:
            print('Loading %s' % model_file)
            model = init_model(os.path.join(path, model_file))
            models.append(model)

        except RuntimeError:
            print('Some error occured!')

    # Evaluate using ensemble:
    ((generator_train, generator_test),
     (x_train, y_train), (x_test, y_test),
     (x_val, y_val)) = load_cifar10()

    # Evaluate the models
    correct = 0
    total = 0
    for x_batch, y_batch in generator_test.flow(x_test, y_test, batch_size=32):
        print('%d/%d' % (total, len(y_test)))
        total += len(y_batch)

        y = predict_models([model], x_batch)
        correct += np.sum(y.flatten() == y_batch.flatten())

        if total >= len(y_test):
            break

    print('Correct: %d/%d (%f)' % (correct, total, correct / total))
