import sys

from keras.models import load_model

from dataset import load_cifar10


def test_model(model_path=None):

    # Load the dataset with augmentations
    start_time = time.time()
    ((generator_train, generator_test),
     (x_train, y_train), (x_test, y_test),
     (x_val, y_val)) = load_cifar10()

    model = load_model(model_path)

    loss = model.evaluate_generator(generator_test.flow(x_test, y_test))
    print('Loss was: %s' % loss)


if __name__ == '__main__':
    print(test_model(sys.argv[1]))
