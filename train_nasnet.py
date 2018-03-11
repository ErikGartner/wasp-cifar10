from datetime import datetime

from keras.optimizers import *
from keras.callbacks import *
from keras.utils import multi_gpu_model
from keras import backend as K

from nasnet import create_nasnet
from dataset import load_cifar10


def create_callbacks(max_epochs):
    cbs = []

    def learningrate_schedule(epoch, lr):
        if epoch == int(max_epochs*0.5) or epoch == int(max_epochs*0.75):
            return lr*0.1
        else:
            return lr

    run_dir = datetime.today().strftime('%Y%m%d-%H%M%S')
    cbs.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=1, min_lr=1e-6, patience=10))
    cbs.append(TensorBoard(log_dir='./logs/%s' % run_dir, batch_size=64))
    cbs.append(ModelCheckpoint(filepath='./weights/weights_%s_.{epoch:02d}-{val_acc:.2f}.ckpt' % run_dir, verbose=1, period=1, save_best_only=True))
    return cbs


def train_model(max_epochs=300, optimizer=SGD(lr=0.1, momentum=0.9, nesterov=True),
                nbr_blocks=2, weight_decay=1e-4, nbr_filters=128, batch_size=64):

    start_time = time.time()

    (generator_train, generator_test), (x_train, y_train), (x_test, y_test), (x_val, y_val) = load_cifar10()
    model = create_nasnet(input_shape=(32, 32, 3),
                          nbr_normal_cells=6,
                          nbr_blocks=nbr_blocks,
                          weight_decay=weight_decay,
                          nbr_classes=10,
                          nbr_filters=nbr_filters)

    cbs = create_callbacks(max_epochs)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['acc'])

    history = model.fit_generator(generator_train.flow(x_train, y_train, batch_size=batch_size, seed=0),
                        callbacks=cbs, epochs=max_epochs,
                        validation_data=generator_test.flow(x_val, y_val, seed=0),
                        verbose=1)


    best_val_acc = max(history.history['val_acc'])
    best_acc = max(history.history['acc'])
    return {
        'loss': -1 * best_acc,
        'true_loss': -1 * best_val_acc,
        'status': 'ok',
        'eval_time': time.time() - start_time,
    }


if __name__ == '__main__':
    print(train_model())
