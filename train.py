from datetime import datetime
import os
import json

from keras.optimizers import *
from keras.callbacks import *
from keras.utils import multi_gpu_model
from keras import backend as K

from densenet import create_densenet
from dataset import load_cifar10


def create_callbacks(max_epochs, run_dir, lr_decrease_factor=0.5, lr_patience=10):
    cbs = []
    cbs.append(ReduceLROnPlateau(monitor='val_loss', factor=lr_decrease_factor,
                                 verbose=1, min_lr=1e-6, patience=lr_patience))
    cbs.append(TensorBoard(log_dir='./logs/%s' % run_dir, batch_size=64))
    cbs.append(ModelCheckpoint(
        filepath='./weights/weights_%s_.{epoch:02d}-{val_acc:.2f}.ckpt' % run_dir,
        verbose=1, period=1, save_best_only=True))
    return cbs


def dump_infomation(dump_dir, model):
    if dump_dir is None:
        return

    with open(os.path.join(dump_dir, 'model.json'), 'w') as f:
        json.dump(model.to_json(), f, indent=2)

    with open(os.path.join(dump_dir, 'model.txt'), 'w') as f:
        f.write(model.summary())


def train_model(max_epochs=300, optimizer=SGD(lr=0.1, momentum=0.9, nesterov=True),
                dense_layers=[13, 13, 13], growth_rate=40, compression=0.5,
                dropout=0.2, weight_decay=1e-4, batch_size=64, logdir='./logs',
                weightsdir='./weights', lr_decrease_factor=0.5, lr_patience=10):

    start_time = time.time()
    ((generator_train, generator_test),
     (x_train, y_train), (x_test, y_test),
     (x_val, y_val)) = load_cifar10()

    model = create_densenet(
        input_shape=(32, 32, 3), dense_layers=dense_layers,
        growth_rate=growth_rate, nbr_classes=10, weight_decay=weight_decay,
        compression=compression, dropout=dropout
    )

    run_dir = datetime.today().strftime('%Y%m%d-%H%M%S')
    dump_infomation(os.path.join(logdir, dump_dir), model)
    cbs = create_callbacks(max_epochs)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])

    history = model.fit_generator(
        generator_train.flow(x_train, y_train, batch_size=batch_size, seed=0),
        callbacks=cbs, epochs=max_epochs,
        validation_data=generator_test.flow(x_val, y_val, seed=0),
        verbose=1
    )

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
