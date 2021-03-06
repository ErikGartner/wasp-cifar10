from datetime import datetime
import os
import json

import tensorflow as tf
from keras.optimizers import *
from keras.callbacks import *
from keras.utils import multi_gpu_model
from keras import backend as K
from keras.models import load_model

from densenet import create_densenet
from dataset import load_cifar10


def create_callbacks(max_epochs, run_dir, start_lr, lr_decrease_factor=0.5,
                     lr_patience=10, model=None):

    class MultiGPUCheckpoint(ModelCheckpoint):

        def __init__(self, filepath, monitor='val_acc', verbose=0,
                     save_best_only=False, save_weights_only=False,
                     mode='auto', period=1):
            super().__init__(filepath, monitor, verbose,
                     save_best_only, save_weights_only,
                     mode, period)

        def set_model(self, model):
            if model is not None:
                # Disable changing model
                return
            else:
                super().set_model(model)

    def cosine_decay(epoch, lr):
        lr = tf.train.linear_cosine_decay(start_lr, epoch + 1, max_epochs,
                                          num_periods=0.5, # 0.5 means no restart.
                                          alpha=0.0,
                                          beta=0.0,
                                          name=None)
        lr = K.eval(lr)
        return lr

    cbs = []
    checkpointing = MultiGPUCheckpoint(filepath='./weights/weights_%s_.{epoch:02d}-{val_acc:.3f}.ckpt' % run_dir,
                                       verbose=1, period=1, save_best_only=True)
    checkpointing.model = model
    cbs.append(LearningRateScheduler(cosine_decay, verbose=1))
    #cbs.append(ReduceLROnPlateau(monitor='val_loss', factor=lr_decrease_factor,
    #                             verbose=1, min_lr=1e-7, patience=lr_patience))
    cbs.append(TensorBoard(log_dir='./logs/%s' % run_dir, batch_size=64))
    cbs.append(checkpointing)
    return cbs


def dump_infomation(dump_dir, model, dense_layers, growth_rate, compression,
                    dropout, weight_decay, batch_size):
    if dump_dir is None:
        return

    os.makedirs(dump_dir, exist_ok=True)
    with open(os.path.join(dump_dir, 'model.json'), 'w') as f:
        json.dump(model.to_json(), f, indent=2)

    def print_to_file(s):
        with open(os.path.join(dump_dir, 'model.txt'), 'a') as f:
            print(s, file=f)
    model.summary(print_fn=print_to_file)

    with open(os.path.join(dump_dir, 'params.txt'), 'w') as f:
        f.write('dense_layers: %s\n' % dense_layers)
        f.write('growth_rate: %s\n' % growth_rate)
        f.write('compression: %s\n' % compression)
        f.write('dropout: %s\n' % dropout)
        f.write('weight_decay: %s\n' % weight_decay)
        f.write('batch_size: %s\n' % batch_size)


def train_model(max_epochs=300, start_lr=0.1,
                dense_layers=[20, 20, 20], growth_rate=60, compression=0.5,
                dropout=0.0, weight_decay=1e-4, batch_size=64, logdir='./logs',
                weightsdir='./weights', lr_decrease_factor=0.5, lr_patience=10,
                nbr_gpus=1, model_path=None, initial_epoch=0):
    # Create a dir in the logs catalog and dump info
    run_dir = datetime.today().strftime('%Y%m%d-%H%M%S-%f')

    # Load the dataset with augmentations
    start_time = time.time()
    ((generator_train, generator_test),
     (x_train, y_train), (x_test, y_test),
     (x_val, y_val)) = load_cifar10()

    # Create model using supplied params
    # Load model from file if the argument model_path is supplied.
    # Use mutli_gpu setup if enabled
    if nbr_gpus > 1:
        with tf.device('/cpu:0'):
            if model_path is not None:
                orig_model = load_model(model_path)
            else:
                orig_model = create_densenet(
                    input_shape=(32, 32, 3), dense_layers=dense_layers,
                    growth_rate=growth_rate, nbr_classes=10, weight_decay=weight_decay,
                    compression=compression, dropout=dropout
                )
        model = multi_gpu_model(orig_model, nbr_gpus)

    else:
        if model_path is not None:
            orig_model = load_model(model_path)
        else:
            orig_model = create_densenet(
                input_shape=(32, 32, 3), dense_layers=dense_layers,
                growth_rate=growth_rate, nbr_classes=10, weight_decay=weight_decay,
                compression=compression, dropout=dropout
            )
        model = orig_model

    # Write model info to file
    dump_infomation(os.path.join(logdir, run_dir), orig_model, dense_layers,
                    growth_rate, compression, dropout, weight_decay,
                    batch_size)

    # Setup optimizer
    optimizer = SGD(lr=start_lr, momentum=0.9, nesterov=True)

    cbs = create_callbacks(max_epochs, run_dir, start_lr, lr_decrease_factor,
                           lr_patience, orig_model)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])

    history = model.fit_generator(
        generator_train.flow(x_train, y_train, batch_size=batch_size, seed=0),
        callbacks=cbs, epochs=max_epochs,
        validation_data=generator_test.flow(x_val, y_val, seed=0),
        use_multiprocessing=True, workers=2, max_queue_size=batch_size,
        verbose=1, initial_epoch=initial_epoch
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
    K.set_image_data_format('channels_last')
    print(train_model())
