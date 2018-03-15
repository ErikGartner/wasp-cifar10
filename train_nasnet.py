from datetime import datetime

from keras.optimizers import *
from keras.callbacks import *
from keras.utils import multi_gpu_model
from keras import backend as K

from nasnet import create_nasnet
from dataset import load_cifar10


def create_callbacks(max_epochs, run_dir, lr_decrease_factor=0.5, lr_patience=10,
                     model=None):

    class MultiGPUCheckpoint(ModelCheckpoint):

        def __init__(self, filepath, monitor='val_loss', verbose=0,
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

    cbs = []
    checkpointing = MultiGPUCheckpoint(filepath='./weights/weights_%s_.{epoch:02d}-{val_acc:.2f}.ckpt' % run_dir,
                                       verbose=1, period=1, save_best_only=True)
    checkpointing.model = model
    cbs.append(ReduceLROnPlateau(monitor='val_loss', factor=lr_decrease_factor,
                                 verbose=1, min_lr=1e-7, patience=lr_patience))
    cbs.append(TensorBoard(log_dir='./logs/%s' % run_dir, batch_size=64))
    cbs.append(checkpointing)
    return cbs


def train_model(max_epochs=300, start_lr=0.025,
                nbr_blocks=2, weight_decay=1e-4, nbr_filters=32, batch_size=32,
                logdir='./logs', weightsdir='./weights_nasnet', lr_decrease_factor=0.5,
                lr_patience=10, nbr_gpus=1, model_path=None, initial_epoch=0):

    # Create a dir in the logs catalog and dump info
    run_dir = 'nasnet_%s' % datetime.today().strftime('%Y%m%d-%H%M%S-%f')

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
                orig_model = create_nasnet(input_shape=(32, 32, 3),
                                           nbr_normal_cells=6,
                                           nbr_blocks=nbr_blocks,
                                           weight_decay=weight_decay,
                                           nbr_classes=10,
                                           nbr_filters=nbr_filters,
                                           stem_multiplier=3,
                                           filter_multiplier=2,
                                           dimension_reduction=2,
                                           final_filters=768)
        model = multi_gpu_model(orig_model, nbr_gpus)

    else:
        if model_path is not None:
            orig_model = load_model(model_path)
        else:
            orig_model = create_nasnet(input_shape=(32, 32, 3),
                                       nbr_normal_cells=6,
                                       nbr_blocks=nbr_blocks,
                                       weight_decay=weight_decay,
                                       nbr_classes=10,
                                       nbr_filters=nbr_filters,
                                       stem_multiplier=3,
                                       filter_multiplier=2,
                                       dimension_reduction=2,
                                       final_filters=768)
        model = orig_model

    # # Write model info to file
    # dump_infomation(os.path.join(logdir, run_dir), orig_model, dense_layers,
    #                 growth_rate, compression, dropout, weight_decay,
    #                 batch_size)

    # Setup optimizer
    optimizer = SGD(lr=start_lr, momentum=0.9, nesterov=True, clipnorm=5.0)

    cbs = create_callbacks(max_epochs, run_dir, lr_decrease_factor, lr_patience, orig_model)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  loss_weights=[1, 0.4],    # Weight the auxiliary head by 0.4
                  metrics=['accuracy'])

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
