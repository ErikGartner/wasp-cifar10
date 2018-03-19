from datetime import datetime

from keras.optimizers import *
from keras.callbacks import *
from keras.utils import multi_gpu_model
from keras import backend as K
import tensorflow as tf

from nasnet import create_nasnet
from dataset import load_cifar10, multi_generator


def create_callbacks(max_epochs, run_dir, start_lr,
                     lr_decrease_factor=0.5, lr_patience=10,
                     model=None, epoch_tensor=None, ):

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

    def update_epoch_tensor(epoch, logs):
        if epoch_tensor is not None:
            tf.assign(epoch_tensor, epoch + 1, name='update_epoch_tensor')

    def cosine_decay(epoch, lr):
        return tf.train.noisy_linear_cosine_decay(start_lr, epoch + 1, max_epochs)

    checkpointing = MultiGPUCheckpoint(filepath='./weights/weights_%s_.{epoch:02d}-{val_dense_2_acc:.3f}.ckpt' % run_dir,
                                       verbose=1, period=1, save_best_only=True)
    checkpointing.model = model

    cbs = []
    #cbs.append(ReduceLROnPlateau(monitor='val_loss', factor=lr_decrease_factor,
    #                             verbose=1, min_lr=1e-7, patience=lr_patience))
    cbs.append(LearningRateScheduler(cosine_decay, verbose=1))
    cbs.append(TensorBoard(log_dir='./logs/%s' % run_dir, batch_size=64))
    cbs.append(LambdaCallback(on_epoch_begin=update_epoch_tensor))
    cbs.append(checkpointing)
    return cbs


def train_model(max_epochs=300, start_lr=0.025, drop_path_keep=0.6,
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

    # Create current epoch holding tensor
    epoch_tensor = tf.Variable(initial_epoch, dtype=tf.int32, trainable=False)

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
                                           final_filters=768,
                                           dropout_prob=0.0,
                                           drop_path_keep=drop_path_keep,
                                           max_epochs=max_epochs,
                                           epoch_tensor=epoch_tensor)
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
                                       final_filters=768,
                                       dropout_prob=0.0,
                                       drop_path_keep=drop_path_keep,
                                       max_epochs=max_epochs,
                                       epoch_tensor=epoch_tensor)
        model = orig_model

    # Setup optimizer
    optimizer = SGD(lr=start_lr, momentum=0.9, nesterov=True, clipnorm=5.0)

    cbs = create_callbacks(max_epochs, run_dir, start_lr,lr_decrease_factor,
                           lr_patience, orig_model, epoch_tensor)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  loss_weights=[1, 0.4],    # Weight the auxiliary head by 0.4
                  metrics=['accuracy'])

    # Setup the multi output generators
    train = generator_train.flow(x_train, y_train, batch_size=batch_size, seed=0)
    test = generator_test.flow(x_val, y_val, batch_size=batch_size, seed=0)
    mul_train = multi_generator(train)
    mul_test = multi_generator(test)
    steps_per_epoch = len(train)
    validation_steps = len(test)

    # Start training
    history = model.fit_generator(
        mul_train,
        callbacks=cbs, epochs=max_epochs,
        validation_data=mul_test,
        use_multiprocessing=False, max_queue_size=batch_size,
        verbose=1, initial_epoch=initial_epoch, steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps
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
