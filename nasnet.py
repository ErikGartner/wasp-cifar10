from keras.layers import *
from keras.models import Model
from keras.layers.normalization import *
from keras.regularizers import *
import tensorflow as tf
from keras import backend as K


class DropPath(Layer):

    def __init__(self, keep_prob, **kwargs):
        super().__init__(**kwargs)
        self.keep_prob = keep_prob

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x):
        if keep_prob >= 1:
            return x

        batch_size = tf.shape(x)[0]
        noise_shape = [batch_size, 1, 1, 1]
        random_tensor = self.keep_prob
        random_tensor += tf.random_uniform(noise_shape, dtype=tf.float32)
        binary_tensor = tf.floor(random_tensor)
        x_drop = tf.div(x, self.keep_prob) * binary_tensor
        return K.in_train_phase(x_drop, alt=x)

    def compute_output_shape(self, input_shape):
        return input_shape


def _sep_layer(x, nbr_filters, kernel_size, weight_decay, strides=(1, 1),
               nbr_layers=2, keep_prob=1):

    for i in range(nbr_layers):
        x = Activation('relu')(x)
        x = SeparableConv2D(
            nbr_filters,
            kernel_size,
            padding='same',
            use_bias=False,
            pointwise_initializer='he_normal',
            depthwise_initializer='he_normal',
            pointwise_regularizer=l2(weight_decay),
            depthwise_regularizer=l2(weight_decay),
            strides=strides
        )(x)
        x = BatchNormalization()(x)

        # Only first uses a stride other than (1, 1)
        strides = (1, 1)
    x = _drop_path(x, keep_prob)
    return x


def _identity(x, nbr_filters, strides=(1, 1)):
    if strides[0] > 1 or int(x.shape[3]) != nbr_filters:
        x = Activation('relu')(x)
        x = Convolution2D(nbr_filters, (1, 1), strides=strides, padding='same')(x)
        x = BatchNormalization()(x)
    return x


def _avg_layer(x, pool_size, nbr_filters, strides=(1, 1), keep_prob=1):
    x = AveragePooling2D(pool_size, padding='same', strides=strides)(x)

    if int(x.shape[3]) != nbr_filters:
        x = Convolution2D(nbr_filters, (1, 1), strides=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)
    x = _drop_path(x, keep_prob)
    return x


def _max_layer(x, pool_size, nbr_filters, strides=(1, 1), keep_prob=1):
    x = MaxPooling2D(pool_size, padding='same', strides=strides)(x)

    if int(x.shape[3]) != nbr_filters:
        x = Convolution2D(nbr_filters, (1, 1), strides=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)
    x = _drop_path(x, keep_prob)
    return x


def _tf_pad_layer(x):
    """
    To build a Keras model all tensors must be Keras tensor, therefore we
    wrap this slicing operation.
    """
    pad_arr = [[0, 0], [0, 1], [0, 1], [0, 0]]
    x = tf.pad(x, pad_arr)[:, 1:, 1:, :]
    return x


def _tf_avg_pool(x, stride_spec):
    return tf.nn.avg_pool(x, [1, 1, 1, 1], stride_spec,
                          'VALID', data_format='NHWC')


def _factorized_reduction(x, nbr_filters, strides):
    assert nbr_filters % 2 == 0

    if strides[0] == 1:
        x = Convolution2D(nbr_filters, kernel_size=(1, 1), strides=strides, padding='same')(x)
        x = BatchNormalization()(x)
        return x

    stride_spec = [1, strides[0], strides[1], 1]

    path1 = Lambda(_tf_avg_pool, arguments={'stride_spec': stride_spec},
                   output_shape=lambda x: (x[0], x[1] // 2, x[2] // 2, x[3]))(x)

    path1 = Convolution2D(nbr_filters // 2, (1, 1), padding='same')(path1)

    path2 = Lambda(_tf_pad_layer, output_shape=lambda in_shape: in_shape)(x)
    path2 = Lambda(_tf_avg_pool, arguments={'stride_spec': stride_spec},
                   output_shape=lambda x: (x[0], x[1] // 2, x[2] // 2, x[3]))(path2)

    path2 = Convolution2D(nbr_filters // 2, (1, 1), padding='same')(path2)

    x = Concatenate(axis=3)([path1, path2])
    x = BatchNormalization()(x)
    return x


def _drop_path(x, keep_prob):
    x = DropPath(keep_prob=keep_prob)(x)
    return x


def _calc_drop_keep_prob(keep_prob, cell_nbr, total_cells, epoch_tensor,
                         max_epochs):
    """
    Scales the keep_prob with the layer number and epoch.
    """
    if keep_prob == 1:
        return 1

    prob = keep_prob

    layer_ratio = (cell_nbr + 1) / total_cells
    prob = 1 - layer_ratio * (1 - prob)
    current_ratio = (epoch_tensor / max_epochs)
    current_ratio = tf.cast(current_ratio, tf.float32)
    current_ratio = tf.minimum(1.0, current_ratio)
    prob = (1 - current_ratio * (1 - prob))
    return prob


def _concatenate_result(x_list):
    target_filters = int(x_list[-1].shape[3])

    for i in range(len(x_list)):
        x_list[i] = _factorized_reduction(x_list[i], target_filters, (1, 1))

    x = Concatenate(axis=-1)(x_list)
    return x


def _create_cell_base(x, x_1, nbr_filters):
    x_1 = _reduce_prev_layer(x, x_1, nbr_filters)

    x = Activation('relu')(x)
    x = Convolution2D(nbr_filters, (1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)

    return (x, x_1)


def _create_reduction_cell(x, x_1, nbr_filters, weight_decay, strides, keep_prob,
                           cell_nbr, total_cells, epoch_tensor, max_epochs):
    x,  x_1 = _create_cell_base(x, x_1, nbr_filters)

    dp_prob = _calc_drop_keep_prob(keep_prob, cell_nbr, total_cells,
                                   epoch_tensor, max_epochs)

    sep7x7 = _sep_layer(x_1, nbr_filters, (7, 7), weight_decay, strides=strides, keep_prob=dp_prob)
    sep5x5 = _sep_layer(x, nbr_filters, (5, 5), weight_decay, strides=strides, keep_prob=dp_prob)
    y1 = Add()([sep7x7, sep5x5])

    max3x3 = _max_layer(x, (3, 3), nbr_filters, strides=strides, keep_prob=dp_prob)
    sep7x7 = _sep_layer(x_1, nbr_filters, (7, 7), weight_decay, strides=strides, keep_prob=dp_prob)
    y2 = Add()([max3x3, sep7x7])

    avg3x3 = _avg_layer(x, (3, 3), nbr_filters, strides=strides, keep_prob=dp_prob)
    sep5x5 = _sep_layer(x_1, nbr_filters, (5, 5), weight_decay, strides=strides, keep_prob=dp_prob)
    y3 = Add()([avg3x3, sep5x5])

    max3x3 = _max_layer(x, (3, 3), nbr_filters, strides=strides, keep_prob=dp_prob)
    sep3x3 = _sep_layer(y1, nbr_filters, (3, 3), weight_decay, keep_prob=dp_prob)
    z1 = Add()([max3x3, sep3x3])

    # We only apply strides when working on x or x_1
    avg3x3 = _avg_layer(y1, (3, 3), nbr_filters, keep_prob=dp_prob)
    ident = _identity(y2, nbr_filters)
    z2 = Add()([avg3x3, ident])

    result = _concatenate_result([z1, z2, y3])
    return result, int(result.shape[3])


def _create_normal_cell(x, x_1, nbr_filters, weight_decay, keep_prob, cell_nbr,
                        total_cells, epoch_tensor, max_epochs):
    x,  x_1 = _create_cell_base(x, x_1, nbr_filters)

    dp_prob = _calc_drop_keep_prob(keep_prob, cell_nbr, total_cells,
                                   epoch_tensor, max_epochs)

    sep3x3 = _sep_layer(x, nbr_filters, (3, 3), weight_decay, keep_prob=dp_prob)
    ident = _identity(x, nbr_filters)
    y1 = Add()([sep3x3, ident])

    sep3x3 = _sep_layer(x_1, nbr_filters, (3, 3), weight_decay, keep_prob=dp_prob)
    sep5x5 = _sep_layer(x, nbr_filters, (5, 5), weight_decay, keep_prob=dp_prob)
    y2 = Add()([sep3x3, sep5x5])

    avg3x3 = _avg_layer(x, (3, 3), nbr_filters, keep_prob=dp_prob)
    ident = _identity(x_1, nbr_filters)
    y3 = Add()([avg3x3, ident])

    avg3x3_1 = _avg_layer(x_1, (3, 3), nbr_filters, keep_prob=dp_prob)
    avg3x3_2 = _avg_layer(x_1, (3, 3), nbr_filters, keep_prob=dp_prob)
    y4 = Add()([avg3x3_1, avg3x3_2])

    sep5x5 = _sep_layer(x_1, nbr_filters, (5, 5), weight_decay, keep_prob=dp_prob)
    sep3x3 = _sep_layer(x_1, nbr_filters, (3, 3), weight_decay, keep_prob=dp_prob)
    y5 = Add()([sep5x5, sep3x3])

    result = _concatenate_result([y1, y2, y3, y4, y5])
    return result, int(result.shape[3])


def _reduce_prev_layer(x, x_1, nbr_filters):

    if x_1 is None:
        x_1 = x

    if int(x.shape[2]) != int(x_1.shape[2]):
        x_1 = Activation('relu')(x_1)
        x_1 = _factorized_reduction(x_1, nbr_filters, strides=(2, 2))

    elif int(x.shape[3]) != int(x_1.shape[3]):
        # Mismatch in nbr of filters
        x_1 = Activation('relu')(x_1)
        x_1 = Convolution2D(nbr_filters, kernel_size=(1, 1), strides=(1, 1),
                            padding='same')(x_1)
        x_1 = BatchNormalization()(x_1)

    return x_1


def _create_auxhead(x, nbr_classes, final_filters=768):
    """
    Aux head creates an auxiliary loss in the network to help training.
    The loss is weighted by 0.4 in the paper and placed before the 2nd reduction
    layer.
    """
    x = Activation('relu')(x)
    x = AveragePooling2D((5, 5), strides=(3, 3), padding='valid')(x)
    x = Convolution2D(128, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(final_filters, (int(x.shape[1]), int(x.shape[2])), strides=(1, 1), padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Flatten()(x)
    x = Dense(nbr_classes, activation='softmax', kernel_initializer='he_normal')(x)
    return x


def _create_head(x, nbr_classes, dropout_prob=0.0):
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    if dropout_prob > 0:
        x = Dropout(dropout_prob)(x)
    x = Dense(nbr_classes, activation='softmax', kernel_initializer='he_normal')(x)
    return x


def _create_stem(x, nbr_filters, stem_multiplier):
    x = Convolution2D(nbr_filters * stem_multiplier, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    return x


def create_nasnet(input_shape, nbr_normal_cells, nbr_blocks, weight_decay,
                  nbr_classes, nbr_filters, stem_multiplier, filter_multiplier,
                  dimension_reduction, final_filters, max_epochs, dropout_prob,
                  drop_path_keep, epoch_tensor):

    ipt = Input(input_shape)
    x = _create_stem(ipt, nbr_filters, stem_multiplier)
    x_1 = None

    cell_nbr = 0
    filters = nbr_filters
    for i in range(nbr_blocks):
        for j in range(nbr_normal_cells):
            cell_nbr += 1
            y, _ = _create_normal_cell(x, x_1, filters, weight_decay, drop_path_keep,
                                       cell_nbr, nbr_normal_cells * nbr_blocks, epoch_tensor,
                                       max_epochs)
            x_1 = x
            x = y

        if i == 1:
            # Before the second reduction cell, add the auxiliary head
            aux_head = _create_auxhead(x, nbr_classes, final_filters)

        # Reduction cell decreases HxW but increases filters
        filters = filters * filter_multiplier
        y, _ = _create_reduction_cell(x, x_1, filters, weight_decay,
                                      (dimension_reduction, dimension_reduction),
                                      drop_path_keep, cell_nbr, nbr_normal_cells * nbr_blocks,
                                      epoch_tensor, max_epochs)
        x_1 = x
        x = y

    y = _create_head(x, nbr_classes, dropout_prob)
    model = Model(inputs=ipt, outputs=[y, aux_head], name='NASNet')
    return model


if __name__ == '__main__':
    model = create_nasnet(input_shape=(32, 32, 3),
                          nbr_normal_cells=6,
                          nbr_blocks=2,
                          weight_decay=1e-4,
                          nbr_classes=10,
                          nbr_filters=32,
                          stem_multiplier=3,
                          filter_multiplier=2,
                          dimension_reduction=2,
                          final_filters=768,
                          dropout_prob=0.0,
                          drop_path_keep=0.6,
                          max_epochs=300,
                          epoch_tensor=tf.Variable(1, dtype=tf.int32, trainable=False))
    print(model.summary())
