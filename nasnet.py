from keras.layers import *
from keras.models import Model
from keras.layers.normalization import *
from keras.regularizers import *
import tensorflow as tf


def _sep_layer(x, nbr_filters, kernel_size, weight_decay, strides=(1, 1),
               nbr_layers=2):

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
    return x


def _identity(x, nbr_filters, strides=(1, 1)):
    if strides[0] > 1 or int(x.shape[3]) != nbr_filters:
        x = Activation('relu')(x)
        x = Convolution2D(nbr_filters, (1, 1), strides=strides, padding='same')(x)
        x = BatchNormalization()(x)
    return x


def _avg_layer(x, pool_size, nbr_filters, strides=(1, 1)):
    x = AveragePooling2D(pool_size, padding='same', strides=strides)(x)

    if int(x.shape[3]) != nbr_filters:
        x = Convolution2D(nbr_filters, (1, 1), strides=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)
    return x


def _max_layer(x, pool_size, nbr_filters, strides=(1, 1)):
    x = MaxPooling2D(pool_size, padding='same', strides=strides)(x)

    if int(x.shape[3]) != nbr_filters:
        x = Convolution2D(nbr_filters, (1, 1), strides=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)
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


def _create_reduction_cell(x, x_1, nbr_filters, weight_decay, strides):
    x,  x_1 = _create_cell_base(x, x_1, nbr_filters)

    sep7x7 = _sep_layer(x_1, nbr_filters, (7, 7), weight_decay, strides=strides)
    sep5x5 = _sep_layer(x, nbr_filters, (5, 5), weight_decay, strides=strides)
    y1 = Add()([sep7x7, sep5x5])

    max3x3 = _max_layer(x, (3, 3), nbr_filters, strides=strides)
    sep7x7 = _sep_layer(x_1, nbr_filters, (7, 7), weight_decay, strides=strides)
    y2 = Add()([max3x3, sep7x7])

    avg3x3 = _avg_layer(x, (3, 3), nbr_filters, strides=strides)
    sep5x5 = _sep_layer(x_1, nbr_filters, (5, 5), weight_decay, strides=strides)
    y3 = Add()([avg3x3, sep5x5])

    max3x3 = _max_layer(x, (3, 3), nbr_filters, strides=strides)
    sep3x3 = _sep_layer(y1, nbr_filters, (3, 3), weight_decay)
    z1 = Add()([max3x3, sep3x3])

    # We only apply strides when working on x or x_1
    avg3x3 = _avg_layer(y1, (3, 3), nbr_filters)
    ident = _identity(y2, nbr_filters)
    z2 = Add()([avg3x3, ident])

    result = _concatenate_result([z1, z2, y3])
    return result, int(result.shape[3])


def _create_normal_cell(x, x_1, nbr_filters, weight_decay):
    x,  x_1 = _create_cell_base(x, x_1, nbr_filters)

    sep3x3 = _sep_layer(x, nbr_filters, (3, 3), weight_decay)
    ident = _identity(x, nbr_filters)
    y1 = Add()([sep3x3, ident])

    sep3x3 = _sep_layer(x_1, nbr_filters, (3, 3), weight_decay)
    sep5x5 = _sep_layer(x, nbr_filters, (5, 5), weight_decay)
    y2 = Add()([sep3x3, sep5x5])

    avg3x3 = _avg_layer(x, (3, 3), nbr_filters)
    ident = _identity(x_1, nbr_filters)
    y3 = Add()([avg3x3, ident])

    avg3x3_1 = _avg_layer(x_1, (3, 3), nbr_filters)
    avg3x3_2 = _avg_layer(x_1, (3, 3), nbr_filters)
    y4 = Add()([avg3x3_1, avg3x3_2])

    sep5x5 = _sep_layer(x_1, nbr_filters, (5, 5), weight_decay)
    sep3x3 = _sep_layer(x_1, nbr_filters, (3, 3), weight_decay)
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


def _create_head(x, nbr_classes, final_filters):
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


def _create_steam(x, nbr_filters, stem_multiplier):
    x = Convolution2D(nbr_filters * stem_multiplier, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    return x


def create_nasnet(input_shape, nbr_normal_cells, nbr_blocks, weight_decay,
                  nbr_classes, nbr_filters, stem_multiplier, filter_multiplier,
                  dimension_reduction, final_filters):

    ipt = Input(input_shape)
    x = _create_steam(ipt, nbr_filters, stem_multiplier)
    x_1 = None

    filters = nbr_filters
    for i in range(nbr_blocks):

        for j in range(nbr_normal_cells):
            y, _ = _create_normal_cell(x, x_1, filters,
                                             weight_decay)
            x_1 = x
            x = y

        # Reduction cell decreases HxW but increases filters
        filters = filters * filter_multiplier
        y, _ = _create_reduction_cell(x, x_1, filters, weight_decay, (dimension_reduction, dimension_reduction))
        x_1 = x
        x = y

    y = _create_head(x, nbr_classes, final_filters)
    model = Model(inputs=ipt, outputs=y, name='NASNet')
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
                          final_filters=768)
    print(model.summary())
