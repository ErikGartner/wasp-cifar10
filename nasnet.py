from keras.layers import *
from keras.models import Model
from keras.layers.normalization import *
from keras.regularizers import *


def _sep_layer(x, nbr_filters, kernel_size, weight_decay, strides=(1, 1)):
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
    return x


def _avg_layer(pool_size, strides=(1, 1)):
    return AveragePooling2D(pool_size, padding='same', strides=strides)


def _max_layer(pool_size, strides=(1, 1)):
    return MaxPooling2D(pool_size, padding='same', strides=strides)


def _create_reduction_cell(x, x_1, nbr_filters, weight_decay):

    sep7x7 = _sep_layer(x_1, nbr_filters, (7, 7), weight_decay, strides=(2, 2))
    sep5x5 = _sep_layer(x, nbr_filters, (5, 5), weight_decay, strides=(2, 2))
    y1 = Add()([sep7x7, sep5x5])

    max3x3 = _max_layer((3, 3), strides=(2, 2))(x)
    sep7x7 = _sep_layer(x_1, nbr_filters, (7, 7), weight_decay, strides=(2, 2))
    y2 = Add()([max3x3, sep7x7])

    avg3x3 = _avg_layer((3, 3), strides=(2, 2))(x)
    sep5x5 = _sep_layer(x_1, nbr_filters, (5, 5), weight_decay, strides=(2, 2))
    y3 = Add()([avg3x3, sep5x5])

    max3x3 = _max_layer((3, 3), strides=(2, 2))(x)
    sep3x3 = _sep_layer(y1, nbr_filters, (3, 3), weight_decay)
    z1 = Add()([max3x3, sep3x3])

    avg3x3 = _avg_layer((3, 3))(y1)
    z2 = Add()([avg3x3, y2])

    result = Concatenate(axis=-1)([z1, z2, y3])
    result = _sep_layer(result, nbr_filters * 2, (1, 1), weight_decay)
    return result, int(result.shape[3])


def _create_normal_cell(x, x_1, nbr_filters, weight_decay):
    sep3x3 = _sep_layer(x, nbr_filters, (3, 3), weight_decay)
    y1 = Add()([sep3x3, x])

    sep3x3 = _sep_layer(x_1, nbr_filters, (3, 3), weight_decay)
    sep5x5 = _sep_layer(x, nbr_filters, (5, 5), weight_decay)
    y2 = Add()([sep3x3, sep5x5])

    avg3x3 = _avg_layer((3, 3))(x)
    y3 = Add()([avg3x3, x_1])

    avg3x3_1 = _avg_layer((3, 3))(x_1)
    avg3x3_2 = _avg_layer((3, 3))(x_1)
    y4 = Add()([avg3x3_1, avg3x3_2])

    sep5x5 = _sep_layer(x_1, nbr_filters, (5, 5), weight_decay)
    sep3x3 = _sep_layer(x_1, nbr_filters, (3, 3), weight_decay)
    y5 = Add()([sep5x5, sep3x3])

    result = Concatenate(axis=-1)([y1, y2, y3, y4, y5])
    result = _sep_layer(result, nbr_filters, (1, 1), weight_decay)
    return result, int(result.shape[3])


def _create_head(x, nbr_classes):
    x = Flatten()(x)
    x = Dense(nbr_classes, activation='softmax', kernel_initializer='he_normal')(x)
    return x


def create_nasnet(input_shape, nbr_normal_cells, nbr_blocks, weight_decay, nbr_classes, nbr_filters):

    ipt = Input(input_shape)
    # Sample up to nbr_filters of input
    x = _sep_layer(ipt, nbr_filters, (1, 1), weight_decay)
    x_1 = x

    filters = nbr_filters
    for i in range(nbr_blocks):
        for j in range(nbr_normal_cells):
            y, filters = _create_normal_cell(x, x_1, filters,
                                             weight_decay)
            x_1 = x
            x = y

        y, filters = _create_reduction_cell(x, x_1, filters, weight_decay)
        x_1 = y
        x = y

    y = _create_head(x, nbr_classes)
    model = Model(inputs=ipt, outputs=y, name='NASNet')
    return model


if __name__ == '__main__':
    model = create_nasnet(input_shape=(32, 32, 3),
                          nbr_normal_cells=6,
                          nbr_blocks=2,
                          weight_decay=1e-4,
                          nbr_classes=10,
                          nbr_filters=128)
    print(model.summary())
