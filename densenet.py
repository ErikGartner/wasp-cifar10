from keras.layers import *
from keras.models import Model
from keras.layers.normalization import *
from keras.regularizers import *


def _create_bnreluconv(x, n_channels, dropout, weight_decay, bottleneck_width=4):
    # bottleneck layer
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(n_channels*bottleneck_width, (1,1),
                      kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(x)
    if dropout > 0:
        x = Dropout(dropout)(x)

    # regular layer
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(n_channels, (3,3), padding='same', kernel_initializer='he_normal')(x)
    if dropout > 0:
        x = Dropout(dropout)(x)
    return x


def _create_dense_block(x, n_channels, n_layers, growth_rate, dropout, weight_decay):
    for i in range(n_layers):
        y = _create_bnreluconv(x, growth_rate, dropout, weight_decay)
        x = Concatenate(axis=-1)([x, y]) # split?
    return x, n_channels+n_layers*growth_rate


def _create_transition_layer(x, n_channels, dropout, weight_decay, compression):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(int(n_channels*compression), (1,1), padding='same', use_bias=False,
                      kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(x)
    if dropout > 0:
        x = Dropout(dropout)(x)
    x = AveragePooling2D((2,2), strides=(2,2))(x)
    return x


def _create_initial_layer(x, n_channels):
    x = Convolution2D(n_channels, (3,3), strides=(1,1), kernel_initializer='he_normal', padding='same')(x)
    return x


def _create_classification_layer(x, n_channels, nbr_classes, pool_size):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(nbr_classes, activation='softmax', kernel_initializer='he_normal')(x)
    return x


def create_densenet(input_shape, dense_layers,
                    growth_rate, nbr_classes,
                    weight_decay, compression=1, dropout=0):
    x = Input(input_shape)
    y = _create_initial_layer(x, growth_rate)
    dense_blocks = len(dense_layers)

    for i in range(dense_blocks):
        y, n_channels = _create_dense_block(y, growth_rate, dense_layers[i],
                                           growth_rate, dropout, weight_decay)
        if i < dense_blocks - 1:
            y = _create_transition_layer(y, n_channels, dropout, weight_decay,
                                         compression)
        else:
            y = _create_classification_layer(y, n_channels, nbr_classes, 8)
    model = Model(inputs=x, outputs=y, name='DenseNet')
    return model
