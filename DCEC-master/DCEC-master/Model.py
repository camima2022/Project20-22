from keras.layers import *
from keras.models import Sequential, Model
from keras import backend as K
import keras
from tensorflow import split


def conv_block(x, nb_filter, nb_row, nb_col, border_mode='same', subsample=(1, 1)):
    x = Convolution2D(nb_filter, kernel_size=(nb_row, nb_col), strides=subsample, padding=border_mode)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

# def inception_A(input):
#     if K.image_dim_ordering() == "th":
#         channel_axis = 1
#     else:
#         channel_axis = -1
#
#     a1 = conv_block(input, 96, 1, 1)
#
#     a2 = conv_block(input, 64, 1, 1)
#     a2 = conv_block(a2, 96, 3, 3)
#
#     a3 = conv_block(input, 64, 1, 1)
#     a3 = conv_block(a3, 96, 3, 3)
#     a3 = conv_block(a3, 96, 3, 3)
#
#     a4 = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(input)
#     a4 = conv_block(a4, 96, 1, 1)
#
#     # m = merge([a1, a2, a3, a4], mode='concat', concat_axis=channel_axis)
#     m = concatenate([a1, a2, a3, a4], mode='concat', concat_axis=-1)
#
#     return m

def CNN(input_shape=(300, 300, 3)):
    input = Input(shape=input_shape)
    x = Convolution2D(32,(3, 3), strides=2, padding='valid', activation='relu')(input)
    x = Convolution2D(32,(3, 3), strides=1, padding='valid', activation='relu')(x)
    x = Convolution2D(64,(3, 3), strides=1, padding='same', activation='relu')(x)

    a1 = Convolution2D(96, (3, 3), strides=2, padding='valid', activation='relu')(x)
    a2 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(x)
    x = concatenate([a1, a2], axis=-1)

    # x = subtract(x)

    # # # input1 = Input(shape=(96,))
    # x1 = Conv2DTranspose(64, (3, 3), strides=1, padding='same', activation='relu')(x)
    # # # input2 = Input(shape=(64,))
    # x2 = UpSampling2D(size=(1, 1))(x)
    # x2 = Dense(64, activation='relu')(x2)

    # split0, split1, split2 = tf.split(x, [4, 15, 11], 4)
    x1, x2 = split(x, [96, 64], 3)
    x1 = Conv2DTranspose(64, (3, 3), strides=2, padding='valid', activation='relu')(x1)
    x2 = UpSampling2D(size=(2, 2))(x2)
    x = x2
    # out = Dense(4)(subtracted)





    # m = merge([a1, a2, a3, a4], mode='concat', concat_axis=channel_axis)
    # m = concatenate([a1, a2, a3, a4], axis=-1)

    return Model(input, x)

model = CNN()
model.summary()

def inception_stem(input):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    # Input Shape is 299 x 299 x 3 (th) or 3 x 299 x 299 (th)
    x = conv_block(input, 32, 3, 3, subsample=(2, 2), border_mode='valid')
    x = conv_block(x, 32, 3, 3, border_mode='valid')
    x = conv_block(x, 64, 3, 3)

    x1 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid')(x)
    x2 = conv_block(x, 96, 3, 3, subsample=(2, 2), border_mode='valid')

    x = keras.layers.merge([x1, x2], mode='concat', concat_axis=channel_axis)

    x1 = conv_block(x, 64, 1, 1)
    x1 = conv_block(x1, 96, 3, 3, border_mode='valid')

    x2 = conv_block(x, 64, 1, 1)
    x2 = conv_block(x2, 64, 1, 7)
    x2 = conv_block(x2, 64, 7, 1)
    x2 = conv_block(x2, 96, 3, 3, border_mode='valid')

    x = merge([x1, x2], mode='concat', concat_axis=channel_axis)

    x1 = conv_block(x, 192, 3, 3, subsample=(2, 2), border_mode='valid')
    x2 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid')(x)

    x = merge([x1, x2], mode='concat', concat_axis=channel_axis)
    return x