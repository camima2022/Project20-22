import keras
from keras import Input, Model
from keras.layers import Convolution2D, MaxPooling2D


def vgg():
    input = Input(shape=(224,224,1))
    x = Convolution2D(64, (3, 3), (1, 1), padding='same', activation='relu')(input)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Convolution2D(128, (3, 3), (1, 1), padding='same', activation='relu')(x)

    return Model(input, x)
model = vgg()
model.summary()