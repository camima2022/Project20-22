from keras.layers import MaxPooling2D, Conv2D, Conv2DTranspose, merge, Dense, Flatten, Reshape, MaxPooling2D, \
    Convolution2D, BatchNormalization, Activation, AveragePooling2D, Input, concatenate, Dropout, subtract, \
    UpSampling2D, average, Average

from keras.models import Sequential, Model
from keras.utils.vis_utils import plot_model
import numpy as np

from keras import backend as K
from tensorflow import split


def CAE_old(input_shape=(16, 16, 1), filters=[32, 64, 128, 10]):
    model = Sequential()
    if input_shape[0] % 8 == 0:
        pad3 = 'same'
    else:
        pad3 = 'valid'
    model.add(Conv2D(filters[0], 5, strides=2, padding='same', activation='relu', name='conv1', input_shape=input_shape))

    model.add(Conv2D(filters[1], 5, strides=2, padding='same', activation='relu', name='conv2'))

    model.add(Conv2D(filters[2], 3, strides=2, padding=pad3, activation='relu', name='conv3'))

    model.add(Flatten())
    model.add(Dense(units=filters[3], name='embedding'))
    model.add(Dense(units=filters[2]*int(input_shape[0]/8)*int(input_shape[0]/8), activation='relu'))

    model.add(Reshape((int(input_shape[0]/8), int(input_shape[0]/8), filters[2])))
    model.add(Conv2DTranspose(filters[1], 3, strides=2, padding=pad3, activation='relu', name='deconv3'))

    model.add(Conv2DTranspose(filters[0], 5, strides=2, padding='same', activation='relu', name='deconv2'))
    # model.add(UpSampling2D(size=(2, 2)))

    model.add(Conv2DTranspose(input_shape[2], 5, strides=2, padding='same', name='deconv1'))
    model.summary()
    return model
'''
def conv_block(x, nb_filter, nb_row, nb_col, border_mode='same', subsample=(1, 1)):
    # if K.image_dim_ordering() == "th":
    #     channel_axis = 1
    # else:
    #     channel_axis = -1

    x = Convolution2D(nb_filter,(nb_row, nb_col), strides=subsample, padding=border_mode)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

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

    x = merge([x1, x2], mode='concat', concat_axis=channel_axis)

    x1 = conv_block(x, 64, 1, 1)
    x1 = conv_block(x1, 96, 3, 3, border_mode='valid')

    x2 = conv_block(x, 64, 1, 1)
    x2 = conv_block(x2, 64, 1, 7)
    x2 = conv_block(x2, 64, 7, 1)
    x2 = conv_block(x2, 96, 3, 3, border_mode='valid')

    x = merge([x1, x2], mode='concat', concat_axis=channel_axis)

    x1 = conv_block(x, 192, 3, 3, subsample=(2, 2), border_mode='valid')
    x2 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)

    x = merge([x1, x2], mode='concat', concat_axis=channel_axis)
    return x

def inception_A(input):

    a1 = conv_block(input, 96, 1, 1)

    a2 = conv_block(input, 64, 1, 1)
    a2 = conv_block(a2, 96, 3, 3)

    a3 = conv_block(input, 64, 1, 1)
    a3 = conv_block(a3, 96, 3, 3)
    a3 = conv_block(a3, 96, 3, 3)

    a4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
    a4 = conv_block(a4, 96, 1, 1)

    m = concatenate([a1, a2, a3, a4], axis=-1)

    return m

def reduction_A(input):
    # if K.image_dim_ordering() == "th":
    #     channel_axis = 1
    # else:
    channel_axis = -1

    r1 = conv_block(input, 384, 3, 3, subsample=(2, 2), border_mode='same')

    r2 = conv_block(input, 192, 1, 1)
    r2 = conv_block(r2, 224, 3, 3)
    r2 = conv_block(r2, 256, 3, 3, subsample=(2, 2), border_mode='same')

    r3 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(input)

    # m = merge([r1, r2, r3], mode='concat', concat_axis=channel_axis)
    m = concatenate([r1, r2, r3], axis=-1)
    # m = Model(input, [r1, r2, r3])
    return m

def dereduction_A(input):
    pass
'''

# model = CAE_old()
def CAE_VGG(input_shape=(16, 16, 1)):
    model = Sequential()
    if input_shape[0] % 8 == 0:
        pad3 = 'same'
    else:
        pad3 = 'valid'

    input = Input(shape=input_shape)
    x = Convolution2D(64, (3, 3), strides=1, padding='same', activation='relu')(input)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    x = Convolution2D(128, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    x = Convolution2D(256, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    x = Flatten()(x)

    x = Dense(units=128)(x)
    x = Dropout(.3)(x)
    x = Dense(units=128)(x)
    x = Dropout(.2)(x)
    x = Dense(units=10, name='embedding')(x)

    x = Dropout(.2)(x)
    x = Dense(units=128)(x)
    x = Dropout(.3)(x)
    x = Dense(units=128)(x)

    x = Dense(units=256*int(input_shape[0]/8)*int(input_shape[0]/8), activation='relu')(x)
    x = Reshape((2, 2, 256))(x)

    x = Conv2DTranspose(256, (3, 3), strides=2, padding='same', activation='relu')(x)
    x = Conv2DTranspose(128, (3, 3), strides=2, padding='same', activation='relu')(x)
    x = Conv2DTranspose(64, (3, 3), strides=2, padding='same', activation='relu')(x)
    x = Conv2DTranspose(1, (3, 3), strides=1, padding='same', activation='relu')(x)

    model = Model(input, x)
    return model

def CAE(input_shape=(12, 12, 1)):

    input = Input(shape=input_shape)
    x = Convolution2D(8, (3, 3), strides=1, padding='same', activation='relu')(input)
    x = Convolution2D(8, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = Convolution2D(16, (3, 3), strides=1, padding='same', activation='relu')(x)

    a1 = Convolution2D(24, (3, 3), strides=2, padding='same', activation='relu', name='a1')(x)
    a2 = MaxPooling2D(pool_size=(2, 2), strides=2, name='a2')(x)

    x = concatenate([a1, a2], axis=-1)

    b1 =Convolution2D(16, (1, 1), strides=1, padding='same', activation='relu', name='b1_1')(x)
    b1 =Convolution2D(16, (7, 1), strides=1, padding='same', activation='relu', name='b1_2')(b1)
    b1 =Convolution2D(16, (1, 7), strides=1, padding='same', activation='relu', name='b1_3')(b1)
    b1 =Convolution2D(24, (3, 3), strides=1, padding='same', activation='relu', name='b1_4')(b1)
    #
    b2 =Convolution2D(16, (1, 1), strides=1, padding='same', activation='relu', name='b2_1')(x)
    b2 =Convolution2D(24, (3, 3), strides=1, padding='same', activation='relu', name='b2_2')(b2)
    #
    x = concatenate([b1, b2], axis=-1)

    c1 = MaxPooling2D(pool_size=(2, 2), strides=2, name='c1')(x)
    c2 = Convolution2D(48, (3, 3), strides=2, padding='same', activation='relu', name='c2')(x)

    x = concatenate([c1, c2], axis=-1, name='H1') # #
    x = Flatten()(x)
    x = Dense(units=10, name='embedding')(x)
    x = Dense(units=96 * int(input_shape[0] / 4) * int(input_shape[0] / 4), activation='relu')(x)
    #
    x = Reshape((int(input_shape[0] / 4), int(input_shape[0] / 4), 96))(x)

    c11, c12 = split(x, [48, 48], 3)

    c11 = UpSampling2D(size=(2, 2), name='c11')(c11)
    c12 = Conv2DTranspose(48, (3, 3), strides=2, padding='same', activation='relu', name='c12')(c12)
    # #
    # # # c12 = Dense(80)(c12)
    x = Average()([c11, c12])
    # # #
    b11, b12 = split(x, [24, 24], 3)
    b11 = Conv2DTranspose(24, (3, 3), strides=1, padding='same', activation='relu', name='b11_4')(b11)
    b11 = Conv2DTranspose(16, (1, 7), strides=1, padding='same', activation='relu', name='b11_3')(b11)
    b11 = Conv2DTranspose(16, (7, 1), strides=1, padding='same', activation='relu', name='b11_2')(b11)
    b11 = Conv2DTranspose(16, (1, 1), strides=1, padding='same', activation='relu', name='b11_1')(b11)

    b12 = Conv2DTranspose(24, (3, 3), strides=1, padding='same', activation='relu', name='b12_2')(b12)
    b12 = Conv2DTranspose(16, (1, 1), strides=1, padding='same', activation='relu', name='b12_1')(b12)

    x = Average()([b11, b12])
    x = Dense(40)(x)

    a11, a12 = split(x, [24, 16], 3)
    a11 = Conv2DTranspose(16, (3, 3), strides=2, padding='same', activation='relu', name='a11')(a11)
    a12 = UpSampling2D(size=(2, 2), name='a12')(a12)
    #
    # a11 = Dense(16)(a11)
    x = Average()([a11, a12])
    #
    x = Conv2DTranspose(16, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = Conv2DTranspose(8, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = Conv2DTranspose(8, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = Conv2DTranspose(1, (3, 3), strides=1, padding='same', activation='relu')(x)

    model = Model(input, x)

    return model

model = CAE()
model.summary()
# model = CAE()
# model.summary()

# if __name__ == "__main__":
#     from time import time
#
#     # setting the hyper parameters
#     import argparse
#     parser = argparse.ArgumentParser(description='train')
#     parser.add_argument('--dataset', default='usps', choices=['mnist', 'usps'])
#     parser.add_argument('--n_clusters', default=10, type=int)
#     parser.add_argument('--batch_size', default=256, type=int)
#     parser.add_argument('--epochs', default=200, type=int)
#     parser.add_argument('--save_dir', default='results/temp', type=str)
#     args = parser.parse_args()
#     print(args)
#
#     import os
#     if not os.path.exists(args.save_dir):
#         os.makedirs(args.save_dir)
#
#     # load dataset
#     from datasets import load_mnist, load_usps
#     if args.dataset == 'mnist':
#         x, y = load_mnist()
#     elif args.dataset == 'usps':
#         x, y = load_usps('data/usps')
#
#     # define the model
#     model = CAE(input_shape=x.shape[1:], filters=[32, 64, 128, 10])
#     plot_model(model, to_file=args.save_dir + '/%s-pretrain-model.png' % args.dataset, show_shapes=True)
#     model.summary()
#
#     # compile the model and callbacks
#     optimizer = 'adam'
#     model.compile(optimizer=optimizer, loss='mse')
#     from keras.callbacks import CSVLogger
#     csv_logger = CSVLogger(args.save_dir + '/%s-pretrain-log.csv' % args.dataset)
#
#     # begin training
#     t0 = time()
#     model.fit(x, x, batch_size=args.batch_size, epochs=args.epochs, callbacks=[csv_logger])
#     print('Training time: ', time() - t0)
#     model.save(args.save_dir + '/%s-pretrain-model-%d.h5' % (args.dataset, args.epochs))
#
#     # extract features
#     feature_model = Model(inputs=model.input, outputs=model.get_layer(name='embedding').output)
#     features = feature_model.predict(x)
#     print('feature shape=', features.shape)
#
#     # use features for clustering
#     from sklearn.cluster import KMeans
#     km = KMeans(n_clusters=args.n_clusters)
#
#     features = np.reshape(features, newshape=(features.shape[0], -1))
#     pred = km.fit_predict(features)
#     from . import metrics
#     print('acc=', metrics.acc(y, pred), 'nmi=', metrics.nmi(y, pred), 'ari=', metrics.ari(y, pred))
