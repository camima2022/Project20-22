from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Model
import DataCollection
import warnings
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
import LSC_DensityCanopyKmeans
import tensorflow.keras.layers as layers
import tensorflow as tf
import metrics

from tensorflow.keras.layers import *

length = 128

def CLSTM_Attention_AE():
    encode_inputs = Input(shape=(length, 1))
    e = Conv1D(filters=128, kernel_size=5, strides=4, padding='same', activation='relu')(encode_inputs)
    e = BatchNormalization(momentum=0.9)(e)
    cp_input = Conv1D(filters=64, kernel_size=5, strides=4, padding='same', activation='relu')(e)
    avgpool = GlobalAveragePooling1D(name='channel_avgpool')(cp_input)
    maxpool = GlobalMaxPool1D(name='channel_maxpool')(cp_input)

    Dense_layer1 = Dense(16, activation='relu', name='channel_fc1')
    Dense_layer2 = Dense(64, activation='relu', name='channel_fc2')
    avg_out = Dense_layer2(Dense_layer1(avgpool))
    max_out = Dense_layer2(Dense_layer1(maxpool))
    channel = layers.add([avg_out, max_out])

    channel = Activation('sigmoid', name='channel_sigmoid')(channel)
    channel = Reshape((1, 64), name='channel_reshape')(channel)
    channel_out = tf.multiply(cp_input, channel)

    # Spatial Attention
    avgpool = tf.reduce_mean(channel_out, axis=2, keepdims=True, name='spatial_avgpool')
    maxpool = tf.reduce_max(channel_out, axis=2, keepdims=True, name='spatial_maxpool')

    spatial = Concatenate(axis=2)([avgpool, maxpool])
    spatial = Conv1D(1, 5, strides=1, padding='same', name='spatial_conv2d')(spatial)
    spatial_out = Activation('sigmoid', name='spatial_sigmoid')(spatial)

    CBAM_out = tf.multiply(channel_out, spatial_out, name='multiply')

    # r = Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='residual')(CP_input)
    x = layers.add([CBAM_out, cp_input])
    img_output = Activation('relu', name='relu3')(x)
    cp_output = BatchNormalization(momentum=0.9)(img_output)
    e = Bidirectional(LSTM(units=32, return_sequences=True, recurrent_dropout=0.2, dropout=0.1), name='LSTM1')(cp_output)
    e = BatchNormalization(momentum=0.9)(e)
    encode_outputs = Bidirectional(LSTM(units=2, return_sequences=False, recurrent_dropout=0.2, dropout=0.1))(e)


    decode_input = RepeatVector(25)(encode_outputs)
    d = Bidirectional(LSTM(units=2, return_sequences=True, recurrent_dropout=0.2, dropout=0.1))(decode_input)
    d = BatchNormalization(momentum=0.9)(d)
    d_cp_input = Bidirectional(LSTM(units=32, return_sequences=True, recurrent_dropout=0.2, dropout=0.1))(d)
    d_avgpool = GlobalAveragePooling1D(name='d_channel_avgpool')(d_cp_input)
    d_maxpool = GlobalMaxPool1D(name='d_channel_maxpool')(d_cp_input)

    d_Dense_layer1 = Dense(16, activation='relu', name='d_channel_fc1')
    d_Dense_layer2 = Dense(64, activation='relu', name='d_channel_fc2')
    d_avg_out = d_Dense_layer2(d_Dense_layer1(d_avgpool))
    d_max_out = d_Dense_layer2(d_Dense_layer1(d_maxpool))
    d_channel = layers.add([d_avg_out, d_max_out])

    d_channel = Activation('sigmoid', name='d_channel_sigmoid')(d_channel)
    d_channel = Reshape((1, 64), name='d_channel_reshape')(d_channel)
    d_channel_out = tf.multiply(d_cp_input, d_channel)

    # Spatial Attention
    d_avgpool = tf.reduce_mean(d_channel_out, axis=2, keepdims=True, name='d_spatial_avgpool')
    d_maxpool = tf.reduce_max(d_channel_out, axis=2, keepdims=True, name='d_spatial_maxpool')

    d_spatial = Concatenate(axis=2)([d_avgpool, d_maxpool])
    d_spatial = Conv1D(1, 5, strides=1, padding='same', name='d_spatial_conv2d')(d_spatial)
    d_spatial_out = Activation('sigmoid', name='d_spatial_sigmoid')(d_spatial)

    d_CBAM_out = tf.multiply(d_channel_out, d_spatial_out, name='d_multiply')

    # r = Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='residual')(CP_input)
    d = layers.add([d_CBAM_out, d_cp_input])
    d_img_output = Activation('relu', name='d_relu3')(d)
    d_cp_output = BatchNormalization(momentum=0.9)(d_img_output)

    d = Conv1DTranspose(filters=64, kernel_size=5, strides=4, padding='same', activation='relu')(d_cp_output)
    d = BatchNormalization(momentum=0.9)(d)
    d = Conv1DTranspose(filters=128, kernel_size=5, strides=4, padding='same', activation='relu')(d)
    decode_output = TimeDistributed(Dense(1))(d)

    autoEncoder = Model(inputs=encode_inputs, outputs=decode_output, name='AE')
    Encoder = Model(inputs=encode_inputs, outputs=encode_outputs, name='Encoder')

    return autoEncoder, Encoder

def NIN_LSTM_Attention_AE():
    encode_inputs = Input(shape=(length, 1))
    e = Conv1D(filters=128, kernel_size=5, strides=4, padding='same', activation='relu')(encode_inputs)
    e = BatchNormalization(momentum=0.9)(e)
    e = Conv1D(filters=64, kernel_size=1, strides=1, padding='valid', activation='relu')(e)
    e = BatchNormalization(momentum=0.9)(e)
    e = Conv1D(filters=64, kernel_size=5, strides=4, padding='same', activation='relu')(e)
    e = BatchNormalization(momentum=0.9)(e)
    e = Conv1D(filters=32, kernel_size=1, strides=1, padding='valid', activation='relu')(e)
    cp_input = BatchNormalization(momentum=0.9)(e)
    avgpool = GlobalAveragePooling1D(name='channel_avgpool')(cp_input)
    maxpool = GlobalMaxPool1D(name='channel_maxpool')(cp_input)

    Dense_layer1 = Dense(8, activation='relu', name='channel_fc1')
    Dense_layer2 = Dense(32, activation='relu', name='channel_fc2')
    avg_out = Dense_layer2(Dense_layer1(avgpool))
    max_out = Dense_layer2(Dense_layer1(maxpool))
    channel = layers.add([avg_out, max_out])

    channel = Activation('sigmoid', name='channel_sigmoid')(channel)
    channel = Reshape((1, 32), name='channel_reshape')(channel)
    channel_out = tf.multiply(cp_input, channel)

    # Spatial Attention
    avgpool = tf.reduce_mean(channel_out, axis=2, keepdims=True, name='spatial_avgpool')
    maxpool = tf.reduce_max(channel_out, axis=2, keepdims=True, name='spatial_maxpool')

    spatial = Concatenate(axis=2)([avgpool, maxpool])
    spatial = Conv1D(1, 5, strides=1, padding='same', name='spatial_conv2d')(spatial)
    spatial_out = Activation('sigmoid', name='spatial_sigmoid')(spatial)

    CBAM_out = tf.multiply(channel_out, spatial_out, name='multiply')

    # r = Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='residual')(CP_input)
    x = layers.add([CBAM_out, cp_input])
    img_output = Activation('relu', name='relu3')(x)
    cp_output = BatchNormalization(momentum=0.9)(img_output)
    e = Bidirectional(LSTM(units=32, return_sequences=True, recurrent_dropout=0.2, dropout=0.1), name='LSTM1')(cp_output)
    e = BatchNormalization(momentum=0.9)(e)
    encode_outputs = Bidirectional(LSTM(units=2, return_sequences=False, recurrent_dropout=0.2, dropout=0.1))(e)


    decode_input = RepeatVector(8)(encode_outputs)
    d = Bidirectional(LSTM(units=2, return_sequences=True, recurrent_dropout=0.2, dropout=0.1))(decode_input)
    d = BatchNormalization(momentum=0.9)(d)
    d_cp_input = Bidirectional(LSTM(units=32, return_sequences=True, recurrent_dropout=0.2, dropout=0.1))(d)
    d_avgpool = GlobalAveragePooling1D(name='d_channel_avgpool')(d_cp_input)
    d_maxpool = GlobalMaxPool1D(name='d_channel_maxpool')(d_cp_input)

    d_Dense_layer1 = Dense(16, activation='relu', name='d_channel_fc1')
    d_Dense_layer2 = Dense(64, activation='relu', name='d_channel_fc2')
    d_avg_out = d_Dense_layer2(d_Dense_layer1(d_avgpool))
    d_max_out = d_Dense_layer2(d_Dense_layer1(d_maxpool))
    d_channel = layers.add([d_avg_out, d_max_out])

    d_channel = Activation('sigmoid', name='d_channel_sigmoid')(d_channel)
    d_channel = Reshape((1, 64), name='d_channel_reshape')(d_channel)
    d_channel_out = tf.multiply(d_cp_input, d_channel)

    # Spatial Attention
    d_avgpool = tf.reduce_mean(d_channel_out, axis=2, keepdims=True, name='d_spatial_avgpool')
    d_maxpool = tf.reduce_max(d_channel_out, axis=2, keepdims=True, name='d_spatial_maxpool')

    d_spatial = Concatenate(axis=2)([d_avgpool, d_maxpool])
    d_spatial = Conv1D(1, 5, strides=1, padding='same', name='d_spatial_conv2d')(d_spatial)
    d_spatial_out = Activation('sigmoid', name='d_spatial_sigmoid')(d_spatial)

    d_CBAM_out = tf.multiply(d_channel_out, d_spatial_out, name='d_multiply')

    # r = Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='residual')(CP_input)
    d = layers.add([d_CBAM_out, d_cp_input])
    d_img_output = Activation('relu', name='d_relu3')(d)
    d_cp_output = BatchNormalization(momentum=0.9)(d_img_output)
    d = Conv1DTranspose(filters=32, kernel_size=1, strides=1, padding='valid', activation='relu')(d_cp_output)
    d = BatchNormalization(momentum=0.9)(d)
    d = Conv1DTranspose(filters=64, kernel_size=5, strides=4, padding='same', activation='relu')(d)
    d = BatchNormalization(momentum=0.9)(d)
    d = Conv1DTranspose(filters=64, kernel_size=1, strides=1, padding='valid', activation='relu')(d)
    d = BatchNormalization(momentum=0.9)(d)
    d = Conv1DTranspose(filters=128, kernel_size=5, strides=4, padding='same', activation='relu')(d)
    decode_output = TimeDistributed(Dense(1))(d)

    autoEncoder = Model(inputs=encode_inputs, outputs=decode_output, name='AE')
    Encoder = Model(inputs=encode_inputs, outputs=encode_outputs, name='Encoder')

    return autoEncoder, Encoder

def CLSTM_AE():
    encode_inputs = Input(shape=(length, 1))
    e = Conv1D(filters=64, kernel_size=5, strides=4, padding='same', activation='relu')(encode_inputs)
    e = BatchNormalization(momentum=0.9)(e)
    e = Conv1D(filters=32, kernel_size=5, strides=4, padding='same', activation='relu')(e)
    e = Bidirectional(LSTM(units=16, return_sequences=True, recurrent_dropout=0.2, dropout=0.1), name='LSTM1')(e)
    e = BatchNormalization(momentum=0.9)(e)
    encode_outputs = Bidirectional(LSTM(units=2, return_sequences=False, recurrent_dropout=0.2, dropout=0.1))(e)

    decode_input = RepeatVector(8)(encode_outputs)
    d = Bidirectional(LSTM(units=2, return_sequences=True, recurrent_dropout=0.2, dropout=0.1))(decode_input)
    d = BatchNormalization(momentum=0.9)(d)
    d = Bidirectional(LSTM(units=16, return_sequences=True, recurrent_dropout=0.2, dropout=0.1))(d)
    d = Conv1DTranspose(filters=32, kernel_size=5, strides=4, padding='same', activation='relu')(d)
    d = BatchNormalization(momentum=0.9)(d)
    d = Conv1DTranspose(filters=64, kernel_size=5, strides=4, padding='same', activation='relu')(d)
    decode_output = TimeDistributed(Dense(1))(d)

    autoEncoder = Model(inputs=encode_inputs, outputs=decode_output, name='AE')
    Encoder = Model(inputs=encode_inputs, outputs=encode_outputs, name='Encoder')
    return autoEncoder, Encoder


#autoEncoder, Encoder = CLSTM_Attention_AE()
#autoEncoder.summary()

#autoEncoder, _ = CLSTM_AE()
#autoEncoder.summary()

autoEncoder, Encoder = NIN_LSTM_Attention_AE()
autoEncoder.summary()


