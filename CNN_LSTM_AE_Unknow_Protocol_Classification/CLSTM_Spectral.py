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


from tensorflow.keras.layers import *

length = 784
nb_classes = 9
save_dir = '/home/hbb/pythonProject/CNN_LSTM_AE_Unknow_Protocol_Classification/saved_model'

# prepare data
def cluster_latent(latent, labels, n_classes=9):
    '''
    :param latent: 2-D array of predicted latent vectors for each sample
    :param labels: vector of true labels
    :param n_classes: integer for number of classes
    :return: evaluation of prediction after applying K-means on the predicted latent vectors
    '''
    km = KMeans(n_clusters=n_classes, random_state=0).fit(latent)
    labels_pred = km.labels_

    #acc_c, acc_all = cluster_acc(labels_pred, labels)
    ari = adjusted_rand_score(labels, labels_pred)
    nmi = normalized_mutual_info_score(labels, labels_pred)

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(labels, labels_pred)

    #print(acc_c)
    #print(acc_all)
    print(ari)
    print(nmi)
    print(cm)

    return  ari, nmi
def cluster_latent_spectral(latent, labels, n_classes=9):

    '''
    :param latent: 2-D array of predicted latent vectors for each sample
    :param labels: vector of true labels
    :param n_classes: integer for number of classes
    :return: evaluation of prediction after applying K-means on the predicted latent vectors
    '''
    lsshc = LSC_DensityCanopyKmeans.LSSHC(mode_framework='eigen_trick', mode_Z_construction='knn', mode_Z_construction_knn='random',
                  mode_sampling_schema='HSV', mode_sampling_approach='probability', m_hyperedge=100, l_hyperedge=40,
                  knn_s=3, k=n_classes)
    labels_pred = lsshc.fit(latent)
    ari = adjusted_rand_score(labels, labels_pred)
    nmi = normalized_mutual_info_score(labels, labels_pred)

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(labels, labels_pred)

    print(ari)
    print(nmi)
    print(cm)

    return ari, nmi
def cluster_viz(latent, labels, save_dir, n_classes=8, fig_name='tsne_viz.png'):
    '''
    :param latent: 2-D array of predicted latent vectors for each sample
    :param labels: vector of true labels
    :param save_dir: string for destination directory
    :param n_classes: integer for number of classes
    :param fig_name: string for figure name
    '''
    colors = cm.rainbow(np.linspace(0, 1, n_classes))

    tsne = TSNE(n_components=2, verbose=1, init='pca', random_state=0)
    tsne_embed = tsne.fit_transform(latent)

    fig, ax = plt.subplots(figsize=(16, 10))
    for i, c in zip(range(n_classes), colors):
        ind_class = np.argwhere(labels == i)
        ax.scatter(tsne_embed[ind_class, 0], tsne_embed[ind_class, 1], color=c, label=i, s=5)
    ax.set_title('t-SNE vizualization of latent vectors')
    ax.set_xlabel('component 1')
    ax.set_ylabel('component 2')
    plt.legend()
    plt.tight_layout()
    fig.savefig(f'{save_dir}/{fig_name}')
    plt.close('all')
def load_dataset_un15():
    import DataCollection_UN15
    from sklearn.model_selection import train_test_split
    path = '/home/hbb/pythonProject/CNN_LSTM_AE_Unknow_Protocol_Classification/un15/'
    class_list = ['dns', 'ftp', 'http', 'pop3', 'smb', 'smtp', 'ssh']
    data, label = DataCollection_UN15.DataPayloadCollection(path, class_list)
    train_x, test_x, train_y, test_y = train_test_split(data, label, test_size=0.25, random_state=0, shuffle=True)
    train_x = train_x.reshape((train_x.shape[0], length , 1))
    test_x = test_x.reshape((test_x.shape[0], length , 1))
    return train_x, test_x, train_y, test_y
train_x, test_x, train_y, test_y = load_dataset_un15()

# create model

def AE():
    # encode
    encode_inputs = Input(shape=(length, 1))
    x = Conv1D(filters=128, kernel_size=5, strides=4, padding='same', activation='relu')(encode_inputs)
    x = BatchNormalization(momentum=0.9)(x)
    cp_input = Conv1D(filters=64, kernel_size=5, strides=2, padding='same', activation='relu')(x)
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

    x = Bidirectional(LSTM(units=32, return_sequences=True, recurrent_dropout=0.2, dropout=0.1))(cp_output)
    x = BatchNormalization(momentum=0.9)(x)
    encode_outputs = Bidirectional(LSTM(units=16, return_sequences=False, recurrent_dropout=0.2, dropout=0.1))(x)

    # decode
    decode_input = RepeatVector(98)(encode_outputs)
    d = Bidirectional(LSTM(units=16, return_sequences=True, recurrent_dropout=0.2, dropout=0.1))(decode_input)
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
    d = Conv1DTranspose(filters=64, kernel_size=5, strides=2, padding='same', activation='relu')(d_cp_output)
    d = BatchNormalization(momentum=0.9)(d)
    d = Conv1DTranspose(filters=128, kernel_size=5, strides=4, padding='same', activation='relu')(d)
    decode_output = TimeDistributed(Dense(1))(d)

    autoEncoder = Model(inputs=encode_inputs, outputs=decode_output, name='AE')
    Encoder = Model(inputs=encode_inputs, outputs=encode_outputs, name='Encoder')
    return autoEncoder, Encoder

def CLSTM_AE():
    encode_inputs = Input(shape=(length, 1))
    e = Conv1D(filters=128, kernel_size=5, strides=4, padding='same', activation='relu')(encode_inputs)
    e = BatchNormalization(momentum=0.9)(e)
    e = Conv1D(filters=64, kernel_size=5, strides=2, padding='same', activation='relu')(e)
    e = Bidirectional(LSTM(units=32, return_sequences=True, recurrent_dropout=0.2, dropout=0.1))(e)
    e = BatchNormalization(momentum=0.9)(e)
    encode_outputs = Bidirectional(LSTM(units=16, return_sequences=False, recurrent_dropout=0.2, dropout=0.1))(e)

    decode_input = RepeatVector(98)(encode_outputs)
    d = Bidirectional(LSTM(units=16, return_sequences=True, recurrent_dropout=0.2, dropout=0.1))(decode_input)
    d = BatchNormalization(momentum=0.9)(d)
    d = Bidirectional(LSTM(units=32, return_sequences=True, recurrent_dropout=0.2, dropout=0.1))(d)
    d = Conv1DTranspose(filters=64, kernel_size=5, strides=2, padding='same', activation='relu')(d)
    d = BatchNormalization(momentum=0.9)(d)
    d = Conv1DTranspose(filters=128, kernel_size=5, strides=4, padding='same', activation='relu')(d)
    decode_output = TimeDistributed(Dense(1))(d)

    autoEncoder = Model(inputs=encode_inputs, outputs=decode_output, name='AE')
    Encoder = Model(inputs=encode_inputs, outputs=encode_outputs, name='Encoder')
    return autoEncoder, Encoder


# training model
ConvAE, Encoder = CLSTM_AE()
ConvAE.summary()

adam = Adam(learning_rate=0.01)
ConvAE.compile(loss='mse',optimizer=adam, metrics=['accuracy'])
history = ConvAE.fit(train_x, train_x, batch_size=128, epochs = 10, verbose = 2, validation_data=(test_x, test_x))

score = ConvAE.evaluate(test_x, test_x, verbose=1)
print('TestSet loss:', score[0])
print('TestSet accuracy:', score[1])
# clustering
encoded_img = Encoder.predict(test_x)
cluster_viz(encoded_img, test_y, save_dir, n_classes=9, fig_name='tsne_viz_encode.png')

from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
encoded_img = min_max_scaler.fit_transform(encoded_img)
print('clustering latent with kmeans')
cluster_latent(encoded_img, test_y, n_classes=7)
print('Clustering latent with spectral kmeans')
cluster_latent_spectral(encoded_img, test_y, n_classes=7)





