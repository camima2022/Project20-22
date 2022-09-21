from time import time
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras import callbacks
from tensorflow.keras.optimizers import SGD
from sklearn.cluster import KMeans
import metrics
from tensorflow.keras.initializers import VarianceScaling

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

num_clusters = 4
length = 128

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

def autoencoder(dims, act='relu'):
    init = VarianceScaling(scale=1. / 3., mode='fan_in',
                           distribution='uniform')
    n_stacks = len(dims) - 1
    x = Input(shape=(dims[0], ), name='input')
    h = x
    for i in range(n_stacks -1):
        h = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(h)
    h = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks -1))(h)
    y = h
    for i in range(n_stacks -1, 0, -1):
        y = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(y)
    y = Dense(dims[0], kernel_initializer=init, name='decoder_0')(y)
    return Model(inputs=x, outputs=y, name='AE'), Model(inputs=x, outputs=h ,name='encoder')

class ClusteringLayer(Layer):
    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)
    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform',
                                        name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True
    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """

        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q
    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

class DEC(object):
    def __init__(self, dims, n_clusters=num_clusters, alpha=1.0):
        super(DEC, self).__init__()
        self.dims = dims
        self.input_dim = dims[0]
        self.n_stacks = len(self.dims) - 1

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.autoencoder, self.encoder = CLSTM_AE()
        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(self.encoder.output)
        self.model = Model(inputs=self.encoder.input, outputs=clustering_layer)
    def pretrain(self, x, y=None, epochs=200, batch_size=256, save_dir='results/temp'):
        print('...Pretraining...')
        pretrain_optimizer = SGD(lr=0.1, momentum=0.9)
        self.autoencoder.compile(optimizer=pretrain_optimizer, loss='mse')
        csv_logger = callbacks.CSVLogger(save_dir + '/pretrain_log.csv')
        cb = [csv_logger]
        if y is not None:
            class PrintACC(callbacks.Callback):
                def __init__(self, x, y):
                    self.x = x
                    self.y = y
                    super(PrintACC, self).__init__()
                def on_epoch_end(self, epoch, logs=None):
                    if int(epochs/10) != 0 and epoch % int(epochs/10) != 0:
                        return
                    feature_model = Model(self.model.input,
                                          self.model.get_layer(name='LSTM1').output)
                    features = feature_model.predict(self.x)
                    features = features.reshape((features.shape[0], -1))
                    from sklearn import preprocessing
                    min_max_scaler = preprocessing.MinMaxScaler()
                    features = min_max_scaler.fit_transform(features)
                    km = KMeans(n_clusters=num_clusters, n_init=20, n_jobs=4)
                    y_pred = km.fit_predict(features)
                    print(' ' * 8 + '|==>  acc: %.4f,  nmi: %.4f  <==|'
                          % (metrics.acc(self.y, y_pred, num_cluster=num_clusters), metrics.nmi(self.y, y_pred)))

            cb.append(PrintACC(x, y))
        t0 = time()
        self.autoencoder.fit(x, x, batch_size=batch_size, epochs=epochs, callbacks=cb)
        print('Pretraining time: %ds' % round(time() - t0))
        self.autoencoder.save_weights(save_dir + '/ae_weights.h5')
        print('Pretrained weights are saved to %s/ae_weights.h5' % save_dir)
        self.pretrained = True
    def load_weights(self, weights):  # load weights of DEC model
        self.model.load_weights(weights)
    def extract_features(self, x):
        return self.encoder.predict(x)
    def predict(self, x):  # predict cluster labels using the output of clustering layer
        q = self.model.predict(x, verbose=0)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T
    def compile(self, loss='kld'):
        optimizer = SGD(learning_rate=0.1, momentum=0.9)
        loss = 'kld'
        self.model.compile(optimizer=optimizer, loss=loss)

    def fit(self, x, y=None, maxiter=2e4, batch_size=256, tol=1e-3,
            update_interval=140, save_dir='./results/temp'):
        print('Update interval', update_interval)
        save_interval = int(x.shape[0] / batch_size) * 5  # 5 epochs
        print('Save interval', save_interval)

        # Step 1: initialize cluster centers using k-means
        #t1 = time()
        print('Initializing cluster centers with k-means.')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        from sklearn import preprocessing
        min_max_scaler = preprocessing.MinMaxScaler()
        features = self.encoder.predict(x)
        features = min_max_scaler.fit_transform(features)
        y_pred = kmeans.fit_predict(features)
        y_pred_last = np.copy(y_pred)
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

        # Step 2: deep clustering
        # logging file
        import csv
        logfile = open(save_dir + '/dec_log.csv', 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'acc', 'nmi', 'ari', 'loss'])
        logwriter.writeheader()

        loss = 0
        index = 0
        index_array = np.arange(x.shape[0])
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q = self.model.predict(x, verbose=0)
                p = self.target_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                y_pred = q.argmax(1)
                if y is not None:
                    acc = np.round(metrics.acc(y, y_pred, num_cluster=num_clusters), 5)
                    nmi = np.round(metrics.nmi(y, y_pred), 5)
                    ari = np.round(metrics.ari(y, y_pred), 5)
                    loss = np.round(loss, 5)
                    logdict = dict(iter=ite, acc=acc, nmi=nmi, ari=ari, loss=loss)
                    logwriter.writerow(logdict)
                    print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f' % (ite, acc, nmi, ari), ' ; loss=', loss)

                # check stop criterion
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)
                if ite > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    logfile.close()
                    break

            # train on batch
            # if index == 0:
            #     np.random.shuffle(index_array)
            idx = index_array[index * batch_size: min((index + 1) * batch_size, x.shape[0])]
            loss = self.model.train_on_batch(x=x[idx], y=p[idx])
            index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

            # save intermediate model
            if ite % save_interval == 0:
                print('saving model to:', save_dir + '/DEC_model_' + str(ite) + '.h5')
                self.model.save_weights(save_dir + '/DEC_model_' + str(ite) + '.h5')

            ite += 1

        # save the trained model
        logfile.close()
        print('saving model to:', save_dir + '/DEC_model_final.h5')
        self.model.save_weights(save_dir + '/DEC_model_final.h5')

        return y_pred
def load_mnist():
    # the data, shuffled and split between train and test sets
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape((x.shape[0], -1))
    x = x / 255
    print('MNIST samples', x.shape)
    return x, y

def load_dataset_ids17():
    import DataCollection_IDS
    from sklearn.model_selection import train_test_split
    path = '/home/hbb/pythonProject/CNN_LSTM_AE_Unknow_Protocol_Classification/IDS17_v3/'
    class_list = ['dns', 'ftp', 'http', 'smb']
    data, label = DataCollection_IDS.DataPayloadCollection(path, class_list)
    train_x, test_x, train_y, test_y = train_test_split(data, label, test_size=0.25, random_state=0, shuffle=True)
    x = np.concatenate((train_x, test_x))
    y = np.concatenate((train_y, test_y))
    print('dataset shapes', x.shape)

    return x, y


if __name__ == "__main__":
    x, y = load_dataset_ids17()
    n_clusters = len(np.unique(y))

    update_interval = 140
    pretrain_epochs = 30
    #init = VarianceScaling(scale=1. / 3., mode='fan_in',
    #                       distribution='uniform')

    dims = [x.shape[-1], 500, 500, 2000, 4]
    batch_size = 256
    tol = 0.001
    maxiter = 2e4
    save_dir = '/home/hbb/pythonProject/DEC-keras-master/results/temp'
    #pretrain_optimizer = SGD(lr=1, momentum=0.9)
    dec = DEC(dims=dims, n_clusters=n_clusters)
    dec.pretrain(x=x, y=y, epochs=pretrain_epochs, batch_size=batch_size,
                 save_dir=save_dir)
    t0 = time()
    dec.compile()
    y_pred = dec.fit(x, y=y, update_interval=update_interval, save_dir=save_dir)
    print('acc:', metrics.acc(y, y_pred, num_cluster=num_clusters))
    print('clustering time: ', (time() - t0))
