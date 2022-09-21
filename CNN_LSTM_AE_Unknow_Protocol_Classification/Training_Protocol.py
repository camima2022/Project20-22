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

warnings.filterwarnings('ignore')

save_dir = '/home/hbb/pythonProject/CNN_LSTM_AE_Unknow_Protocol_Classification/saved_model'
path = '/home/hbb/pythonProject/CNN_LSTM_AE_Unknow_Protocol_Classification/ISCX2012_1000_version/'

class_list = ['bittorrent', 'dns', 'ftp', 'http', 'imap', 'pop3', 'smb', 'smtp', 'ssh']
nb_classes = 9
length = 784

# prepare data
data, label = DataCollection.DataPayloadCollection_9class(path, class_list)

train_x, test_x, train_y, test_y = train_test_split(data, label, test_size= 0.25, random_state=0, shuffle=True)

train_x = train_x.reshape((train_x.shape[0], 784, 1))
test_x = test_x.reshape((test_x.shape[0], 784, 1))

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
    encode_outputs = Bidirectional(LSTM(units=64, return_sequences=False, recurrent_dropout=0.2, dropout=0.1))(x)

    # decode
    decode_input = RepeatVector(98)(encode_outputs)
    d = Bidirectional(LSTM(units=64, return_sequences=True, recurrent_dropout=0.2, dropout=0.1))(decode_input)
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

ConvAE, Encoder = AE()
ConvAE.summary()
adam = Adam(learning_rate=0.01)

ConvAE.compile(loss='mse',optimizer=adam, metrics=['accuracy'])
history = ConvAE.fit(train_x, train_x, batch_size=128, epochs = 10, verbose = 2, validation_data=(test_x, test_x))
ConvAE.save_weights(save_dir + '/'+'save_weight.ckpt')
Encoder.save_weights(save_dir + '/' + 'save_encoder_weight.ckpt')
print('saved model weights')

score = ConvAE.evaluate(test_x, test_x, verbose=1)
print('TestSet loss:', score[0])
print('TestSet accuracy:', score[1])

val_accuracy = history.history['val_accuracy']
acc = history.history['accuracy']
test_loss = history.history['val_loss']
train_loss = history.history['loss']
for i in range(len(val_accuracy)):
    val_accuracy[i] = round(val_accuracy[i], nb_classes)
    acc[i] = round(acc[i], nb_classes)
    test_loss[i] = round(test_loss[i], nb_classes)
    train_loss[i] = round(train_loss[i], nb_classes)
print(val_accuracy)
print(acc)
print(test_loss)
print(train_loss)

result_path = '/home/hbb/pythonProject/CNN_LSTM_AE_Unknow_Protocol_Classification/saved_model'
fig = plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
fig.savefig(result_path+'/pretraining_acc_our_CNN_LSTM_Merge.png')
fig1 = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower left')
fig1.savefig(result_path+'/pretraining_loss_our_CNN_LSTM_Mergee.png')


encoded_img = Encoder.predict(test_x)
print(encoded_img.shape)
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
cluster_viz(encoded_img, test_y, save_dir, n_classes=9, fig_name='tsne_viz_CNN_LSTM_merge.png')
test_x = test_x.reshape(test_x.shape[0], 784)
cluster_viz(test_x, test_y, save_dir, n_classes=9, fig_name='tsne_viz_CNN_LSTM_raw_data_merge.png')

