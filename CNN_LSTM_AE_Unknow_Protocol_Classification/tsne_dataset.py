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

save_dir = '/home/hbb/pythonProject/CNN_LSTM_AE_Unknow_Protocol_Classification/saved_model'

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
def load_mnist():
    # the data, shuffled and split between train and test sets
    #from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape((x_train.shape[0], -1))
    x_test = x_test.reshape((x_test.shape[0], -1))
    x_train = x_train / 255
    x_test = x_test / 255

    return x_train, y_train, x_test, y_test
"""
x_train, y_train, x_test, y_test = load_mnist()
cluster_viz(x_test, y_test, save_dir, n_classes=10, fig_name='tsne_viz_mnist.png')
km = KMeans(n_clusters=10, n_init=20, n_jobs=4)
y_pred = km.fit_predict(x_test)
print(' ' * 8 + '|==>  acc: %.4f,  nmi: %.4f  <==|' % (metrics.acc(y_test, y_pred, num_cluster=10), metrics.nmi(y_test, y_pred)))
"""
def load_dataset():
    import DataCollection
    from sklearn.model_selection import train_test_split
    path = '/home/hbb/pythonProject/CNN_LSTM_AE_Unknow_Protocol_Classification/ids17/'
    class_list = ['dns', 'ftp', 'http', 'ntp', 'smb', 'ssh', 'tls']
    data, label = DataCollection.DataPayloadCollection_ids17(path, class_list)
    train_x, test_x, train_y, test_y = train_test_split(data, label, test_size=0.25, random_state=0, shuffle=True)

    return train_x, test_x, train_y, test_y
def load_dataset_ids17():
    import DataCollection_IDS
    from sklearn.model_selection import train_test_split
    path = '/home/hbb/pythonProject/CNN_LSTM_AE_Unknow_Protocol_Classification/IDS17/'
    class_list = ['dns', 'ftp', 'http', 'smb']
    data, label = DataCollection_IDS.DataPayloadCollection(path, class_list)
    train_x, test_x, train_y, test_y = train_test_split(data, label, test_size=0.25, random_state=0, shuffle=True)

    return train_x, test_x, train_y, test_y
def load_dataset_ids17_v2():
    import DataCollection_IDS
    from sklearn.model_selection import train_test_split
    path = '/home/hbb/pythonProject/CNN_LSTM_AE_Unknow_Protocol_Classification/IDS17_v2/'
    class_list = ['dns', 'ftp', 'http', 'smb']
    data, label = DataCollection_IDS.DataPayloadCollection(path, class_list)
    train_x, test_x, train_y, test_y = train_test_split(data, label, test_size=0.25, random_state=0, shuffle=True)

    return train_x, test_x, train_y, test_y
def load_dataset_ids17_v3():
    import DataCollection_IDS
    from sklearn.model_selection import train_test_split
    path = '/home/hbb/pythonProject/CNN_LSTM_AE_Unknow_Protocol_Classification/IDS17_v3/'
    class_list = ['dns', 'ftp', 'http', 'smb']
    data, label = DataCollection_IDS.DataPayloadCollection(path, class_list)
    train_x, test_x, train_y, test_y = train_test_split(data, label, test_size=0.25, random_state=0, shuffle=True)

    return train_x, test_x, train_y, test_y

def load_dataset_UN15():
    import DataCollection_UN15
    from sklearn.model_selection import train_test_split
    path = '/home/hbb/pythonProject/CNN_LSTM_AE_Unknow_Protocol_Classification/UN15/'
    class_list = ['dns', 'ftp', 'http', 'pop3', 'smb', 'smtp', 'ssh']
    data, label = DataCollection_UN15.DataPayloadCollection(path, class_list)
    train_x, test_x, train_y, test_y = train_test_split(data, label, test_size=0.25, random_state=0, shuffle=True)

    return train_x, test_x, train_y, test_y
train_x, test_x, train_y, test_y = load_dataset_ids17_v3()
cluster_viz(test_x, test_y, save_dir, n_classes=4, fig_name='tsne_viz_IDS17V3.png')

km = KMeans(n_clusters=4, n_init=20, n_jobs=4)
y_pred = km.fit_predict(test_x)
print(' ' * 8 + '|==>  acc: %.4f,  nmi: %.4f,  ari: %.4f  <==|' % (metrics.acc(test_y, y_pred, num_cluster=4), metrics.nmi(test_y, y_pred), metrics.ari(test_y, y_pred)))


