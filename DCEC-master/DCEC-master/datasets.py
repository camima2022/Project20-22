import numpy as np
from sklearn.model_selection import train_test_split


def load_mnist():
    # the data, shuffled and split between train and test sets
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape((x.shape[0], 28, 28, 1)).astype('float32')
    x = x/255.
    print('MNIST:', x.shape)
    return x, y


def load_usps(data_path='./data/usps'):
    import os
    if not os.path.exists(data_path+'/usps_train.jf'):
        if not os.path.exists(data_path+'/usps_train.jf.gz'):
            os.system('wget http://www-i6.informatik.rwth-aachen.de/~keysers/usps_train.jf.gz -P %s' % data_path)
            os.system('wget http://www-i6.informatik.rwth-aachen.de/~keysers/usps_test.jf.gz -P %s' % data_path)
        os.system('gunzip %s/usps_train.jf.gz' % data_path)
        os.system('gunzip %s/usps_test.jf.gz' % data_path)

    with open(data_path + '/usps_train.jf') as f:
        data = f.readlines()
    data = data[1:-1]
    data = [list(map(float, line.split())) for line in data]
    data = np.array(data)
    data_train, labels_train = data[:, 1:], data[:, 0]

    with open(data_path + '/usps_test.jf') as f:
        data = f.readlines()
    data = data[1:-1]
    data = [list(map(float, line.split())) for line in data]
    data = np.array(data)
    data_test, labels_test = data[:, 1:], data[:, 0]

    x = np.concatenate((data_train, data_test)).astype('float32')
    x /= 2.0
    x = x.reshape([-1, 16, 16, 1])
    y = np.concatenate((labels_train, labels_test))
    print('USPS samples', x.shape)
    return x, y


def load_dataset_ids17():
    import DataCollection_IDS
    path = '/home/hbb/pythonProject/data/IDS17_v3/'
    class_list = ['dns', 'ftp', 'http', 'smb']
    data, label = DataCollection_IDS.DataPayloadCollection(path, class_list)
    train_x, test_x, train_y, test_y = train_test_split(data, label, test_size=0.25, random_state=0, shuffle=True)
    x = np.concatenate((train_x, test_x))
    y = np.concatenate((train_y, test_y))
    # print(x.shape)
    # x = x.reshape(-1, 28, 28, 1).astype('float32')
    x = x.reshape((x.shape[0], 12, 12, 1)).astype('float32')
    # print(x.shape)
    print('dataset shapes', x.shape)

    return x, y
# from sklearn.cluster import KMeans
#
# km = KMeans(n_clusters=4)
# x, y = load_dataset_ids17()
# km_labels = km.fit_predict(x)
# import metrics
# acc = np.round(metrics.acc(y, km_labels), 5)
# nmi = np.round(metrics.nmi(y, km_labels), 5)
# ari = np.round(metrics.ari(y, km_labels), 5)
#
# print('Acc', acc, ', nmi', nmi, ', ari', ari)

# x, y = load_dataset_ids17()
# print(x.shape[0])
# print(x[45512::])
    # return data,label