import numpy as np
from sklearn.model_selection import train_test_split

def load_mnist():
    # the data, shuffled and split between train and test sets
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    print(x.shape)

    x = x.reshape(-1, 28, 28, 1).astype('float32')
    # print(x.shape)

    # x = x.reshape(x.shape[0], 784).astype('float32')
    # np.reshape()
    # print(x.shape)

    x = x/255.
    print(x.shape)
    # print('MNIST:', x.shape)
    return x, y

x, y = load_mnist()
print(x,y)

def load_dataset_ids17():
    import DataCollection_IDS
    path = '../data/IDS17_v3/'
    class_list = ['dns', 'ftp', 'http', 'smb']
    data, label = DataCollection_IDS.DataPayloadCollection(path, class_list)
    train_x, test_x, train_y, test_y = train_test_split(data, label, test_size=0.25, random_state=0, shuffle=True)
    x = np.concatenate((train_x, test_x))
    y = np.concatenate((train_y, test_y))
    print(x.shape)
    x = x.reshape(-1, 28, 28, 1).astype('float32')
    print(x.shape)
    # print('dataset shapes', x.shape)

    return x, y
    # return data,label

X, y = load_dataset_ids17()
print(X,y)