from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD, Adam
import Models
import DataCollection
import warnings
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
warnings.filterwarnings('ignore')

def cluster_latent(latent, labels, n_classes=9):
    '''
    :param latent: 2-D array of predicted latent vectors for each sample
    :param labels: vector of true labels
    :param n_classes: integer for number of classes
    :return: evaluation of prediction after applying K-means on the predicted latent vectors
    '''
    km = KMeans(n_clusters=n_classes, random_state=0).fit(latent)
    labels_pred = km.labels_

    acc_c, acc_all = cluster_acc(labels_pred, labels)
    ari = adjusted_rand_score(labels, labels_pred)
    nmi = normalized_mutual_info_score(labels, labels_pred)

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(labels, labels_pred)

    print(acc_c)
    print(acc_all)
    print(ari)
    print(nmi)
    print(cm)

    return acc_c, acc_all, ari, nmi
def cluster_acc(pred, true):
    '''
    :param pred: predicted cluster
    :param true: true labels
    :return: classification accuracy for complete data and per class
    '''
    c_dict = {}
    all_pred = 0
    for c in np.unique(true):
        c_ind = np.where(true == c)
        true_c = true[c_ind]
        pred_c = pred[c_ind]
        pred_c_unique = np.unique(pred_c, return_counts=True)
        pred_c_max = pred_c_unique[0][pred_c_unique[1].argmax()]
        num_pred_c = pred_c_unique[1].max()
        all_pred += num_pred_c

        c_dict[str(c)] = [str(pred_c_max), num_pred_c / len(true_c)]

    return c_dict, all_pred / len(true)
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

save_dir = '/home/hbb/pythonProject/CNN_LSTM_AE_Unknow_Protocol_Classification/saved_model'

nb_classes = 9
path = '/home/hbb/pythonProject/CNN_LSTM_AE_Unknow_Protocol_Classification/ISCX2012_1000_version/'

class_list = ['bittorrent', 'dns', 'ftp', 'http', 'imap', 'pop3', 'smb', 'smtp', 'ssh']

data, label = DataCollection.DataPayloadCollection_9class(path, class_list)

train_x, test_x, train_y, test_y = train_test_split(data, label, test_size= 0.25, random_state=0, shuffle=True)
#cluster_viz(test_x, test_y, save_dir, n_classes=9, fig_name='tsne_viz_raw_data.png')

train_x = train_x.reshape((train_x.shape[0], 784, 1))
test_x = test_x.reshape((test_x.shape[0], 784, 1))

autoencoder = Models.AutoEncoder()
autoencoder.model.summary()
model_path = 'saved_model/autoencoder_our.h5'
checkpoint = ModelCheckpoint(model_path, monitor="val_loss", verbose=1, save_best_only=True, mode="auto", save_weights_only=True)

early = EarlyStopping(monitor="val_loss", mode="auto", patience=10)

callbacks_list = [checkpoint, early]
#sgd = SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True)
adam = Adam(learning_rate=0.001)
autoencoder.model.compile(loss='mse',optimizer=adam, metrics=['accuracy'])
history = autoencoder.model.fit(train_x, train_x, batch_size=64, epochs = 10, verbose = 2, validation_data=(test_x, test_x), callbacks=callbacks_list)
score = autoencoder.model.evaluate(test_x, test_x, verbose=1)
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

result_path = '/home/hbb/pythonProject/CNN_LSTM_AE_Unknow_Protocol_Classification/saved_model/'
fig = plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
fig.savefig(result_path+'/pretraining_acc_our.png')
fig1 = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower left')
fig1.savefig(result_path+'/pretraining_loss_our.png')

import matplotlib.pyplot as plt

encoder = autoencoder.Encoder()

decoded_imgs = encoder.predict(test_x)

print(decoded_imgs.shape)

cluster_viz(decoded_imgs, test_y, save_dir, n_classes=9)
cluster_latent(decoded_imgs, test_y, n_classes=9)

"""
#encoded_imgs = encoder.predict(x_test)
#decoded_imgs = decoder.predict(encoded_imgs)
n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n):
# 展示原始图像
    ax = plt.subplot(2, n, i)
    plt.imshow(test_x[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # 展示自编码器重构后的图像
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig(result_path + '/raw_decoded_images_our.jpg')

"""
"""

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000) # TSNE降维，降到2
# 只需要显示前500个
plot_only = 500
# 降维后的数据
low_dim_embs = tsne.fit_transform(compressed_train_x[:plot_only, :])
# 标签
labels = train_y[:plot_only]
print(labels)
plot_with_labels(low_dim_embs, labels)
"""
"""
from sklearn import datasets
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

#digits = datasets.load_digits(n_class=5)
#X = digits.data
#y = digits.target
X = compressed_train_x[:1000,:]
y = train_y[:1000]
print(X.shape)  # 901,64


def plot_embedding_2d(X, y, title=None):
    #Plot an embedding X with the class label y colored by the domain d.
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # Plot colors numbers
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        # plot colored number
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set3(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    plt.savefig("tnse_trail.jpg")



print("Computing t-SNE embedding")
tsne2d = TSNE(n_components=2, init='pca', random_state=0)

X_tsne_2d = tsne2d.fit_transform(X)
plot_embedding_2d(X_tsne_2d[:, 0:2], y, "t-SNE 2D")



"""

