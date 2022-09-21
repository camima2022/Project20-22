from sklearn.cluster import KMeans, DBSCAN, OPTICS
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.model_selection import train_test_split
from time import time
from sklearn.cluster import SpectralClustering, KMeans

num_clusters = 6
def load_dataset_ids17():
    import DataCollection_IDS
    path = '/home/hbb/pythonProject/data/IDS17_v4/'
    class_list = ['dns', 'ftp', 'http', 'smb', 'ssh', 'tls']
    data, label = DataCollection_IDS.DataPayloadCollection(path, class_list)
    train_x, test_x, train_y, test_y = train_test_split(data, label, test_size=0.25, random_state=0, shuffle=True)
    x = np.concatenate((train_x, test_x))
    y = np.concatenate((train_y, test_y))
    print('dataset shapes', x.shape)

    return x, y
def load_dataset_un15():
    import DataCollection_UN15
    #from sklearn.model_selection import train_test_split
    path = '/home/hbb/pythonProject/NIN_BLSTM_Attention/UN15_7class_v2/'
    class_list = ['bittorrent', 'dns', 'ftp', 'http', 'imap', 'smtp', 'ssh']
    data, label = DataCollection_UN15.DataPayloadCollection(path, class_list)
    train_x, test_x, train_y, test_y = train_test_split(data, label, test_size=0.25, random_state=0, shuffle=True)
    x = np.concatenate((train_x, test_x))
    y = np.concatenate((train_y, test_y))
    x = x.reshape((x.shape[0], 128, 1))
    # print('dataset shapes', x.shape)

    return x, y
def load_dataset_iscx12():
    import DataCollection_ISCX12
    #from sklearn.model_selection import train_test_split
    path = '/home/hbb/pythonProject/NIN_BLSTM_Attention/ISCX12_14/'
    class_list = ['BiTtorrent', 'DNS', 'FTP', 'HTTP', 'IMAP', 'NBNS', 'POP', 'SSH', 'TLS']
    data, label = DataCollection_ISCX12.DataPayloadCollection(path, class_list)
    train_x, test_x, train_y, test_y = train_test_split(data, label, test_size=0.25, random_state=0, shuffle=True)
    x = np.concatenate((train_x, test_x))
    y = np.concatenate((train_y, test_y))
    #x = x.reshape((x.shape[0], 128, 1))
    # print('dataset shapes', x.shape)

    return x, y
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

X, y = load_dataset_ids17()
# print(x)
res = list()

km = KMeans(n_clusters=num_clusters, n_init=20, n_jobs=4)
for i in range(50):
    t0 = time()
    km_pred = km.fit_predict(X)
    _, all_acc = cluster_acc(km_pred, y)
    print(' ' * 8 + '|==>  acc: %.4f,  nmi: %.4f,  ari: %.4f  <==|'
                  % (all_acc, metrics.normalized_mutual_info_score(y, km_pred), metrics.adjusted_rand_score(y, km_pred)))
    print('clustering time: ', (time() - t0))
    one = [i, all_acc, metrics.normalized_mutual_info_score(y, km_pred), metrics.adjusted_rand_score(y, km_pred)]
    res.append(one)

# E = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
# E = [i for i in np.arange(0, 0.2, 0.01)]
# for eps in E:
#     for minpts in range(5, 26):
#         one = list()
#         t1 = time()
#         db = DBSCAN(eps=eps, min_samples=minpts, metric='euclidean')
#         db_pred = db.fit_predict(X)
#         _, all_acc = cluster_acc(db_pred, y)
#         print('EPS = ', eps, ' Min_samples = ', minpts)
#         print(' ' * 8 + '|==>  acc: %.4f,  nmi: %.4f,  ari: %.4f  <==|'
#               % (all_acc, metrics.normalized_mutual_info_score(y, db_pred), metrics.adjusted_rand_score(y, db_pred)))
#         print('clustering time: ', (time() - t1))
#         one = [eps, minpts, all_acc, metrics.normalized_mutual_info_score(y, db_pred), metrics.adjusted_rand_score(y, db_pred)]
#         res.append(one)



# op = OPTICS(min_samples=5)
# op_pred = op.fit_predict(x)
# _, op_acc = cluster_acc(op_pred, y)
# print(' ' * 8 + '|==>  acc: %.4f,  nmi: %.4f,  ari: %.4f  <==|'
#       % (op_acc, metrics.nmi(y, op_pred), metrics.ari(y, op_pred)))
# from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
# sc = silhouette_score
# ari = adjusted_rand_score
# nmi = normalized_mutual_info_score
# ch = metrics.calinski_harabasz_score
#
# for i, gamma in enumerate((0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1)):
#     for j, k in enumerate((3, 4, 5, 6, 7, 8)):
#         y_pred = SpectralClustering(n_clusters=k, gamma=gamma).fit_predict(X)
#         print('k:', k, 'gamma:', gamma)
#         print('NMI: ', nmi(y, y_pred), '\n', "ARI: ", ari(y, y_pred), '\n', "SC: ", sc(X, y_pred), '\n', "Calinski-Harabasz Score", ch(X, y_pred))

import csv
newfile = open('result/kmeans_ids17res_v4.csv', 'w', newline='')
filewriter = csv.writer(newfile)
# list1 = [[1,2,3,4],[1,2,3,4],[5,6,7,8]]
for s in res:
    filewriter.writerow(s)
newfile.close()