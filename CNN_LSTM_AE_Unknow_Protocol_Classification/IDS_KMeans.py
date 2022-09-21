from sklearn.cluster import KMeans, DBSCAN, OPTICS
import numpy as np
import metrics

num_clusters = 4
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

x, y = load_dataset_ids17()
km = KMeans(n_clusters=num_clusters, n_init=20, n_jobs=4)
km_pred = km.fit_predict(x)
print(' ' * 8 + '|==>  acc: %.4f,  nmi: %.4f,  ari: %.4f  <==|'
      % (metrics.acc(y, km_pred, num_cluster=num_clusters), metrics.nmi(y, km_pred), metrics.ari(y, km_pred)))

db = DBSCAN(eps=1, min_samples=25, metric='euclidean')
db_pred = db.fit_predict(x)
_, all_acc = cluster_acc(db_pred, y)
print(' ' * 8 + '|==>  acc: %.4f,  nmi: %.4f,  ari: %.4f  <==|'
      % (all_acc, metrics.nmi(y, db_pred), metrics.ari(y, db_pred)))

op = OPTICS(min_samples=5)
op_pred = op.fit_predict(x)
_, op_acc = cluster_acc(op_pred, y)
print(' ' * 8 + '|==>  acc: %.4f,  nmi: %.4f,  ari: %.4f  <==|'
      % (op_acc, metrics.nmi(y, op_pred), metrics.ari(y, op_pred)))
