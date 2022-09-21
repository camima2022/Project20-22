import math
import copy
import numpy as np
from sklearn.cluster import DBSCAN
import sklearn.metrics.pairwise as pairwise
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score


def returnMinptsCandidate(DistMatrix, EpsCandidate, X):
    """
    :param DistMatrix: 距离矩阵
    :param EpsCandidate: Eps候选列表
    :return: Minpts候选列表
    """
    MinptsCandidate = []
    for k in range(len(EpsCandidate)):
        tmp_eps = EpsCandidate[k]
        # for i in range(len(DistMatrix)):
        #     for j in range(len(DistMatrix[i])):
        #         if DistMatrix[i][j] <= tmp_eps:
        #             tmp_count = tmp_count + 1
        # 快250+倍
        tmp_count = np.sum(DistMatrix <= tmp_eps)
        MinptsCandidate.append(tmp_count / len(X))
    return MinptsCandidate

def getMetric(labels, labels_pred):
    ari = adjusted_rand_score(labels, labels_pred)
    nmi = normalized_mutual_info_score(labels, labels_pred)

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(labels, labels_pred)

    print(ari)
    print(nmi)
    print(cm)

def returnEpsCandidate(dataSet):
    """
    :param dataSet: 数据集
    :return: eps候选集合
    """
    # self.DistMatrix = self.CalculateDistMatrix(dataSet)
    DistMatrix = pairwise.euclidean_distances(dataSet)
    tmp_matrix = copy.deepcopy(DistMatrix)
    for i in range(len(tmp_matrix)):
        tmp_matrix[i].sort()
    EpsCandidate = []
    for k in range(1, len(dataSet)):
        # Dk = self.returnDk(tmp_matrix,k)
        Dk = tmp_matrix[:, k]
        # DkAverage = self.returnDkAverage(Dk)
        # 快160+倍
        DkAverage = np.mean(Dk)
        EpsCandidate.append(DkAverage)
    return EpsCandidate
def loadDataSet():
    import DataCollection
    from sklearn.model_selection import train_test_split
    path = '/home/hbb/pythonProject/CNN_LSTM_AE_Unknow_Protocol_Classification/ISCX2012_1000_version/'
    class_list = ['bittorrent', 'dns', 'ftp', 'http', 'imap', 'pop3', 'smb', 'smtp', 'ssh']
    datas, labels = DataCollection.DataPayloadCollection_9class(path, class_list)

    train_x, test_x, train_y, test_y = train_test_split(datas, labels, test_size=0.01, random_state=0, shuffle=True)

    return test_x, test_y

test_x, test_y = loadDataSet()
EpsCadidate = returnEpsCandidate(test_x)
print(test_x)
DistMatrix = pairwise.euclidean_distances(test_x)
MinptsCandidate = returnMinptsCandidate(DistMatrix, EpsCadidate, test_x)

k = 0
EpsCadidate[k] = 1.35
db = DBSCAN(eps=EpsCadidate[k], min_samples=MinptsCandidate[k]).fit(test_x)
print('Eps: ', EpsCadidate[k], " Minpts: ", MinptsCandidate[k])
print(db.labels_)
num_clusters = max(db.labels_) + 1
print(num_clusters)

print("Metrics: ")
getMetric(test_y, db.labels_)



