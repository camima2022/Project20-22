import math
import copy
import numpy as np
from sklearn.cluster import DBSCAN
import sklearn.metrics.pairwise as pairwise


class Adapter_DBSCAN():

    # 默认统计聚类个数在2-25之间的聚类情况, 参数符合python左闭右开
    def __init__(self, num_cluster_range=(2, 26)):
        self.num_cluster_range = num_cluster_range

    def returnEpsCandidate(self, dataSet):
        """
        :param dataSet: 数据集
        :return: eps候选集合
        """
        # self.DistMatrix = self.CalculateDistMatrix(dataSet)
        self.DistMatrix = pairwise.euclidean_distances(dataSet)
        tmp_matrix = copy.deepcopy(self.DistMatrix)
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

    def returnMinptsCandidate(self, DistMatrix, EpsCandidate, X):
        """
        :param DistMatrix: 距离矩阵
        :param EpsCandidate: Eps候选列表
        :return: Minpts候选列表
        """
        MinptsCandidate = []
        for k in range(len(EpsCandidate)):
            tmp_eps = EpsCandidate[k]
            tmp_count = 0
            # for i in range(len(DistMatrix)):
            #     for j in range(len(DistMatrix[i])):
            #         if DistMatrix[i][j] <= tmp_eps:
            #             tmp_count = tmp_count + 1
            # 快250+倍
            tmp_count = np.sum(DistMatrix <= tmp_eps)
            MinptsCandidate.append(tmp_count / len(X))
        return MinptsCandidate

    def fit(self, X):
        self.EpsCandidate = self.returnEpsCandidate(X)
        self.MinptsCandidate = self.returnMinptsCandidate(self.DistMatrix, self.EpsCandidate, X)
        self.do_multi_dbscan(X)
        self.set_num_clusters_range(self.num_cluster_range)

    def do_multi_dbscan(self, X):
        self.all_predict_dict = {}
        self.all_param_dict = {}

        for i in range(len(self.EpsCandidate)):
            eps = self.EpsCandidate[i]
            minpts = self.MinptsCandidate[i]
            db = DBSCAN(eps=eps, min_samples=minpts).fit(X)
            num_clusters = max(db.labels_) + 1
            # 统计符合范围的聚类情况

            if num_clusters not in self.all_predict_dict.keys():
                self.all_predict_dict[num_clusters] = []
                self.all_param_dict[num_clusters] = []

            self.all_predict_dict[num_clusters].append(db.labels_)
            self.all_param_dict[num_clusters].append({"eps": eps, "minpts": minpts})

    # 筛选聚类个数，比如Multi-DBSCAN共产生了3聚类、15聚类、136聚类三种情况
    # 我想只看10～20的聚类情况，就可以设置set_num_clusters_range(10~21)后调用get_predict_dict()
    def set_num_clusters_range(self, num_cluster_range: tuple):
        self.predict_dict = {}
        self.param_dict = {}
        # 统计符合范围的聚类情况

        for num_cluster, labels, params in zip(self.all_predict_dict.keys(),self.all_predict_dict.values(), self.all_param_dict.values()):
            if num_cluster >= num_cluster_range[0] and num_cluster < num_cluster_range[1]:
                self.predict_dict[num_cluster] = labels
                self.param_dict[num_cluster] = params

    # 获取当前Multi-DBSCAN的聚类预测信息,格式为{聚类个数:[[预测可能1],[预测可能2],...]}
    # 比如聚类个数为3的情况可能有多种，所以有预测可能1、预测可能2...
    def get_predict_dict(self):
        if self.predict_dict is None:
            raise RuntimeError("get_predict_dict before fit")
        return self.predict_dict

    # 获取当前Multi-DBSCAN的聚类参数信息,格式为{聚类个数:[{"eps":x1,"minpts":y1},{"eps":x2,"minpts":y2},...]}
    def get_info_dict(self):
        if self.param_dict is None:
            raise RuntimeError("get_info_dict before fit")
        return self.param_dict


def loadDataSet(fileName):
    """
    输入：文件名
    输出：数据集
    描述：从文件读入数据集
    """
    dataSet = []
    with open(fileName) as fr:
        for line in fr.readlines():
            curline = line.strip().split(",")
            fltline = list(map(float, curline))
            dataSet.append(fltline)
    return np.array(dataSet)

def loadDataSet_protocol():
    import DataCollection
    from sklearn.model_selection import train_test_split
    path = '/home/hbb/pythonProject/CNN_LSTM_AE_Unknow_Protocol_Classification/ISCX2012_1000_version/'
    class_list = ['bittorrent', 'dns', 'ftp', 'http', 'imap', 'pop3', 'smb', 'smtp', 'ssh']
    datas, labels = DataCollection.DataPayloadCollection_9class(path, class_list)

    train_x, test_x, train_y, test_y = train_test_split(datas, labels, test_size=0.01, random_state=1234, shuffle=True)

    return test_x, test_y






if __name__ == '__main__':
    X= loadDataSet()
    DB = Adapter_DBSCAN()
    DB.fit(X)

    # 输出15聚类的情况
    DB.set_num_clusters_range((15, 16))
    # label预测信息
    predict_dict = DB.get_predict_dict()
    # 参数信息
    info_dict = DB.get_info_dict()

    print(predict_dict)
    print("================================")
    print(info_dict)
