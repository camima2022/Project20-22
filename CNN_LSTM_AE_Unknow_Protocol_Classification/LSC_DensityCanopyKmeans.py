from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
import scipy.io as sio
import numpy as np
import DensityBasedCanopy_Kmeans
from sklearn.metrics.pairwise import pairwise_distances
from functools import reduce
from time import time
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import jaccard_score # scikit 0.21
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import accuracy_score
import DataCollection
from sklearn.model_selection import train_test_split
"""
# prepare dataset
save_dir = '/home/hbb/pythonProject/CNN_LSTM_AE_Unknow_Protocol_Classification/saved_model'
path = '/home/hbb/pythonProject/CNN_LSTM_AE_Unknow_Protocol_Classification/ISCX2012_1000_version/'

class_list = ['bittorrent', 'dns', 'ftp', 'http', 'imap', 'pop3', 'smb', 'smtp', 'ssh']
nb_classes = 9
datas, labels = DataCollection.DataPayloadCollection_9class(path, class_list)
train_x, test_x, train_y, test_y = train_test_split(datas, labels, test_size= 0.25, random_state=0, shuffle=True)
data = test_x
target = test_y
distance_matrix = pairwise_distances(data, metric='cosine')
kmeansTypes = ["random", "k-means++", "densityCanopy"]
"""
def canopy(X, T1, T2, distance_metric='euclidean', filemap=None):
    canopies = dict()
    X1_dist = pairwise_distances(X, metric=distance_metric)
    canopy_points = set(range(X.shape[0]))
    while canopy_points:
        point = canopy_points.pop()
        i = len(canopies)
        canopies[i] = {"c":point, "points": list(np.where(X1_dist[point] < T2)[0])}
        canopy_points = canopy_points.difference(set(np.where(X1_dist[point] < T1)[0]))
    if filemap:
        for canopy_id in canopies.keys():
            canopy = canopies.pop(canopy_id)
            canopy2 = {"c":filemap[canopy['c']], "points":list()}
            for point in canopy['points']:
                canopy2["points"].append(filemap[point])
            canopies[canopy_id] = canopy2
    return canopies

def getMetrics(y_true, y_pred):
    #y_true = y_true.map(
    #    {"D1": 0, "D2": 1, "D3": 2, "D4": 3, "b": 0, "g": 1, 1: 0, 2: 1, 3: 2, "Iris-setosa": 0, "Iris-versicolor": 1,
    #     "Iris-virginica": 2})

    # print(y_true)
    print("Jaccard: ", jaccard_score(y_true, y_pred, average="macro"))
    print("AdjustedRandIndex: ", adjusted_rand_score(y_true, y_pred))
    print("Accuracy: ", accuracy_score(y_true, y_pred))

def euclideanDistance(vector1, vector2):
    # print(vector1)
    # print(vector2)
    return np.sqrt(np.sum(np.power(vector1 - vector2, 2)))

def getDistance(row_center, row_sample):
    # print(row_center)
    row_center = np.asarray(row_center)
    # row_center = np.asarray(row_sample)
    return euclideanDistance(row_center, row_sample)

def getSquaredError(data, kmeans, k):
    distances = []
    for i in range(k): # Qtd de clusters
        distance = 0
        for index_labels, value_labels in enumerate(kmeans.labels_): #kmeans.labels_ possui o cluster de cada elemento
            if (i == value_labels):
                #print(value_labels)
                distance = distance + getDistance(kmeans.cluster_centers_[value_labels], data[index_labels])
        distances.append(distance) #Erro quadratico medio de cada cluster
    distances = np.asarray(distances)
    error = np.sum(distances)
    return error

def get_mean_distance(distance_matrix):
    n = len(distance_matrix)
    return distance_matrix.sum() / (n * (n-1)) if n > 1 else distance_matrix.sum()

def get_density(distance_matrix, mean_distance):
    densities = np.zeros(len(distance_matrix))

    matrix = distance_matrix - mean_distance
    matrix[matrix >= 0] = 0
    matrix[matrix < 0] = 1

    for i, row_i in enumerate(matrix):
        densities[i] = row_i.sum()
    return densities

def cluster_tightness(distance_matrix, mean_distance):
    tightness = np.zeros(len(distance_matrix))

    for i, row_i in enumerate(distance_matrix):
        mean = row_i[row_i < mean_distance].mean()
        tightness[i] = 0 if mean == 0 else 1 / mean
    return tightness

def clusters_dissimilarity(densities, distance_matrix):
    n = len(distance_matrix)
    dissimilarity = np.zeros(n)
    for i in range(n):
        d_row = distance_matrix[i]
        d_row = np.delete(d_row, np.where(d_row == 0))
        dissimilarity[i] = d_row.min()
    max_id = np.where(densities == densities.max())
    dissimilarity[max_id] = distance_matrix[max_id].max()
    return dissimilarity

def remove_cluster(c, data, distance_matrix, mean_distance):
    center_row = distance_matrix[c]
    center_row[center_row < mean_distance] = 0

    # pontos a serem removidos
    points = np.where(center_row == 0)
    data = np.delete(data, points, axis=0)

    distance_matrix = np.delete(distance_matrix, points, axis=0)
    distance_matrix = np.delete(distance_matrix, points, axis=1)

    return data, distance_matrix

def removeOutliers(aux_D, densities, inv_a, s, meanDis):
    #remove elemento com densidade = 1 e que o s[i] seja maior que o raio
    outliers = []
    for i, row_i in enumerate(aux_D.values):
        if densities[i] == 1 and s[i] == max(s) and aux_D.shape[0] > 1:
            outliers.append(i)
    aux_D.drop(outliers, inplace=True) #removendo outliers
    aux_D.reset_index(drop=True, inplace=True)
    densities = np.delete(densities, outliers, 0)
    inv_a = np.delete(inv_a, outliers, 0)
    s = np.delete(s, outliers, 0)
    return aux_D, densities, inv_a, s


def densityCanopy(data, distance_matrix):
    print("\n----- -----")
    centers = []
    db = data.copy()
    d_matrix = distance_matrix.copy()

    # Step 1
    mean_distance = get_mean_distance(d_matrix)
    density_set = get_density(d_matrix, mean_distance)
    c = np.argmax(density_set)
    centers.append(db[c])
    db, d_matrix = remove_cluster(c, db, d_matrix, mean_distance)

    # Step 2
    mean_distance = get_mean_distance(d_matrix)
    density_set = get_density(d_matrix, mean_distance)
    tightness_set = cluster_tightness(d_matrix, mean_distance)
    dissimilarity_set = clusters_dissimilarity(density_set, d_matrix)
    w_set = density_set * tightness_set * dissimilarity_set  # Product Weight
    c = np.argmax(w_set)
    centers.append(db[c])
    db, d_matrix = remove_cluster(c, db, d_matrix, mean_distance)

    # Step 3
    while len(db) > 1:

        mean_distance = get_mean_distance(d_matrix)
        density_set = get_density(d_matrix, mean_distance)
        tightness_set = cluster_tightness(d_matrix, mean_distance)

        # Distancia das instancias para os centros já escolhidos
        clusters_distance = pairwise_distances(db, centers, metric='euclidean')

        w_set = density_set * tightness_set
        for i in range(len(db)):
            w_set[i] = reduce(lambda x, y: x * y, clusters_distance[i] * w_set[i])

        c = np.argmax(w_set)
        centers.append(db[c])
        db, d_matrix = remove_cluster(c, db, d_matrix, mean_distance)

    if len(db) == 1:
        centers.append(db[0])

    return centers

class LSSHC:
    def __init__(self, mode_framework='edge_sampling', mode_Z_construction='knn', mode_Z_construction_knn='kmeans',
                 mode_sampling_schema='HSV', mode_sampling_approach='probability', m_hyperedge=6, l_hyperedge =4,
                 knn_s=3, k=2):
        '''

        :param mode_framework: 'tradition' / 'eigen_trick' / 'edge_sampling'
        :param mode_Z_construction: 'origin' / 'knn'
        :param mode_Z_construction_knn: 'random' / 'kmeans'
        :param mode_sampling_schema: 'HS' / 'HSE' / 'HSV'
        :param mode_sampling_approach: 'random' / 'kmeans' / 'probability'
        :param m_hyperedge: the number of hyperedges
        :param l_hyperedge: the number of the sampled hyperedges
        :param knn_s: s - nearest neighbor for hyperedges construction using knn
        :param k: the number of clusters
        '''
        self.mode_framework = mode_framework
        self.mode_Z_construction = mode_Z_construction
        self.mode_Z_construction_knn = mode_Z_construction_knn
        self.mode_sampling_schema = mode_sampling_schema
        self.mode_sampling_approach = mode_sampling_approach
        self.m = m_hyperedge
        self.l = l_hyperedge
        self.knn_s = knn_s
        self.k = k
    # select M hyperedges n dataset
    def construct_hyperedges(self, X):
        n = X.shape[0]
        hyperedges_x = 0
        if self.mode_Z_construction_knn == 'random':
            rand_idx = np.array(range(0, n))
            np.random.shuffle(rand_idx)
            sampled_idx = rand_idx[0:self.m]
            hyperedges_x = X[list(sampled_idx), :]
        elif self.mode_Z_construction_knn == 'kmeans':
            kmeans = KMeans(n_clusters=self.m, random_state=0, max_iter=10).fit(X)
            hyperedges_x = kmeans.cluster_centers_
        else:
            print('ERROR: the parameter of mode_Z_construction_knn is not matched.')
            exit(0)
        return hyperedges_x # #


    def construct_Z(self, X, W_e=None):
        n = X.shape[0]
        if W_e == None:
            W_e = np.ones(self.m)
        if self.mode_Z_construction == 'origin':
            Z = X
        elif self.mode_Z_construction == 'knn':
            hyperedges_x = self.construct_hyperedges(X)
            nbrs = NearestNeighbors(n_neighbors=self.knn_s, algorithm='ball_tree').fit(hyperedges_x)
            distances, indices = nbrs.kneighbors(X)
            # build sparse matrix
            indptr = np.arange(0, (n+1) * self.knn_s, self.knn_s)
            indices = indices.flatten()
            distances = distances.flatten()
            distances = np.exp(-distances**2/(np.mean(distances)**2))
            Z = csr_matrix((distances, indices, indptr), shape=(n, self.m))
        else:
            print('ERROR: the parameter of mode_Z_construction is not matched.')
            exit(0)

        # Normalized Z
        D_v = np.sum(Z.multiply(csr_matrix(W_e)), 1)
        D_e = np.sum(Z, 0)
        Z = csr_matrix(1/np.sqrt(D_v)).multiply(Z).multiply(csr_matrix(1/np.sqrt(D_e))).multiply(csr_matrix(W_e))
        print('SUCCESS: Z construction')
        return Z

    def sampling_Z(self, Z):
        if self.mode_framework == 'tradition' or self.mode_framework == 'eigen_trick':
            Z_l = Z
            Z_ll = Z
        elif self.mode_framework == 'edge_sampling':
            Z = Z.toarray()
            if self.mode_sampling_approach == 'random':
                rand_idx = np.array(range(0, self.m))
                np.random.shuffle(rand_idx)
                sampled_idx = rand_idx[0: self.l]
                Z_l = Z[:, list(sampled_idx)]
            elif self.mode_sampling_approach == 'kmeans':
                kmeans = KMeans(n_clusters=self.l, random_state=0, max_iter=10).fit(Z.T)
                Z_l = kmeans.cluster_centers_.T
            elif self.mode_sampling_approach == 'probability':
                Z_sqr = Z ** 2
                D = np.sum(Z_sqr, 0)
                sum_D = np.sum(D)
                prob = D/sum_D
                idx = list(range(0, self.m))
                sampled_idx = np.random.choice(idx, size=self.l, replace=True, p=prob)
                Z_l = Z[:, sampled_idx]
                # Normalized
                t = np.sqrt(prob[sampled_idx]*self.l)
                Z_l = np.divide(Z_l, t)
            else:
                print('ERROR: the parameter of mode_framework is not matched.')
                exit(0)

            # Check the isolated nodes
            D_v_l = np.sum(Z_l, 1)
            zero_idx = np.where(D_v_l == 0)[0]
            # if zero_idx.size != 0:
            #     print('ERROR: there are isolated nodes after sampling,')
            #     exit(0)

            if self.mode_sampling_schema == 'HS':  # Z_l' = Z_l
                Z_ll = Z_l
            elif self.mode_sampling_schema == 'HSE':  # Z_l' = Z^T*Z_l
                Z_ll = Z.T.dot(Z_l)
            elif self.mode_sampling_schema == 'HSV':  # Z_l'= Z*Z^T*Z_l
                Z_ll = Z.dot(Z.T.dot(Z_l))
            else:
                print('ERROR: the parameter of mode_sampling_schema is not matched.')
                exit(0)

        else:
            print('ERROR: the parameter of mode_framework is not matched.')
            exit(0)

        print('SUCCESS: Z_l sampling')
        # Z_l = csr_matrix(Z_l)

        return Z_l, Z_ll

    def eign_refine(self, Z, Z_l, U, E):
        if self.mode_framework == 'tradition':
            U_v = U
        elif self.mode_framework == 'eigen_trick':
            U_e = U
            E = np.diag(E)
            U_v = Z.dot(U_e.dot(E))
        elif self.mode_framework == 'edge_sampling':
            U_l = U
            E = np.diag(E)
            t = Z_l.dot(U_l)
            U_e = Z.T.dot(t)
            U_v = Z.dot(U_e.dot(E))
        return U_v

    def fit(self, X):
        # 1. Hypergraph instance matrix construction (Z)
        Z = self.construct_Z(X)
        print('SUCCESS: construct_Z')
        Z_l, Z_ll = self.sampling_Z(Z)
        print('SUCCESS: sampling_Z')
        # Z_l = Z_l.toarray()  # the multiplication operation of numpy is faster than scipy

        # 2. Laplacian Construction
        if self.mode_framework == 'tradition':
            L = np.dot(Z_ll, Z_ll.T)  # L = Z * Z^T
        else:
            L = np.dot(Z_ll.T, Z_ll)  # L_e = Z

        print('SUCCESS:c construct L')

        # 3. Eigenvector Solving
        E, U = eigsh(L, k=self.k+1)

        print('SUCCESS: compute eigen')
        print('eigenvalue')
        print(E)
        # Remove the largest eigen
        U = U[:, 0:-1]
        E = E[0:-1]

        # 4. Eigenvector Refine
        U_v = self.eign_refine(Z, Z_l, U, E)
        print('SUCCESS: refine eigen')

        # 5. k-means

        from sklearn import preprocessing
        min_max_scaler = preprocessing.MinMaxScaler()
        X_minMax = min_max_scaler.fit_transform(U_v)
        #kmeans = KMeans(n_clusters=self.k, random_state=0).fit(U_v)
        kmeans = KMeans(n_clusters=self.k, random_state=0).fit(X_minMax)
        label = kmeans.labels_
        print('SUCCESS: finished refined')
        print(label)
        print('Spectral Canopy Kmeans: ')

        distance_matrix = pairwise_distances(X_minMax, metric='euclidean')
        centers = np.asarray(densityCanopy(X_minMax, distance_matrix))
        print(centers)
        k = len(centers)
        densityCanopyKmeans = KMeans(n_clusters=k, init=centers, n_init=1, max_iter=100).fit(X_minMax)

        pred_label = densityCanopyKmeans.labels_
        print(pred_label)

        return pred_label

"""
if __name__=='__main__':
    np.random.seed(10)
    lsshc = LSSHC(mode_framework='eigen_trick', mode_Z_construction='knn', mode_Z_construction_knn='random',
                 mode_sampling_schema='HSV', mode_sampling_approach='probability', m_hyperedge=100, l_hyperedge =40,
                 knn_s=5, k=8)
    import DataCollection
    from sklearn.model_selection import train_test_split

    path = '/home/hbb/pythonProject/CNN_LSTM_AE_Unknow_Protocol_Classification/ISCX2012_1000_version/'
    class_list = ['bittorrent', 'dns', 'ftp', 'http', 'imap', 'pop3', 'smb', 'smtp', 'ssh']
    data, label = DataCollection.DataPayloadCollection_9class(path, class_list)
    train_x, test_x, train_y, test_y = train_test_split(data, label, test_size=0.25, random_state=0, shuffle=True)

    pred_label = lsshc.fit(test_x)

    nmi = metrics.normalized_mutual_info_score(pred_label, test_y)
    ari = metrics.adjusted_rand_score(pred_label, test_y)
    cm = metrics.confusion_matrix(pred_label, test_y)
    print(nmi)
    print(ari)
    print(cm)
    #def load_dataset(path, class_list):
        #data, label = DataCollection.DataPayloadCollection_8class(path, class_list)
        #x_train, x_test, y_train, y_test = train_test_split(data, label, random_state=0, test_size=0.2)
        #x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')
        #x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')

    #    return (x_train, y_train), (x_test, y_test)

    '''
    :param mode_framework: 'tradition' / 'eigen_trick' / 'edge_sampling'
    :param mode_Z_construction: 'origin' / 'knn'
    :param mode_Z_construction_knn: 'random' / 'kmeans'
    :param mode_sampling_schema: 'HS' / 'HSE' / 'HSV'
    :param mode_sampling_approach: 'random' / 'kmeans' / 'probability'
    :param m_hyperedge: number of hyperedges
    :param l_hyperedge: number of the sampled hyperedges
    :param knn_s: s - nearest neighbor for hyperedges construction using knn
    '''
    # lsshc.construct_Z()
    # X = np.array([[1,2], [2,3], [3,4], [9,8], [8,7], [7,6], [5,6], [6,7], [7,8], [9,9]])
    #load_fn = 'USPS.mat'
    #load_data = sio.loadmat(load_fn)

    #X = load_data['fea']
    #Y = load_data['gnd'].flatten()
    #X = X.toarray()
    #print(X[0])

    # Y = np.array([1,2])
    # Z = np.sqrt(X * Y)
    # print(Z)
    #pred_label = lsshc.fit(X.toarray())
    #nmi = metrics.normalized_mutual_info_score(pred_label, Y)
    #print(nmi)

"""
