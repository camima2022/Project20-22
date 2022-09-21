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
"""
def DensityCanopyKmeans():
    start = time()
    centers = np.asarray(densityCanopy())
    k = len(centers)
    print(centers)
    print(k)
    kmeans = KMeans(n_clusters=k, init=centers, n_init=1, max_iter=100).fit(data)

    getMetrics(target, kmeans.labels_)

    end = time()
    clustering_time = end - start
    print('clustering time: ', clustering_time)


def run():
    # demo_data = np.random.random_sample((2000, 5))
    # print(demo_data.shape)
    #     data = auxData
    #     demo_data = data.values
    clustering_time = []
    clustering_error = []
    k = 9

    # print(target)

    for kmeansType in kmeansTypes:
        start = time()
        if kmeansType == "densityCanopy":
            centers = np.asarray(densityCanopy())
            print("Density Canopy Kmeans")
            print(centers)
            l = len(centers)
            print(l)
            kmeans = KMeans(n_clusters=l, init=centers, n_init=1, max_iter=100).fit(data)
        else:

            kmeans = KMeans(n_clusters=k, random_state=200, init=kmeansType, n_init=1, max_iter=100).fit(data)

        getMetrics(target, kmeans.labels_)
        # print(kmeans.labels_)
        error = getSquaredError(data, kmeans, k)
        clustering_error.append(error)
        # print("Erro:", error)

        end = time()
        clustering_time.append(end - start)

    print("\n---Tempos---")
    print("random: ", clustering_time[0], " kmeans++: ", clustering_time[1], " densityCanopy: ", clustering_time[2])

    print("\n---Erros---")
    print("random: ", clustering_error[0], "kmeans++: ", clustering_error[1], " densityCanopy: ", clustering_error[2])

DensityCanopyKmeans()
"""
