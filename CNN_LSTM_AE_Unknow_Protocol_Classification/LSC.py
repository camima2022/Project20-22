from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
import scipy.io as sio
import numpy as np

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
        kmeans = KMeans(n_clusters=self.k, random_state=0).fit(U_v)
        label = kmeans.labels_
        print('SUCCESS: finished refined')
        print(label)
        return label


if __name__=='__main__':
    np.random.seed(10)
    lsshc = LSSHC(mode_framework='eigen_trick', mode_Z_construction='knn', mode_Z_construction_knn='random',
                 mode_sampling_schema='HSV', mode_sampling_approach='probability', m_hyperedge=100, l_hyperedge =40,
                 knn_s=5, k=8)
    import DataCollection
    from sklearn.model_selection import train_test_split

    path = '/home/hbb/pythonProject/ClusterGAN_Unknow_Protocol_Classification/ISCX2012_1000_version/'
    class_list = ['bittorrent', 'dns', 'ftp', 'http', 'imap', 'smb', 'smtp', 'ssh']
    data, label = DataCollection.DataPayloadCollection_8class(path, class_list)
    pred_label = lsshc.fit(data)

    nmi = metrics.normalized_mutual_info_score(pred_label, label)
    print(nmi)

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


