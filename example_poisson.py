import numpy as np
import networkx as nx

from fusenet.model import glm


def get_random_poisson(A, n=200, lambd_true=1.0, lambd_noise=0.5):
    p = A.shape[0]
    p1 = int(p+p*(p-1.)/2.)
    sz_tmp = n*p1
    Y = np.random.poisson(lambd_true, sz_tmp).reshape((n, p1))
    E = np.random.poisson(lambd_noise, n*p).reshape((n, p))
    vec_triA = np.zeros((p*(p - 1)/2, 1))
    k = 0
    for j in xrange(A.shape[0]):
        for i in xrange(0, j):
            vec_triA[k, 0] = A[i, j]
            k += 1
    B1 = np.eye(p)
    one_p = np.ones((p, 1))
    P = np.eye(p, p*(p - 1)/2)
    perm = np.random.permutation(p)
    P = P[perm, :]
    B2 = np.multiply(P, np.dot(one_p, vec_triA.T))
    B = np.hstack((B1, B2)).T
    X = np.dot(Y, B) + E
    return X


n_examples = 100
n_features = 10

# Preferential attachment dependency graph
dependency_graph = nx.barabasi_albert_graph(n_features, 2)
true_dependency_matrix = np.array(nx.adjacency_matrix(dependency_graph).todense())
data = get_random_poisson(true_dependency_matrix, n=n_examples)

clf = glm.PoissonFNet(alpha=0.01, l1_ratio=0, verbose=False, max_iter=2)
cls = clf.fit(data, k=3)
dependency_matrix = cls.D_
print dependency_matrix
