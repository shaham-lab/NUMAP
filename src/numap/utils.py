import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.manifold import SpectralEmbedding
from scipy.cluster.hierarchy import linkage, to_tree
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import eigh
import time
from scipy.sparse.linalg import eigsh
from scipy import sparse


def get_spectral_embedding(X=None, n_components=2, n_neighbors=5, affinity_matrix=None):
    if X is not None:
        se = SpectralEmbedding(n_components=n_components, eigen_solver='lobpcg', n_neighbors=n_neighbors)
        return torch.Tensor(se.fit_transform(X))
    else:
        se = SpectralEmbedding(n_components=n_components, eigen_solver='lobpcg', affinity='precomputed')
        return torch.Tensor(se.fit_transform(affinity_matrix))


def get_affinity_matrix(X, n_neighbors=5):
    dists, inds = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='kd_tree').fit(X).kneighbors(X)
    dists, inds = dists[:, 1:], inds[:, 1:]
    dists -= dists[:, 0].reshape(-1, 1)
    scales = np.mean(dists[:, 1:], axis=1).reshape(-1, 1)
    sims = np.exp(-dists ** 2 / scales ** 2)
    W = np.zeros((X.shape[0], X.shape[0]))
    W[np.arange(X.shape[0])[:, None], inds] = sims
    W = (W + W.T) / 2
    return W


def get_laplacian(X, n_neighbors=5):
    W = get_affinity_matrix(X, n_neighbors)
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    return L


def get_laplacian_eigs(X, n_neighbors=5, n_components=2):
    L = get_laplacian(X, n_neighbors)
    # eigvals, eigvecs = eigh(L)
    # return eigvals[:n_components], eigvecs[:, :n_components]
    start_time = time.time()
    sL = sparse.csr_matrix(L)
    print(f'Converting Laplacian to sparse time: {time.time() - start_time}')
    start_time = time.time()
    eigvals, eigvecs = eigsh(sL, k=n_components, which='SM')
    print(f'Computing Laplacian eigenvectors time: {time.time() - start_time}')
    return eigvals, eigvecs


def get_laplacian_eigenvectors(X, n_neighbors=5, n_components=2):
    # return get_laplacian_eigs(X, n_neighbors, n_components)[1]
    W = get_affinity_matrix(X, n_neighbors)
    return get_spectral_embedding(affinity_matrix=W)


def get_laplacian_spectrum(X, n_neighbors=5, n_components=2):
    return get_laplacian_eigs(X, n_neighbors, n_components)[0]


def isConstant(v: np.array):
    return np.all(v == v[0])


def plot_embedding(embedding, y, title, ax, cmap='tab10'):
    ax.scatter(embedding[:, 0], embedding[:, 1], c=y, cmap=cmap, s=1)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


###### Hirearchical clustering ######
def get_depth(node):
    if node is None or node.is_leaf():
        return 0
    return 1 + max(get_depth(node.get_left()), get_depth(node.get_right()))


def compute_depths(node, node_depths):
    if node.id not in node_depths:
        node_depths[node.id] = get_depth(node)
    if not node.is_leaf():
        compute_depths(node.get_left(), node_depths)
        compute_depths(node.get_right(), node_depths)


def build_parent_map(node, parent_map):
    if node.left:
        parent_map[node.left.id] = node
        build_parent_map(node.left, parent_map)
    if node.right:
        parent_map[node.right.id] = node
        build_parent_map(node.right, parent_map)


def lca(u, v, parent_map):
    ancestors_u = set()
    while u:
        ancestors_u.add(u.id)
        u = parent_map.get(u.id, None)
    while v:
        if v.id in ancestors_u:
            return v
        v = parent_map.get(v.id, None)
    return None


def _compute_depths_list(indices, nodes, node_depths, parent_map):
    depths = []
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            node1 = nodes[indices[i]]
            node2 = nodes[indices[j]]
            ancestor = lca(node1, node2, parent_map)
            if ancestor is not None:
                depths.append(node_depths[ancestor.id])
    return depths


def compute_depths_list(X: np.array):
    Z1 = linkage(X, method='ward')
    tree1, nodes1 = to_tree(Z1, rd=True)
    node_depths1 = {}
    parent_map1 = {}
    compute_depths(tree1, node_depths1)
    build_parent_map(tree1, parent_map1)
    indices1 = range(X.shape[0])
    return _compute_depths_list(indices1, nodes1, node_depths1, parent_map1)

######################################
