import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch


def get_affinity_matrix(X, n_neighbors=5):
    """This function computes the affinity matrix W using the Laplacian kernel."""
    Dx = torch.cdist(X, X)
    Dis, indices = get_nearest_neighbors(X, k=n_neighbors + 1)
    Dis -= Dis[:, 0].reshape(Dis.shape[0], 1)
    Dis[Dis < 0] = 0
    scale = compute_scale(Dis)
    W = torch.Tensor(get_laplace_kernel(Dx, scale, indices[:, 1:], device='cpu'))

    # normalize the affinity matrix - random walk normalization
    W /= W.sum(dim=1).reshape(W.shape[0], 1)

    W = W + W.t() - W * W.t()

    return W


def get_nearest_neighbors(X, k=5):
    """This function computes the k-nearest neighbors for each data point in X."""
    X = X.detach().numpy()
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)
    return distances[:, 1:], indices[:, 1:]


def compute_scale(Dis):
    """This function computes the scale for the Laplacian kernel."""
    return Dis.mean()


def get_laplace_kernel(Dx, scale, indices, device='cpu'):
    """This function computes the Laplacian kernel."""
    W = torch.zeros(Dx.shape, device=device)
    for i in range(Dx.shape[0]):
        for j in range(Dx.shape[1]):
            if j in indices[i]:
                W[i, j] = torch.exp(-Dx[i, j] / scale)
    return W