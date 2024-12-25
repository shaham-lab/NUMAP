from src.numap.utils import *

import torch
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score



def _knn_weights_without_self(distances: np.array) -> np.array:
    """defines weight function for the knn accuracy computation, such that the classification won't include the lable
     of the point itself.

    Parameters
    ----------
    distances : np.array
        An array of the distances from the neighbors.

    Returns
    -------
    np.array
        An array of the weights correspondingly to the distances - if distance is 0 than weight is 0, else the weight
        is 1 (so the weight will be uniformly distributed).
    """

    weights = (distances > 0) * 1
    return weights


class Metrics:
    def __init__(self):
        pass

    @staticmethod
    def compute_knn_acc(X: torch.Tensor, y: np.array, k: int = 5, continuous_labels: bool = False):
        """Compute the accuracy of a KNN classifier on the given data.

        Parameters
        ----------
        X : torch.Tensor
            The data.
        y : np.array
            The labels.
        k : int
            The number of neighbors for computing the KNN algorithm.

        Returns
        -------
        float
            The accuracy of the classifier.
        """
        if continuous_labels:
            neigh = KNeighborsRegressor(n_neighbors=k + 1, weights=_knn_weights_without_self, algorithm='kd_tree')
        else:
            neigh = KNeighborsClassifier(n_neighbors=k + 1, weights=_knn_weights_without_self, algorithm='kd_tree')
        neigh.fit(X, y)
        y_pred = neigh.predict(X)

        if continuous_labels:
            return np.mean((y - y_pred) ** 2)

        return accuracy_score(y, y_pred)

    @staticmethod
    def compute_knn_acc_test(X_train: torch.Tensor, X_test: torch.Tensor, y_train: np.array, y_test: np.array,
                             k: int = 5, continuous_labels: bool = False):
        """Compute the accuracy of a KNN classifier on the given data.

        Parameters
        ----------
        X_train : torch.Tensor
            The training data.
        X_test : torch.Tensor
            The test data.
        y_train : np.array
            The training labels.
        y_test : np.array
            The test labels.
        k : int
            The number of neighbors for computing the KNN algorithm.

        Returns
        -------
        float
            The accuracy of the classifier.
        """
        if continuous_labels:
            neigh = KNeighborsRegressor(n_neighbors=k + 1, weights=_knn_weights_without_self, algorithm='kd_tree')
        else:
            neigh = KNeighborsClassifier(n_neighbors=k + 1, weights=_knn_weights_without_self, algorithm='kd_tree')
        neigh.fit(X_train, y_train)
        y_pred = neigh.predict(X_test)

        if continuous_labels:
            return np.mean((y_test - y_pred) ** 2)

        return accuracy_score(y_test, y_pred)

    @staticmethod
    def compute_iou_acc(X: torch.Tensor, embeddings: np.array, k: int = 10):
        """Compute the IOU accuracy on the given data.

        Parameters
        ----------
        X : torch.Tensor
            The data.
        embeddings : np.array
            The embeddings.
        k : int
            The number of neighbors for computing the KNN algorithm.

        Returns
        -------
        float
            The accuracy of the classifier.
        """
        nbrs_X = NearestNeighbors(n_neighbors=k + 1, algorithm='kd_tree').fit(X)
        _, inds_X = nbrs_X.kneighbors(X)
        nbrs_emb = NearestNeighbors(n_neighbors=k + 1, algorithm='kd_tree').fit(embeddings)
        _, inds_emb = nbrs_emb.kneighbors(embeddings)
        inds_X, inds_emb = inds_X[:, 1:], inds_emb[:, 1:]
        # relative_intersections = np.array([
        #     len(np.intersect1d(inds_X[i], inds_emb[i])) / k for i in range(len(inds_X))
        # ])
        relative_intersections = np.mean(np.array([np.isin(row_X, row_emb) for row_X, row_emb in zip(inds_X, inds_emb)]), axis=1)
        iou = np.mean(relative_intersections)

        return iou

    @staticmethod
    def compute_silhouette_score(embeddings: np.array, labels: np.array):
        """Compute the silhouette score on the given data.

        Parameters
        ----------
        embeddings : np.array
            The embeddings.
        labels : np.array
            The labels.

        Returns
        -------
        float
            The silhouette score.
        """
        return silhouette_score(embeddings, labels)

    @staticmethod
    def compute_grassmann_distance(A: np.array, B: np.array, Normalized: bool = True):
        """Compute the Grassmann distance between two sets of subspaces.

        Parameters
        ----------
        A : np.array
            The first set of subspaces.
        B : np.array
            The second set of subspaces.
        Normalized : bool
            Whether to normalize the distance.

        Returns
        -------
        float
            The Grassmann distance.
        """
        if len(A.shape) == 1:
            A /= np.linalg.norm(A)
            B /= np.linalg.norm(B)
            grassmann = 1 - np.square(np.dot(A, B))

        else:
            A = np.linalg.qr(A)[0]
            B = np.linalg.qr(B)[0]

            M = np.dot(np.transpose(A), B)
            _, s, _ = np.linalg.svd(M, full_matrices=False)
            s = 1 - np.square(s)
            grassmann = np.sum(s)

            # grassmann = np.sum(np.sum(A * B, axis=1) ** 2)
            if Normalized:
                grassmann /= A.shape[1]

        return grassmann

    @staticmethod
    def compute_hirearchical_clustering_correlation(X: np.array, embeddings: np.array):
        """Compute the correlation between the hierarchical clustering of the data and the embeddings.

        Parameters
        ----------
        X : np.array
            The data.
        embeddings : np.array
            The embeddings.

        Returns
        -------
        float
            The correlation.
        """
        data_depths = compute_depths_list(X)
        embeddings_depths = compute_depths_list(embeddings)
        return np.corrcoef(data_depths, embeddings_depths)[0, 1]

    @staticmethod
    def compute_fiedler_value_diff(X: np.array, embeddings: np.array, n_neighbors: int = 5):
        """Compute the absolute value of the difference between the Fiedler values of the data and the embeddings.

        Parameters
        ----------
        X : np.array
            The data.
        embeddings : np.array
            The embeddings.
        n_neighbors : int
            The number of neighbors for computing the affinity matrix.

        Returns
        -------
        float
            The difference.
        """
        data_fiedler = get_laplacian_spectrum(X, n_neighbors)[1]
        embeddings_fiedler = get_laplacian_spectrum(embeddings, n_neighbors)[1]
        return np.abs(data_fiedler - embeddings_fiedler)

    @staticmethod
    def compute_L2_spectrum_score(X: np.array = None, embeddings: np.array = None, n_eigvals: int = 10,
                                  n_neighbors: int = 5, X_spectrum=None, embeddings_spectrum=None):
        """Compute the L2 norm of the difference between the eigenvalues of the data and the embeddings.

        Parameters
        ----------
        X : np.array
            The data.
        embeddings : np.array
            The embeddings.
        n_eigvals : int
            The number of eigenvalues to consider.
        n_neighbors : int
            The number of neighbors for computing the affinity matrix.
        X_spectrum : np.array (optional)
            The eigenvalues of the data.
        embeddings_spectrum : np.array (optional)
            The eigenvalues of the embeddings

        Returns
        -------
        float
            The L2 norm.
        """
        if X_spectrum is None:
            X_spectrum = get_laplacian_spectrum(X, n_neighbors)[:n_eigvals]
        if embeddings_spectrum is None:
            embeddings_spectrum = get_laplacian_spectrum(embeddings, n_neighbors)[:n_eigvals]
        return np.linalg.norm(X_spectrum - embeddings_spectrum)

    @staticmethod
    def get_Linf_spectrum_score(X: np.array = None, embeddings: np.array = None, n_eigvals: int = 10,
                                n_neighbors: int = 5, X_spectrum=None, embeddings_spectrum=None):
        """Compute the Linf norm of the difference between the eigenvalues of the data and the embeddings.

        Parameters
        ----------
        X : np.array
            The data.
        embeddings : np.array
            The embeddings.
        n_eigvals : int
            The number of eigenvalues to consider.
        n_neighbors : int
            The number of neighbors for computing the affinity matrix.
        X_spectrum : np.array (optional)
            The eigenvalues of the data.
        embeddings_spectrum : np.array (optional)
            The eigenvalues of the embeddings

        Returns
        -------
        float
            The Linf norm.
        """
        if X_spectrum is None:
            X_spectrum = get_laplacian_spectrum(X, n_neighbors)[:n_eigvals]
        if embeddings_spectrum is None:
            embeddings_spectrum = get_laplacian_spectrum(embeddings, n_neighbors)[:n_eigvals]
        return np.linalg.norm(X_spectrum - embeddings_spectrum, ord=np.inf)

    @staticmethod
    def compute_grassmann_score(X: np.array = None, embeddings: np.array = None, n_eigvecs: int = 10,
                                n_neighbors: int = 5,
                                Normalized: bool = True, X_se=None, embeddings_se=None):
        """Compute the Grassmann distance between the data and the embeddings.

        Parameters
        ----------
        X : np.array
            The data.
        embeddings : np.array
            The embeddings.
        n_neighbors : int
            The number of neighbors for computing the affinity matrix.
        Normalized : bool
            Whether to normalize the distance.
        X_se : np.array (optional)
            The spectral embedding of the data.
        embeddings_se : np.array (optional)
            The spectral embedding of the embeddings.

        Returns
        -------
        float
            The Grassmann distance.
        """
        if X_se is None:
            X_se = get_laplacian_eigenvectors(X, n_neighbors)[:, :n_eigvecs]
        if embeddings_se is None:
            embeddings_se = get_laplacian_eigenvectors(embeddings, n_neighbors)[:, :n_eigvecs]
        return Metrics.compute_grassmann_distance(X_se, embeddings_se, Normalized=Normalized)

    @staticmethod
    def compute_fiedler_vector_distance(X: np.array, embeddings: np.array, n_neighbors: int = 5):
        """Compute the distance between the Fiedler vectors of the data and the embeddings.

        Parameters
        ----------
        X : np.array
            The data.
        embeddings : np.array
            The embeddings.
        n_neighbors : int
            The number of neighbors for computing the affinity matrix.

        Returns
        -------
        float
            The distance.
        """
        # data_fiedler = get_laplacian_eigenvectors(X, n_neighbors)[:, 1]
        # embeddings_fiedler = get_laplacian_eigenvectors(embeddings, n_neighbors)[:, 1]
        # data_2evecs = get_laplacian_eigenvectors(X, n_neighbors)[:, :1]
        # embeddings_2evecs = get_laplacian_eigenvectors(embeddings, n_neighbors)[:, :1]
        # if not isConstant(data_2evecs[:, 1]):
        #     data_fiedler = np.linalg.qr(np.concatenate(np.ones((len(data_2evecs), 1)), data_2evecs[:, 1]))[0][:, 1]
        # else:
        #     data_fiedler = data_2evecs[:, 0]
        #
        # if not isConstant(embeddings_2evecs[:, 1]):
        #     embeddings_fiedler = np.linalg.qr(
        #         np.concatenate(np.ones((len(embeddings_2evecs), 1)), embeddings_2evecs[:, 1]))[0][:, 1]
        # else:
        #     embeddings_fiedler = embeddings_2evecs[:, 0]

        # return Metrics.compute_grassmann_distance(data_fiedler, embeddings_fiedler)
        return Metrics.compute_grassmann_score(X, embeddings, n_eigvecs=2, n_neighbors=n_neighbors, Normalized=True)

    @staticmethod
    def compute_all_metrics(X: np.array, embeddings: np.array, y: np.array = None, n_eigvals: int = 10,
                            n_eigvecs: int = 10,
                            n_neighbors: int = 20, Normalized: bool = True, continuous_labels: bool = False):
        """Compute all the metrics on the given data.

        Parameters
        ----------
        X : np.array
            The data.
        embeddings : np.array
            The embeddings.
        y : np.array
            The labels.
        n_neighbors : int
            The number of neighbors for computing the affinity matrix.
        Normalized : bool
            Whether to normalize the Grassmann distance.

        Returns
        -------
        dict
            A dictionary of the metrics."""
        iou_acc = Metrics.compute_iou_acc(X, embeddings)
        grassmann = Metrics.compute_grassmann_score(X, embeddings, n_eigvecs, n_neighbors, Normalized=Normalized)
        hcc = Metrics.compute_hirearchical_clustering_correlation(X, embeddings)
        fval = Metrics.compute_fiedler_value_diff(X, embeddings, n_neighbors)
        fvec = Metrics.compute_fiedler_vector_distance(X, embeddings, n_neighbors)
        l2_spectrum = Metrics.compute_L2_spectrum_score(X, embeddings, n_eigvals, n_neighbors)
        linf_spectrum = Metrics.get_Linf_spectrum_score(X, embeddings, n_eigvals, n_neighbors)
        dict = {
            'iou_acc': iou_acc,
            'grassmann': grassmann,
            'hcc': hcc,
            'fval': fval,
            'fvec': fvec,
            'l2_spectrum': l2_spectrum,
            'linf_spectrum': linf_spectrum
        }

        if not continuous_labels and y is not None:
            knn_acc = Metrics.compute_knn_acc(embeddings, y)
            silhouette = Metrics.compute_silhouette_score(embeddings, y)
            dict['knn_acc'] = knn_acc
            dict['silhouette'] = silhouette

        return dict

    @staticmethod
    def compute_all_metrics_nohp(X: np.array, embeddings: np.array, y: np.array = None, n_neighbors: int = 20,
                                 continuous_labels: bool = False):
        """Compute all the metrics, which don't require n_eig, on the given data.

        Parameters
        ----------
        X : np.array
            The data.
        embeddings : np.array
            The embeddings.
        y : np.array
            The labels.
        n_neighbors : int
            The number of neighbors for computing the affinity matrix.
        continuous_labels : bool
            Whether the labels are continuous.

        Returns
        -------
        dict
            A dictionary of the metrics.
        """
        iou_acc = Metrics.compute_iou_acc(X, embeddings)
        hcc = Metrics.compute_hirearchical_clustering_correlation(X, embeddings)
        fval = Metrics.compute_fiedler_value_diff(X, embeddings, n_neighbors)
        fvec = Metrics.compute_fiedler_vector_distance(X, embeddings, n_neighbors)
        dict = {
            'iou_acc': iou_acc,
            'hcc': hcc,
            'fval': fval,
            'fvec': fvec,
        }

        if not continuous_labels and y is not None:
            knn_acc = Metrics.compute_knn_acc(embeddings, y)
            silhouette = Metrics.compute_silhouette_score(embeddings, y)
            dict['knn_acc'] = knn_acc
            dict['silhouette'] = silhouette

        return dict

    @staticmethod
    def compute_spectral_metrics(X: np.array, embeddings: np.array, n_eigs: list = [10],
                                 n_neighbors: int = 20, normalized: bool = True):
        """Compute the spectral metrics on the given data.

        Parameters
        ----------
        X : np.array
            The data.
        embeddings : np.array
            The embeddings.
        n_eigs : int
            The number of eigenvalues to consider.
        n_neighbors : int
            The number of neighbors for computing the affinity matrix.
        normalized : bool
            Whether to normalize the Grassmann distance.

        Returns
        -------
        dict
            A dictionary of the metrics.
        """
        max_n_eig = max(n_eigs)
        X_spectrum, X_se = get_laplacian_eigs(X, n_neighbors)[:max_n_eig]
        embeddings_spectrum, embeddings_se = get_laplacian_eigs(embeddings, n_neighbors)[:max_n_eig]
        dict = {}
        for n_eig in n_eigs:
            grassmann = Metrics.compute_grassmann_score(X_se=X_se[:, :n_eig], embeddings_se=embeddings_se[:, :n_eig],
                                                        Normalized=normalized)
            l2_spectrum = Metrics.compute_L2_spectrum_score(X_spectrum=X_spectrum[:n_eig],
                                                            embeddings_spectrum=embeddings_spectrum[:n_eig])
            linf_spectrum = Metrics.get_Linf_spectrum_score(X_spectrum=X_spectrum[:n_eig],
                                                            embeddings_spectrum=embeddings_spectrum[:n_eig])
            dict[f'grassmann_{n_eig}'] = grassmann
            dict[f'l2_spectrum_{n_eig}'] = l2_spectrum
            dict[f'linf_spectrum_{n_eig}'] = linf_spectrum

        return dict
