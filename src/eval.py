import pandas as pd
import time
import numpy as np

from src.metrics import Metrics
from src.numap.utils import get_spectral_embedding


class Eval:
    def __init__(self, X_train, X_test, y_train=None, y_test=None, n_neighbors=50, n_eigs=[4, 6, 8, 10], continuous_labels=False, normalized=True, compute_iou=True):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.n_neighbors = n_neighbors
        self.n_eigs = np.array(n_eigs) - 1
        self.continuous_labels = continuous_labels
        self.normalized = normalized
        self.compute_iou = compute_iou

        self.max_n_eig = max(self.n_eigs)
        self.X_train_se = None
        self.X_test_se = None
        self.initialize_params()

    def initialize_params(self):
        # self.X_train_se = get_laplacian_eigenvectors(self.X_train, self.n_neighbors, self.max_n_eig)
        self.X_train_se = get_spectral_embedding(self.X_train, self.n_neighbors, self.max_n_eig)
        # self.X_test_se = get_laplacian_eigenvectors(self.X_test, self.n_neighbors, self.max_n_eig)
        self.X_test_se = get_spectral_embedding(self.X_test, self.n_neighbors, self.max_n_eig)

    def eval_train(self, embedding, model=None, embedding_time=None):
        start_time = time.time()
        # embedding_se = get_laplacian_eigenvectors(embedding, self.n_neighbors, self.max_n_eig)
        embedding_se = get_spectral_embedding(embedding, self.n_neighbors, self.max_n_eig)
        print(f'Computing SE time: {time.time() - start_time}')

        if self.compute_iou:
            start_time = time.time()
            iou = Metrics.compute_iou_acc(self.X_train, embedding)
            print(f'Computing IOU time: {time.time() - start_time}')
        else:
            iou = 0

        start_time = time.time()
        fvec = Metrics.compute_grassmann_score(X_se=self.X_train_se[:, 0], embeddings_se=embedding_se[:, 0], Normalized=self.normalized)
        print(f'Computing Fiedler Vector time: {time.time() - start_time}')

        if model is None:
            results_dict = {'iou': iou, 'fiedler_vector': fvec}
        else:
            results_dict = {'model': model, 'iou': iou, 'fiedler_vector': fvec}

        if self.y_train is not None:
            start_time = time.time()
            knn = Metrics.compute_knn_acc(embedding, self.y_train, continuous_labels=self.continuous_labels)
            print(f'Computing KNN time: {time.time() - start_time}')
            results_dict['knn'] = knn
            if not self.continuous_labels:
                start_time = time.time()
                silhouette = Metrics.compute_silhouette_score(embedding, self.y_train)
                print(f'Computing Silhouette time: {time.time() - start_time}')
                results_dict['silhouette'] = silhouette

        start_time = time.time()
        for n_eig in self.n_eigs:
            grassmann = Metrics.compute_grassmann_score(X_se=self.X_train_se[:, :n_eig], embeddings_se=embedding_se[:, :n_eig], Normalized=self.normalized)
            results_dict[f'grassmann_{n_eig}'] = grassmann
        print(f'Computing Grassmann time: {time.time() - start_time}')

        if time is not None:
            results_dict['time'] = embedding_time

        return pd.DataFrame(results_dict, index=[0])

    def eval_test(self, embedding_train, embedding_test, model=None, embedding_time=None):
        # embedding_test_se = get_laplacian_eigenvectors(embedding_test, self.n_neighbors, self.max_n_eig)
        embedding_test_se = get_spectral_embedding(embedding_test, self.n_neighbors, self.max_n_eig)
        if self.compute_iou:
            iou = Metrics.compute_iou_acc(self.X_test, embedding_test)
        else:
            iou = 0
        fvec = Metrics.compute_grassmann_score(X_se=self.X_test_se[:, 0], embeddings_se=embedding_test_se[:, 0], Normalized=self.normalized)
        if model is None:
            results_dict = {'iou': iou, 'fiedler_vector': fvec}
        else:
            model = model + ' (test)'
            results_dict = {'model': model, 'iou': iou, 'fiedler_vector': fvec}

        if self.y_test is not None:
            knn = Metrics.compute_knn_acc_test(embedding_train, embedding_test, self.y_train, self.y_test, continuous_labels=self.continuous_labels)
            results_dict['knn'] = knn
            if not self.continuous_labels:
                silhouette = Metrics.compute_silhouette_score(embedding_test, self.y_test)
                results_dict['silhouette'] = silhouette

        for n_eig in self.n_eigs:
            grassmann = Metrics.compute_grassmann_score(X_se=self.X_test_se[:, :n_eig], embeddings_se=embedding_test_se[:, :n_eig], Normalized=self.normalized)
            results_dict[f'grassmann_{n_eig}'] = grassmann

        if time is not None:
            results_dict['time'] = embedding_time

        return pd.DataFrame(results_dict, index=[0])


# def evaluate_embedding(X, embedding, y=None, n_neighbors=5, n_eigs=[4, 6, 8, 10], continuous_labels=False, normalized=True, model=None, time=None):
#     iou = Metrics.compute_iou_acc(X, embedding)
#     fvec = Metrics.compute_fiedler_vector_distance(X, embedding)
#     if model is None:
#         results_dict = {'iou': iou, 'fiedler_vector': fvec}
#     else:
#         results_dict = {'model': model, 'iou': iou, 'fiedler_vector': fvec}
#     if y is not None:
#         knn = Metrics.compute_knn_acc(embedding, y, continuous_labels=continuous_labels)
#         results_dict['knn'] = knn
#         if not continuous_labels:
#             silhouette = Metrics.compute_silhouette_score(embedding, y)
#             results_dict['silhouette'] = silhouette
#
#     max_n_eig = max(n_eigs)
#     X_se = get_laplacian_eigenvectors(X, n_neighbors)[:, :max_n_eig]
#     embeddings_se = get_laplacian_eigenvectors(embedding, n_neighbors)[:, :max_n_eig]
#     for n_eig in n_eigs:
#         grassmann = Metrics.compute_grassmann_score(X_se=X_se[:, :n_eig], embeddings_se=embeddings_se[:, :n_eig],
#                                                     Normalized=normalized)
#         results_dict[f'grassmann_{n_eig}'] = grassmann
#
#     if time is not None:
#         results_dict['time'] = time
#
#     return pd.DataFrame(results_dict, index=[0])
#
#
# def evaluate_embedding_test(X_test, embedding_train, embedding_test, y_train=None, y_test=None, n_neighbors=5,
#                             n_eigs=[4, 6, 8, 10], continuous_labels=False, normalized=True, model=None):
#     iou = Metrics.compute_iou_acc(X_test, embedding_test)
#     fvec = Metrics.compute_fiedler_vector_distance(X_test, embedding_test)
#     if model is None:
#         results_dict = {'iou': iou, 'fiedler_vector': fvec}
#     else:
#         model = model + ' (test)'
#         results_dict = {'model': model, 'iou': iou, 'fiedler_vector': fvec}
#     if y_test is not None:
#         knn = Metrics.compute_knn_acc_test(embedding_train, embedding_test, y_train, y_test,
#                                            continuous_labels=continuous_labels)
#         results_dict['knn'] = knn
#         if not continuous_labels:
#             silhouette = Metrics.compute_silhouette_score(embedding_test, y_test)
#             results_dict['silhouette'] = silhouette
#
#     max_n_eig = max(n_eigs)
#     X_se = get_laplacian_eigenvectors(X_test, n_neighbors)[:, :max_n_eig]
#     embeddings_se = get_laplacian_eigenvectors(embedding_test, n_neighbors)[:, :max_n_eig]
#     for n_eig in n_eigs:
#         grassmann = Metrics.compute_grassmann_score(X_se=X_se[:, :n_eig], embeddings_se=embeddings_se[:, :n_eig],
#                                                     Normalized=normalized)
#         results_dict[f'grassmann_{n_eig}'] = grassmann
#
#     return pd.DataFrame(results_dict, index=[0])
