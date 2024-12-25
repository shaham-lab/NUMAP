from .umap_pytorch import PUMAP
from .utils import *

from sklearn.neighbors import KNeighborsRegressor

from grease import GrEASE


class NUMAP():
    def __init__(self,
                 encoder=None,
                 n_neighbors=10,
                 min_dist=0.1,
                 metric="euclidean",
                 n_components=2,
                 se_dim=2,
                 se_neighbors=10,
                 random_state=None,
                 lr=1e-2,
                 epochs=10,
                 batch_size=64,
                 num_workers=1,
                 num_gpus=1,
                 use_se=True,
                 use_residual_connections=False,
                 use_grease=False,
                 grease_batch_size=1024,
                 grease_lr=1e-3,
                 learn_from_se=True,
                 negative_sample_rate=5,
                 use_concat=False,
                 use_alpha=False,
                 alpha=0.0,
                 init_method='identity',
                 grease_hiddens=[128, 256, 256],
                 grease=None,
                 use_true_eigenvectors=True, ):
        self.encoder = encoder
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.n_components = n_components
        self.se_dim = se_dim
        self.se_neighbors = se_neighbors
        self.random_state = random_state
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_gpus = num_gpus
        self.use_se = use_se
        self.use_residual_connections = use_residual_connections
        self.use_grease = use_grease
        self.grease = grease
        self.grease_batch_size = grease_batch_size
        self.use_concat = use_concat
        self.learn_from_se = learn_from_se

        if self.use_grease and not self.use_se:
            print("Warning: use_grease is set to True but use_se is set to False. Setting use_se to True.")
            self.use_se = True

        if self.use_concat and not self.use_se:
            print("Warning: use_concat is set to True but use_se is set to False. Setting use_se to True.")
            self.use_se = True

        if self.use_concat and self.learn_from_se:
            print(
                "Warning: use_concat is set to True but learn_from_se is set to True. Setting learn_from_se to False.")
            self.learn_from_se = False

        self.pumap = None
        self.se = None
        self.grease_lr = grease_lr
        self.negative_sample_rate = negative_sample_rate
        self.use_alpha = use_alpha
        self.alpha = alpha
        self.init_method = init_method
        if self.init_method not in ['identity', 'xavier', 'one_hot']:
            raise ValueError(f"Invalid init_method: {self.init_method}")

        self.grease_hiddens = grease_hiddens
        self.use_true_eigenvectors = use_true_eigenvectors

        self.knn = None

    def fit(self, X):
        # normalize the data
        # X = (X - X.mean()) / X.std()

        self.pumap = PUMAP(
            encoder=self.encoder,
            decoder=None,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            n_components=self.n_components,
            random_state=self.random_state,
            lr=self.lr,
            epochs=self.epochs,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            num_gpus=self.num_gpus,
            use_residual_connections=self.use_residual_connections,
            learn_from_se=self.learn_from_se,
            negative_sample_rate=self.negative_sample_rate,
            use_concat=self.use_concat,
            use_alpha=self.use_alpha,
            alpha=self.alpha,
            init_method=self.init_method,
        )

        if self.use_grease:
            if self.grease is None:
                self.grease = GrEASE(n_components=self.se_dim, spectral_hiddens=self.grease_hiddens,
                                   spectral_batch_size=self.grease_batch_size,
                                   spectral_n_nbg=self.se_neighbors, spectral_lr=self.grease_lr,
                                   should_true_eigenvectors=self.use_true_eigenvectors)
                self.grease.fit(X)

            self.se = self.grease.transform(X)[:, :self.se_dim]
            self.se = torch.tensor(self.se)
        else:
            self.se = get_spectral_embedding(X, n_components=self.se_dim,
                                             n_neighbors=self.se_neighbors) if self.use_se else X
            self.knn = KNeighborsRegressor(n_neighbors=self.se_neighbors)
            self.knn.fit(X, self.se)

        # plt.scatter(self.se[:, 0], self.se[:, 1], s=1)
        # plt.savefig("figures/se.png")
        # plt.cla()

        self.pumap.fit(X, self.se)

    def transform(self, X, is_train=False):
        # S = get_spectral_embedding(X, n_components=self.n_components)
        if self.use_grease:
            S = self.grease.transform(X)[:, :self.se_dim]
            S = torch.tensor(S)
        else:
            if is_train:
                S = self.se
            else:
                S = torch.Tensor(self.knn.predict(X))

        SX = torch.cat([S, X], dim=1)

        return self.pumap.transform(SX)
