from umap_pytorch import PUMAP

from grease import GrEASE


class NUMAP_FT():
    def __init__(self,
                 encoder=None,
                 n_neighbors=10,
                 min_dist=0.1,
                 metric="euclidean",
                 n_components=2,
                 se_neighbors=10,
                 random_state=None,
                 lr=1e-2,
                 epochs=10,
                 batch_size=64,
                 num_workers=1,
                 num_gpus=1,
                 grease_batch_size=1024,
                 grease_lr=1e-3,
                 negative_sample_rate=5,
                 grease=None,
                 frozen_layers=2,
                 grease_hiddens=[128, 256],):
        self.encoder = encoder
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.n_components = n_components
        self.se_neighbors = se_neighbors
        self.random_state = random_state
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_gpus = num_gpus
        self.grease = grease
        self.grease_batch_size = grease_batch_size

        self.pumap = None
        self.se = None
        self.grease_lr = grease_lr
        self.negative_sample_rate = negative_sample_rate
        self.frozen_layers = frozen_layers
        self.grease_hiddens = grease_hiddens


    def fit(self, X):
        # normalize the data
        # X = (X - X.mean()) / X.std()
        # print(self.grease_hiddens)

        if self.grease is None:
            self.grease = GrEASE(n_components=self.n_components, spectral_hiddens=self.grease_hiddens,
                               spectral_batch_size=self.grease_batch_size,
                               spectral_n_nbg=self.se_neighbors, spectral_lr=self.grease_lr,)
            self.grease.fit(X)

        # print(self.scase._spectralnet.spec_net)
        # print(self.scase._spectralnet.spec_net.layers)
        # print(self.scase._spectralnet.spec_net.architecture)
        # exit()

        # self.se = self.scase.transform(X)
        # self.se = torch.tensor(self.se)

        # plt.scatter(self.se[:, 0], self.se[:, 1], s=1)
        # plt.savefig("figures/scase.png")
        # plt.cla()

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
            model='numap_ft',
            grease=self.grease,
            frozen_layers=self.frozen_layers,
        )

        self.pumap.fit(X, self.se)

    def transform(self, X):
        return self.pumap.transform(X)
