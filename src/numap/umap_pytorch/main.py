import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss
import torch.nn.functional as F

from .data import UMAPDataset
from .modules import get_umap_graph, umap_loss
from .model import default_encoder, default_decoder, encoder_ft

from umap.umap_ import find_ab_params
import dill

""" Model """


class Model(pl.LightningModule):
    def __init__(
            self,
            lr: float,
            encoder: nn.Module,
            decoder=None,
            beta=1.0,
            min_dist=0.1,
            reconstruction_loss=F.binary_cross_entropy_with_logits,
            match_nonparametric_umap=False,
            negative_sample_rate=5,
    ):
        super().__init__()
        self.lr = lr
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta  # weight for reconstruction loss
        self.match_nonparametric_umap = match_nonparametric_umap
        self.reconstruction_loss = reconstruction_loss
        self._a, self._b = find_ab_params(1.0, min_dist)
        self.negative_sample_rate = negative_sample_rate

        # self.loss_lst = []

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        if not self.match_nonparametric_umap:
            (edges_to_exp, edges_from_exp) = batch
            embedding_to, embedding_from = self.encoder(edges_to_exp), self.encoder(edges_from_exp)
            encoder_loss = umap_loss(embedding_to, embedding_from, self._a, self._b, edges_to_exp.shape[0],
                                     negative_sample_rate=self.negative_sample_rate)
            self.log("umap_loss", encoder_loss, prog_bar=True)

            # self.loss_lst.append(encoder_loss)
            # plt.plot(self.loss_lst)
            # plt.savefig("figures/2circles/2circles_loss.png")

            if self.decoder:
                recon = self.decoder(embedding_to)
                recon_loss = self.reconstruction_loss(recon, edges_to_exp)
                self.log("recon_loss", recon_loss, prog_bar=True)
                return encoder_loss + self.beta * recon_loss
            else:
                return encoder_loss

        else:
            data, embedding = batch
            embedding_parametric = self.encoder(data)
            encoder_loss = mse_loss(embedding_parametric, embedding)
            self.log("encoder_loss", encoder_loss, prog_bar=True)
            if self.decoder:
                recon = self.decoder(embedding_parametric)
                recon_loss = self.reconstruction_loss(recon, data)
                self.log("recon_loss", recon_loss, prog_bar=True)
                return encoder_loss + self.beta * recon_loss
            else:
                return encoder_loss


""" Datamodule """


class Datamodule(pl.LightningDataModule):
    def __init__(
            self,
            dataset,
            batch_size,
            num_workers,
    ):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            num_workers=0,  # self.num_workers
            shuffle=True,
        )


class PUMAP():
    def __init__(
            self,
            encoder=None,
            decoder=None,
            n_neighbors=10,
            min_dist=0.1,
            metric="euclidean",
            n_components=2,
            beta=1.0,
            reconstruction_loss=F.binary_cross_entropy_with_logits,
            random_state=None,
            lr=1e-3,
            epochs=10,
            batch_size=64,
            num_workers=1,
            num_gpus=1,
            match_nonparametric_umap=False,
            use_residual_connections=False,
            learn_from_se=True,
            negative_sample_rate=5,
            use_concat=False,
            use_alpha=False,
            alpha=0.0,
            init_method='identity',
            model='numap',
            grease=None,
            frozen_layers=2,
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.n_components = n_components
        self.beta = beta
        self.reconstruction_loss = reconstruction_loss
        self.random_state = random_state
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_gpus = num_gpus
        self.match_nonparametric_umap = match_nonparametric_umap
        self.use_residual_connections = use_residual_connections
        self.learn_from_se = learn_from_se
        self.negative_sample_rate = negative_sample_rate
        self.use_concat = use_concat
        self.use_alpha = use_alpha
        self.alpha = alpha
        self.init_method = init_method
        self.model = model
        self.grease = grease
        self.frozen_layers = frozen_layers

    def fit(self, X, S):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.model == 'numap':
            SX = torch.cat([S, X], dim=1)
            if self.learn_from_se:
                input_dims = S.shape[1:]
            elif self.use_concat:
                input_dims = S.shape[1] + X.shape[1]
            else:
                input_dims = X.shape[1:]

            trainer = pl.Trainer(
                accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                devices=1,
                max_epochs=self.epochs
            )
            # encoder = default_encoder(X.shape[1:], self.n_components) if self.encoder is None else self.encoder
            encoder = default_encoder(input_dims, self.n_components, self.use_residual_connections, self.learn_from_se,
                                      self.use_concat, self.use_alpha, self.alpha,
                                      S, self.init_method, device=device) if self.encoder is None else self.encoder

            if self.decoder is None or isinstance(self.decoder, nn.Module):
                decoder = self.decoder
            elif self.decoder == True:
                # decoder = default_decoder(X.shape[1:], self.n_components)
                decoder = default_decoder(S.shape[1:], self.n_components)

            self.model = Model(self.lr, encoder, decoder, beta=self.beta, min_dist=self.min_dist,
                               reconstruction_loss=self.reconstruction_loss,
                               negative_sample_rate=self.negative_sample_rate)
            graph = get_umap_graph(X, n_neighbors=self.n_neighbors, metric=self.metric, random_state=self.random_state)

            trainer.fit(
                model=self.model,
                # datamodule=Datamodule(UMAPDataset(X, graph), self.batch_size, self.num_workers)
                datamodule=Datamodule(UMAPDataset(SX, graph), self.batch_size, self.num_workers)
            )
        elif self.model == 'numap_ft':
            trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=self.epochs)
            encoder = encoder_ft(X.shape[1:], self.n_components, self.grease._spectralnet.spec_net, self.grease.ortho_matrix, self.frozen_layers)

            self.model = Model(self.lr, encoder, None, beta=self.beta, min_dist=self.min_dist,
                               reconstruction_loss=self.reconstruction_loss,
                               negative_sample_rate=self.negative_sample_rate)
            graph = get_umap_graph(X, n_neighbors=self.n_neighbors, metric=self.metric, random_state=self.random_state)
            trainer.fit(
                model=self.model,
                datamodule=Datamodule(UMAPDataset(X, graph), self.batch_size, self.num_workers)
            )

    @torch.no_grad()
    def transform(self, X):
        print(f"Reducing array of shape {X.shape} to ({X.shape[0]}, {self.n_components})")
        return self.model.encoder(X).detach().cpu().numpy()

    @torch.no_grad()
    def inverse_transform(self, Z):
        return self.model.decoder(Z).detach().cpu().numpy()

    def save(self, path):
        with open(path, 'wb') as oup:
            dill.dump(self, oup)
        print(f"Pickled PUMAP object at {path}")


def load_pumap(path):
    print("Loading PUMAP object from pickled file.")
    with open(path, 'rb') as inp: return dill.load(inp)


if __name__ == "__main__":
    pass
