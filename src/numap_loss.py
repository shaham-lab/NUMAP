import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GraphCrossEntropyLoss(nn.Module):
    def __init__(self, high_graph, eps=1e-8):
        """
        Initialize the loss function.
        :param high_graph: The weight matrix of the high-dimensional graph.
        """
        super(GraphCrossEntropyLoss, self).__init__()
        self.W_high = high_graph.reshape(-1) + eps
        self.eps = eps

    def forward(self, W_low: torch.Tensor) -> torch.Tensor:
        """
        Compute the cross-entropy between two graphs represented by the weight matrices W_high and W_low.

        Args:
            W_low (torch.Tensor):  Weight matrix of the low-dimensional graph.

        Returns:
            torch.Tensor: The cross-entropy loss.
        """

        # n_samples = W_low.shape[0]
        W_low = W_low.reshape(-1) + self.eps

        # W_low_0_mask = W_low < self.eps
        # W_low_1_mask = W_low > 1 - self.eps
        # W_high_0_mask = torch.logical_or(self.W_high < self.eps, W_low_0_mask)
        # W_high_1_mask = torch.logical_or(self.W_high > 1 - self.eps, W_low_1_mask)
        #
        # # loss11 is the loss1 for the elements in W_low that are close to 0
        # loss11 = self.W_high[W_low_0_mask].sum() / n_samples
        # # loss12 is the loss1 for the elements in W_low that are not close to 0 and W_high is not close to 0
        # loss12 = (self.W_high[~W_high_0_mask] *
        #           torch.log(self.W_high[~W_high_0_mask] / W_low[~W_high_0_mask])).sum() / n_samples
        # # loss21 is the loss2 for the elements in W_low that are close to 1
        # loss21 = (1 - self.W_high[W_low_1_mask]).sum() / n_samples
        # # loss22 is the loss2 for the elements in W_low that are not close to 1
        # loss22 = ((1 - self.W_high[~W_high_1_mask]) *
        #           torch.log((1 - self.W_high[~W_high_1_mask]) / (1 - W_low[~W_high_1_mask]))).sum() / n_samples
        #
        # print(f'loss11: {loss11}, loss12: {loss12}, loss21: {loss21}, loss22: {loss22}')
        #
        # loss1 = loss11 + loss12
        # loss2 = loss21 + loss22

        loss1 = (self.W_high * torch.log(self.W_high / W_low)).mean()
        loss2 = ((1 - self.W_high) * torch.log((1 - self.W_high) / (1 - W_low))).mean()

        # loss1 = -(self.W_high * torch.log(W_low)).mean()
        # loss2 = -((1 - self.W_high) * torch.log(1 - W_low)).mean()

        return loss1 + loss2


class UMAPLoss(nn.Module):
    def __init__(self,
                 batch_size=800,
                 negative_sample_rate=5,
                 _a=1,
                 _b=1,
                 edge_weights=None,
                 parametric_embedding=False,
                 repulsion_strength=1.0):
        """
        Generate a PyTorch-compatible loss function for UMAP loss

        Parameters
        ----------
        batch_size : int
            size of mini-batches
        negative_sample_rate : int
            number of negative samples per positive samples to train on
        _a : float
            distance parameter in embedding space
        _b : float
            distance parameter in embedding space
        edge_weights : array
            weights of all edges from sparse UMAP graph
        parametric_embedding : bool
            whether the embedding is parametric or nonparametric
        repulsion_strength : float, optional
            strength of repulsion vs attraction for cross-entropy, by default 1.0
        """
        super(UMAPLoss, self).__init__()
        self.batch_size = batch_size
        self.negative_sample_rate = negative_sample_rate
        self._a = _a
        self._b = _b
        self.edge_weights = edge_weights
        self.parametric_embedding = parametric_embedding
        self.repulsion_strength = repulsion_strength

        if not parametric_embedding:
            self.weights_tiled = torch.tensor(
                np.tile(edge_weights, negative_sample_rate + 1), dtype=torch.float32
            )

    def forward(self, embed_to_from):
        # split out to/from
        embedding_to, embedding_from = torch.split(embed_to_from, embed_to_from.shape[1] // 2, dim=1)

        # get negative samples
        embedding_neg_to = embedding_to.repeat(self.negative_sample_rate, 1)
        repeat_neg = embedding_from.repeat(self.negative_sample_rate, 1)
        embedding_neg_from = repeat_neg[torch.randperm(repeat_neg.size(0))]

        # distances between samples (and negative samples)
        distance_embedding = torch.cat(
            [
                torch.norm(embedding_to - embedding_from, dim=1),
                torch.norm(embedding_neg_to - embedding_neg_from, dim=1),
            ],
            dim=0,
        )

        # convert distances to probabilities
        log_probabilities_distance = self.convert_distance_to_log_probability(distance_embedding, self._a, self._b)

        # set true probabilities based on negative sampling
        probabilities_graph = torch.cat(
            [torch.ones(self.batch_size), torch.zeros(self.batch_size * self.negative_sample_rate)], dim=0
        )

        # compute cross entropy
        attraction_loss, repellant_loss, ce_loss = self.compute_cross_entropy(
            probabilities_graph,
            log_probabilities_distance,
            self.repulsion_strength,
        )

        if not self.parametric_embedding:
            ce_loss = ce_loss * self.weights_tiled

        return ce_loss.mean()

    def convert_distance_to_log_probability(self, distance, _a, _b):
        # UMAP-specific distance to probability conversion
        return torch.log1p(torch.exp(-_a * (distance ** _b)))

    def compute_cross_entropy(self, probabilities_graph, log_probabilities_distance, repulsion_strength):
        attraction_loss = -probabilities_graph * log_probabilities_distance
        repellant_loss = -(1.0 - probabilities_graph) * F.logsigmoid(-log_probabilities_distance)
        ce_loss = attraction_loss + repulsion_strength * repellant_loss
        return attraction_loss, repellant_loss, ce_loss

