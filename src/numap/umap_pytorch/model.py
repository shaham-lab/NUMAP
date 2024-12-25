import torch
import torch.nn as nn
import numpy as np


class conv_encoder(nn.Module):
    def __init__(self, n_components=2, device='cuda'):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=64, kernel_size=3, stride=2, padding=1,
            ),
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1,
            ),
            nn.Flatten(),
            nn.Linear(6272, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, n_components)
        ).to(device)

    def forward(self, X):
        return self.encoder(X)


class default_encoder(nn.Module):
    def __init__(self, dims, n_components=2, use_residual_connections=False, learn_from_se=True, use_concat=False,
                 use_alpha=False, alpha=0.0, se=None, init_method='identity', device='cuda'):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(dims), 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, n_components),
        ).to(device)
        self.input_dim = np.prod(dims)
        self.use_residual_connections = use_residual_connections
        self.n_components = n_components
        self.learn_from_se = learn_from_se
        self.use_concat = use_concat
        self.use_alpha = use_alpha
        self.alpha = alpha
        self.se = se

        # Initialize weights
        self._initialize_weights(init_method)

    def forward(self, X):
        if self.use_residual_connections and (self.learn_from_se and not self.use_alpha):
            return self.encoder(X[:, :self.input_dim]) + X[:, :self.n_components]
        elif self.use_residual_connections and self.use_concat:
            return self.encoder(X) + X[:, :self.n_components]
        elif self.use_residual_connections and (not self.learn_from_se and not self.use_alpha):
            return self.encoder(X[:, -self.input_dim:]) + X[:, :self.n_components]
        elif self.use_residual_connections and (not self.learn_from_se and self.use_alpha):
            return self.alpha * X[:, :self.n_components] + (1 - self.alpha) * self.encoder(X[:, -self.input_dim:])
        elif self.use_residual_connections and (self.learn_from_se and self.use_alpha):
            return self.alpha * X[:, :self.n_components] + (1 - self.alpha) * self.encoder(X[:, :self.input_dim])
        return self.encoder(X)

    def _initialize_weights(self, init_type):
        if init_type == 'one_hot':
            method = self._init_one_hot
        elif init_type == 'identity':
            method = self._init_identity
        else:
            method = self._init_xavier
        for m in self.encoder:
            if isinstance(m, nn.Linear):
                method(m)

    @staticmethod
    def _init_one_hot(layer):
        with torch.no_grad():
            rows, cols = layer.weight.shape
            one_hot_matrix = torch.zeros(rows, cols)
            for i in range(rows):
                one_hot_matrix[i, i % cols] = 1
            layer.weight.copy_(one_hot_matrix)
            if layer.bias is not None:
                layer.bias.fill_(0.0)

    @staticmethod
    def _init_identity(layer):
        with torch.no_grad():
            if layer.weight.shape[0] == layer.weight.shape[1]:
                layer.weight.copy_(torch.eye(layer.weight.shape[0]))
            else:
                rows, cols = layer.weight.shape
                eye_matrix = torch.eye(min(rows, cols))
                layer.weight[:eye_matrix.shape[0], :eye_matrix.shape[1]].copy_(eye_matrix)
            # add noise to the identity matrix
            layer.weight += 1e-2 * torch.randn_like(layer.weight)
            if layer.bias is not None:
                layer.bias.fill_(0.0)

    @staticmethod
    def _init_xavier(layer):
        nn.init.xavier_normal_(layer.weight)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0.0)


class encoder_ft(nn.Module):
    def __init__(self, input_dims, n_components=2, pretrained_model=None, ortho_matrix=None, frozen_layers=2):
        super().__init__()
        # self.encoder = pretrained_model
        hiddens = pretrained_model.architecture
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.product(input_dims), hiddens[0]),
            nn.ReLU(),
        )
        for i in range(1, len(hiddens)):
            self.encoder.add_module(f'hidden_{i}', nn.Linear(hiddens[i-1], hiddens[i]))
            self.encoder.add_module(f'relu_{i}', nn.ReLU())

        # initialize weights
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                layer.load_state_dict(pretrained_model.layers[0].state_dict(), strict=False)

        self.encoder.add_module('output', nn.Linear(hiddens[-1], n_components))
        last_layer_weights = torch.Tensor(ortho_matrix[:, 1:])
        self.encoder.output.weight.data = last_layer_weights.T
        self.encoder.cuda()


        # self.freeze_layers(frozen_layers)

    def forward(self, X):
        return self.encoder(X)

    # def freeze_layers(self, frozen_layers):
    #     layers_frozen = 0
    #     for layer in self.encoder:
    #         for param in layer.parameters():
    #             param.requires_grad = False
    #         layers_frozen += 1
    #         if layers_frozen >= frozen_layers:
    #             break


class default_decoder(nn.Module):
    def __init__(self, dims, n_components):
        super().__init__()
        self.dims = dims
        self.decoder = nn.Sequential(
            nn.Linear(n_components, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, np.product(dims)),
        ).cuda()

    def forward(self, X):
        return self.decoder(X).view(X.shape[0], *self.dims)


if __name__ == "__main__":
    model = conv_encoder(2)
    print(model.parameters)
    print(model(torch.randn((12, 1, 28, 28)).cuda()).shape)
