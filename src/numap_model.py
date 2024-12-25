import torch.nn as nn
import torch.nn.init as init
import torch


class NUMAP(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim):
        super(NUMAP, self).__init__()
        layers = []
        in_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # print the number of NaNs in self.network(x)
        # print(f'Nans in self.network(x): {torch.isnan(self.network(x)).sum()}')
        # print the weight of the first layer
        # print(f'Weight of the first layer: {self.network[0].weight}')
        # print the gradient of the first layer
        # print(f'Gradient of the first layer: {self.network[0].weight.grad}')
        return self.network(x) + x
