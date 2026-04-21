import torch
from torch import nn

from models.utils import MLP


class SharedTrunk(nn.Module):
    def __init__(self, n_covariates, hidden_sizes, representation_size, activation):
        super().__init__()
        self.layers = MLP(
            input_size=n_covariates,
            hidden_sizes=hidden_sizes,
            output_size=representation_size,
            activation=activation,
            activation_after_final_layer=True,
        )


class THead(nn.Module):
    def __init__(self, representation_size, hidden_sizes, activation):
        super().__init__()
        self.t_layers = MLP(
            input_size=representation_size,
            hidden_sizes=hidden_sizes,
            output_size=1,
            activation=activation,
            activation_after_final_layer=False,
        )
        self.c_layers = MLP(
            input_size=representation_size,
            hidden_sizes=hidden_sizes,
            output_size=1,
            activation=activation,
            activation_after_final_layer=False,
        )

    def forward(self, representation, treatment):
        return self.t_layers(representation) * treatment + self.c_layers(representation) * (1 - treatment)


class SHead(nn.Module):
    def __init__(self, representation_size, hidden_sizes, activation):
        super().__init__()
        self.layers = MLP(
            input_size=representation_size + 1,
            hidden_sizes=hidden_sizes,
            output_size=1,
            activation=activation,
            activation_after_final_layer=False,
        )

    def forward(self, representation, treatment):
        x = torch.cat([representation, treatment], dim=1)
        return self.layers(x)
