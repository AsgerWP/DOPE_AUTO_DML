import torch
from torch import nn

from models.utils import MLP


class DOPENeuralNet(nn.Module):
    def __init__(
        self, n_covariates, shared_hidden_layers, not_shared_hidden_layers, representation_size, activation, branch_type
    ):
        super().__init__()
        self.shared_trunk = SharedTrunk(n_covariates, shared_hidden_layers, representation_size, activation)
        if branch_type == "T":
            self.outcome_head = THead(representation_size, not_shared_hidden_layers, activation)
            self.riesz_head = THead(representation_size, not_shared_hidden_layers, activation)
        elif branch_type == "S":
            self.outcome_head = SHead(representation_size, not_shared_hidden_layers, activation)
            self.riesz_head = SHead(representation_size, not_shared_hidden_layers, activation)
        else:
            raise ValueError("Invalid branch type. Must be 'T' or 'S'.")

    def outcome_forward(self, covariates, treatment):
        representation = self.shared_trunk(covariates)
        return self.outcome_head(representation, treatment)

    def riesz_forward(self, covariates, treatment):
        representation = self.shared_trunk(covariates)
        return self.riesz_head(representation, treatment)


class SharedTrunk(nn.Module):
    def __init__(self, n_covariates, hidden_sizes, representation_size, activation):
        super().__init__()
        self.layers = MLP(
            input_size=n_covariates,
            hidden_sizes=hidden_sizes,
            output_size=representation_size,
            activation=activation,
        )

    def forward(self, covariates):
        return self.layers(covariates)


class THead(nn.Module):
    def __init__(self, representation_size, hidden_sizes, activation):
        super().__init__()
        self.t_layers = MLP(
            input_size=representation_size,
            hidden_sizes=hidden_sizes,
            output_size=1,
            activation=activation,
        )
        self.c_layers = MLP(
            input_size=representation_size,
            hidden_sizes=hidden_sizes,
            output_size=1,
            activation=activation,
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
        )

    def forward(self, representation, treatment):
        x = torch.cat([representation, treatment], dim=1)
        return self.layers(x)
