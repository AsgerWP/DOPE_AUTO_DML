import torch
from torch import nn

from models.utils import MLP


class DOPENeuralNet(nn.Module):
    def __init__(
        self,
        n_covariates,
        shared_hidden_layers,
        not_shared_hidden_layers,
        representation_size,
        activation,
        branch_type,
        dropout_prob=0,
    ):
        super().__init__()
        self.shared_trunk = SharedTrunk(
            n_covariates, shared_hidden_layers, representation_size, activation, dropout_prob
        )
        if branch_type == "T":
            self.outcome_head = THead(representation_size, not_shared_hidden_layers, activation, dropout_prob)
            self.riesz_head = THead(representation_size, not_shared_hidden_layers, activation, dropout_prob)
        elif branch_type == "S":
            self.outcome_head = SHead(representation_size, not_shared_hidden_layers, activation, dropout_prob)
            self.riesz_head = SHead(representation_size, not_shared_hidden_layers, activation, dropout_prob)
        else:
            raise ValueError("Invalid branch type. Must be 'T' or 'S'.")

    def outcome_forward(self, covariates, treatment):
        representation = self.shared_trunk(covariates)
        return self.outcome_head(representation, treatment)

    def riesz_forward(self, covariates, treatment):
        representation = self.shared_trunk(covariates)
        return self.riesz_head(representation, treatment)

    def calculate_estimates(self, covariates, treatment, outcome):
        plugin_terms = self.outcome_forward(covariates, torch.ones_like(treatment)) - self.outcome_forward(
            covariates, torch.zeros_like(treatment)
        )
        correction_terms = self.riesz_forward(covariates, treatment) * (
            outcome - self.outcome_forward(covariates, treatment)
        )
        dr_terms = plugin_terms + correction_terms
        return {"point_estimate": dr_terms.mean(), "var_estimate": dr_terms.var()}


class SharedTrunk(nn.Module):
    def __init__(self, n_covariates, hidden_sizes, representation_size, activation, dropout_prob):
        super().__init__()
        self.layers = MLP(
            input_size=n_covariates,
            hidden_sizes=hidden_sizes,
            output_size=representation_size,
            activation=activation,
            dropout_prob=dropout_prob,
        )

    def forward(self, covariates):
        return self.layers(covariates)


class THead(nn.Module):
    def __init__(self, representation_size, hidden_sizes, activation, dropout_prob):
        super().__init__()
        self.t_layers = MLP(
            input_size=representation_size,
            hidden_sizes=hidden_sizes,
            output_size=1,
            activation=activation,
            dropout_prob=dropout_prob,
        )
        self.c_layers = MLP(
            input_size=representation_size,
            hidden_sizes=hidden_sizes,
            output_size=1,
            activation=activation,
            dropout_prob=dropout_prob,
        )

    def forward(self, representation, treatment):
        return self.t_layers(representation) * treatment + self.c_layers(representation) * (1 - treatment)


class SHead(nn.Module):
    def __init__(self, representation_size, hidden_sizes, activation, dropout_prob):
        super().__init__()
        self.layers = MLP(
            input_size=representation_size + 1,
            hidden_sizes=hidden_sizes,
            output_size=1,
            activation=activation,
            dropout_prob=dropout_prob,
        )

    def forward(self, representation, treatment):
        x = torch.cat([representation, treatment], dim=1)
        return self.layers(x)
