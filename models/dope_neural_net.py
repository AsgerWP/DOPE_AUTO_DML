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

    def outcome_mse_loss(self, batch):
        covariates, treatment, outcome = batch
        representation = self.shared_trunk(covariates)
        outcome_prediction = self.outcome_head(representation, treatment)
        return nn.functional.mse_loss(outcome_prediction, outcome)

    def riesz_loss(self, batch):
        covariates, treatment, _ = batch
        representation = self.shared_trunk(covariates)
        riesz_prediction = self.riesz_head(representation, treatment)
        riesz_treatment = self.riesz_head(representation, torch.ones_like(treatment))
        riesz_control = self.riesz_head(representation, torch.zeros_like(treatment))

        return (riesz_prediction**2 - 2 * (riesz_treatment - riesz_control)).mean()

    def calculate_estimates(self, covariates, treatment, outcome):
        representation = self.shared_trunk(covariates)
        outcome_prediction = self.outcome_head(representation, treatment)
        riesz_prediction = self.riesz_head(representation, treatment)
        outcome_treatment = self.outcome_head(representation, torch.ones_like(treatment))
        outcome_control = self.outcome_head(representation, torch.zeros_like(treatment))

        plugin_terms = outcome_treatment - outcome_control
        correction_terms = riesz_prediction * (outcome - outcome_prediction)
        dr_terms = plugin_terms + correction_terms

        return {"point_estimate": dr_terms.mean(), "var_estimate": dr_terms.var()}

    def freeze_shared_trunk(self):
        for param in self.shared_trunk.parameters():
            param.requires_grad = False


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
