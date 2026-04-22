import torch
from torch import nn

from models.utils import MLP, THead, SHead


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
        self.shared_trunk = MLP(
            n_covariates, shared_hidden_layers, representation_size, activation, dropout_prob, False
        )
        if branch_type == "T":
            self.outcome_head = THead(representation_size, not_shared_hidden_layers, activation, dropout_prob)
            self.riesz_head = THead(representation_size, not_shared_hidden_layers, activation, dropout_prob)
        elif branch_type == "S":
            self.outcome_head = SHead(representation_size, not_shared_hidden_layers, activation, dropout_prob)
            self.riesz_head = SHead(representation_size, not_shared_hidden_layers, activation, dropout_prob)
        else:
            raise ValueError("Invalid branch type. Must be 'T' or 'S'.")

    def freeze_shared_trunk(self):
        for param in self.shared_trunk.parameters():
            param.requires_grad = False

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
