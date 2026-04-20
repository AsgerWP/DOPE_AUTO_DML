import torch
from torch import nn


class DOPENeuralNet(nn.Module):
    def __init__(
        self, shared_dimensions, outcome_dimensions, riesz_dimensions, activation
    ):
        super().__init__()
        shared_layers = self.build_layers(shared_dimensions, activation)
        self.shared_layers = nn.Sequential(*shared_layers)

        outcome_layers = self.build_layers(outcome_dimensions, activation)
        outcome_layers.append(nn.Linear(outcome_dimensions[-1], 1))
        self.outcome_layers = nn.Sequential(*outcome_layers)

        riesz_layers = self.build_layers(riesz_dimensions, activation)
        riesz_layers.append(nn.Linear(riesz_dimensions[-1], 1))
        self.riesz_layers = nn.Sequential(*riesz_layers)

    def get_shared_representation(self, covariates, treatments):
        return torch.cat([(self.shared_layers(covariates)), treatments], dim=1)

    def get_outcome_predictions(self, covariates, treatments):
        return self.outcome_layers(
            self.get_shared_representation(covariates=covariates, treatments=treatments)
        )

    def get_riesz_predictions(self, covariates, treatments):
        return self.riesz_layers(
            self.get_shared_representation(covariates=covariates, treatments=treatments)
        )

    @staticmethod
    def build_layers(dimensions, activation):
        layers = []
        for input_dim, output_dim in zip(dimensions[:-1], dimensions[1:]):
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(activation())
        return layers
