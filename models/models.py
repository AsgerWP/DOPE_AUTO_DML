import torch
from torch import nn


class DOPENeuralNet(nn.Module):
    def __init__(self, shared_dimensions, outcome_dimensions, riesz_dimensions, activation):
        super().__init__()
        shared_layers = build_layers(dimensions=shared_dimensions, activation=activation)
        self.shared_layers = nn.Sequential(*shared_layers)

        outcome_layers = build_layers(dimensions=outcome_dimensions, activation=activation)
        outcome_layers.append(nn.Linear(outcome_dimensions[-1], 1))
        self.outcome_layers = nn.Sequential(*outcome_layers)

        riesz_layers = build_layers(dimensions=riesz_dimensions, activation=activation)
        riesz_layers.append(nn.Linear(riesz_dimensions[-1], 1))
        self.riesz_layers = nn.Sequential(*riesz_layers)

    def get_outcome_predictions(self, covariates, treatments):
        shared_representation = torch.cat([(self.shared_layers(covariates)), treatments], dim=1)
        return self.outcome_layers(shared_representation)

    def get_riesz_predictions(self, covariates, treatments):
        shared_representation = torch.cat([(self.shared_layers(covariates)), treatments], dim=1)
        return self.riesz_layers(shared_representation)


class RieszNet(nn.Module):
    def __init__(self, shared_dimensions, outcome_dimensions, activation):
        super().__init__()
        shared_layers = build_layers(dimensions=shared_dimensions, activation=activation)
        self.shared_layers = nn.Sequential(*shared_layers)

        outcome_layers = build_layers(dimensions=outcome_dimensions, activation=activation)
        outcome_layers.append(nn.Linear(outcome_dimensions[-1], 1))
        self.outcome_layers = nn.Sequential(*outcome_layers)

        self.riesz_layers = nn.Linear(shared_dimensions[-1], 1)

        self.epsilon = nn.Parameter(torch.tensor(0.0))

    def get_outcome_predictions(self, covariates, treatments):
        shared_representation = self.shared_layers(torch.cat((covariates, treatments), dim=1))
        outcome_prediction = self.outcome_layers(shared_representation)
        riesz_prediction = self.riesz_layers(shared_representation)
        return outcome_prediction + self.epsilon * riesz_prediction

    def get_uncorrected_outcome_predictions(self, covariates, treatments):
        shared_representation = self.shared_layers(torch.cat((covariates, treatments), dim=1))
        return self.outcome_layers(shared_representation)

    def get_riesz_predictions(self, covariates, treatments):
        shared_representation = self.shared_layers(torch.cat((covariates, treatments), dim=1))
        return self.riesz_layers(shared_representation)


def build_layers(dimensions, activation):
    layers = []
    for input_dim, output_dim in zip(dimensions[:-1], dimensions[1:]):
        layers.append(nn.Linear(input_dim, output_dim))
        layers.append(activation())
    return layers
