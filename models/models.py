import torch
from torch import nn


class DOPENeuralNet(nn.Module):
    def __init__(self, shared_dimensions, outcome_dimensions, riesz_dimensions, activation):
        super().__init__()
        self.shared_layers = MLP(dimensions=shared_dimensions, activation=activation, use_final_activation=True)
        self.outcome_layers = MLP(dimensions=outcome_dimensions, activation=activation, use_final_activation=False)
        self.riesz_layers = MLP(dimensions=riesz_dimensions, activation=activation, use_final_activation=False)

    def get_outcome_predictions(self, covariates, treatments):
        shared_representation = torch.cat([(self.shared_layers(covariates)), treatments], dim=1)
        return self.outcome_layers(shared_representation)

    def get_riesz_predictions(self, covariates, treatments):
        shared_representation = torch.cat([(self.shared_layers(covariates)), treatments], dim=1)
        return self.riesz_layers(shared_representation)


class RieszNet(nn.Module):
    def __init__(self, shared_dimensions, outcome_dimensions, activation):
        super().__init__()
        self.shared_layers = MLP(dimensions=shared_dimensions, activation=activation, use_final_activation=True)
        self.outcome_layers = MLP(dimensions=outcome_dimensions, activation=activation, use_final_activation=False)
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


class MLP(nn.Module):
    def __init__(self, dimensions, activation, use_final_activation):
        super().__init__()
        if len(dimensions) < 2:
            raise ValueError("MLP requires at least two dimensions (input and output).")

        layers = []
        pairs = list(zip(dimensions[:-1], dimensions[1:]))
        for i, (in_features, out_features) in enumerate(pairs):
            layers.append(nn.Linear(in_features, out_features))
            is_last = i == len(pairs) - 1
            if not is_last or use_final_activation:
                layers.append(activation())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
