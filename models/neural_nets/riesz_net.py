import torch
from torch import nn

from models.neural_nets.neural_net import NeuralNetwork
from models.neural_nets.utils import MLP, TBranch, SBranch


class RieszNet(NeuralNetwork):
    def __init__(
        self,
        moment_functional,
        outcome_branch_type,
        n_covariates,
        shared_hidden_layers,
        not_shared_hidden_layers,
        activation,
        loss_weights,
        dropout_prob=0,
    ):
        super().__init__(moment_functional)
        self.shared_trunk = MLP(
            input_size=n_covariates + 1,
            hidden_sizes=shared_hidden_layers[:-1],
            output_size=shared_hidden_layers[-1],
            activation=activation,
            dropout_prob=dropout_prob,
            activation_after_final_layer=True,
        )
        if outcome_branch_type == "T":
            self.outcome_branch = TBranch(
                representation_size=shared_hidden_layers[-1],
                hidden_sizes=not_shared_hidden_layers,
                activation=activation,
                dropout_prob=dropout_prob,
            )
        elif outcome_branch_type == "S":
            self.outcome_branch = SBranch(
                representation_size=shared_hidden_layers[-1],
                hidden_sizes=not_shared_hidden_layers,
                activation=activation,
                dropout_prob=dropout_prob,
            )
        else:
            raise ValueError("Invalid branch type. Must be 'T' or 'S'.")

        self.riesz_branch = nn.Linear(shared_hidden_layers[-1], 1)
        self.loss_weights = loss_weights
        self.epsilon = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def uncorrected_outcome_forward(self, covariates, treatment):
        device = next(self.parameters()).device
        covariates = covariates.to(device)
        treatment = treatment.to(device)
        net_input = torch.cat((covariates, treatment), dim=1)
        return self.outcome_branch(self.shared_trunk(net_input), treatment)

    def riesz_forward(self, covariates, treatment):
        device = next(self.parameters()).device
        covariates = covariates.to(device)
        treatment = treatment.to(device)
        net_input = torch.cat((covariates, treatment), dim=1)
        return self.riesz_branch(self.shared_trunk(net_input.to(device)))

    def outcome_forward(self, covariates, treatment):
        return self.uncorrected_outcome_forward(covariates, treatment) + self.epsilon * self.riesz_forward(
            covariates, treatment
        )

    def get_riesz_net_loss(self, batch):
        covariates, treatment, outcome = batch

        outcome_loss = nn.functional.mse_loss(self.uncorrected_outcome_forward(covariates, treatment), outcome)
        riesz_loss = (
            self.riesz_forward(covariates, treatment) ** 2
            - 2 * self.moment_functional(self.riesz_forward, covariates, treatment)
        ).mean()
        tmle_loss = nn.functional.mse_loss(self.outcome_forward(covariates, treatment), outcome)

        return (
            self.loss_weights["riesz"] * riesz_loss
            + self.loss_weights["outcome"] * outcome_loss
            + self.loss_weights["tmle"] * tmle_loss
        )

    def fit(self, data, lr, weight_decay, batch_size, epochs, patience, lambda_lasso=0):
        self._fit(
            data=data,
            loss_fn=self.get_riesz_net_loss,
            trunk=self.shared_trunk,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            lambda_lasso=lambda_lasso,
        )
