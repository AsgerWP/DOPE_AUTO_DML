from torch import nn

from models.neural_nets.neural_net import NeuralNetwork
from models.neural_nets.utils import MLP, TBranch, SBranch


class SeparateNeuralNets(NeuralNetwork):
    def __init__(
        self,
        moment_functional,
        n_covariates,
        shared_hidden_layers,
        not_shared_hidden_layers,
        activation,
        outcome_branch_type,
        riesz_branch_type,
        activation_after_final_shared_layer,
        dropout_prob=0,
    ):
        super().__init__(moment_functional)
        self.outcome_trunk = MLP(
            input_size=n_covariates,
            hidden_sizes=shared_hidden_layers[:-1],
            output_size=shared_hidden_layers[-1],
            activation=activation,
            dropout_prob=dropout_prob,
            activation_after_final_layer=activation_after_final_shared_layer,
        )
        self.riesz_trunk = MLP(
            input_size=n_covariates,
            hidden_sizes=shared_hidden_layers[:-1],
            output_size=shared_hidden_layers[-1],
            activation=activation,
            dropout_prob=dropout_prob,
            activation_after_final_layer=activation_after_final_shared_layer,
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

        if riesz_branch_type == "T":
            self.riesz_branch = TBranch(
                representation_size=shared_hidden_layers[-1],
                hidden_sizes=not_shared_hidden_layers,
                activation=activation,
                dropout_prob=dropout_prob,
            )
        elif riesz_branch_type == "S":
            self.riesz_branch = SBranch(
                representation_size=shared_hidden_layers[-1],
                hidden_sizes=not_shared_hidden_layers,
                activation=activation,
                dropout_prob=dropout_prob,
            )
        else:
            raise ValueError("Invalid branch type. Must be 'T' or 'S'.")

    def outcome_forward(self, covariates, treatment):
        device = next(self.parameters()).device
        covariates = covariates.to(device)
        treatment = treatment.to(device)
        return self.outcome_branch(self.outcome_trunk(covariates), treatment)

    def riesz_forward(self, covariates, treatment):
        device = next(self.parameters()).device
        covariates = covariates.to(device)
        treatment = treatment.to(device)
        return self.riesz_branch(self.riesz_trunk(covariates), treatment)

    def get_outcome_mse_loss(self, batch):
        covariates, treatment, outcome = batch
        return nn.functional.mse_loss(self.outcome_forward(covariates, treatment), outcome)

    def get_riesz_loss(self, batch):
        covariates, treatment, _ = batch
        return (
            self.riesz_forward(covariates, treatment) ** 2
            - 2 * self.moment_functional(self.riesz_forward, covariates, treatment)
        ).mean()

    def fit_riesz_branch(self, batch_size, data, epochs, lr, patience, weight_decay, lambda_lasso=0):
        self._fit(
            data=data,
            loss_fn=self.get_riesz_loss,
            trunk=self.outcome_trunk,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            lambda_lasso=lambda_lasso,
        )

    def fit_outcome_branch(self, batch_size, data, epochs, lr, patience, weight_decay, lambda_lasso=0):
        self._fit(
            data=data,
            loss_fn=self.get_outcome_mse_loss,
            trunk=self.riesz_trunk,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            lambda_lasso=lambda_lasso,
        )
