from torch import nn

from models.neural_nets.neural_net import NeuralNetwork
from models.neural_nets.utils import MLP, TBranch, SBranch


class DOPENeuralNet(NeuralNetwork):
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
        self.shared_trunk = MLP(
            input_size=n_covariates,
            hidden_sizes=shared_hidden_layers[:-1],
            output_size=shared_hidden_layers[-1],
            activation=activation,
            dropout_prob=dropout_prob,
            activation_after_final_layer=activation_after_final_shared_layer,
        )
        if outcome_branch_type == "t_learner":
            self.outcome_branch = TBranch(
                representation_size=shared_hidden_layers[-1],
                hidden_sizes=not_shared_hidden_layers,
                activation=activation,
                dropout_prob=dropout_prob,
            )
        elif outcome_branch_type == "s_learner":
            self.outcome_branch = SBranch(
                representation_size=shared_hidden_layers[-1],
                hidden_sizes=not_shared_hidden_layers,
                activation=activation,
                dropout_prob=dropout_prob,
            )
        else:
            raise ValueError("Invalid branch type. Must be 't_learner' or 's_learner'.")

        if riesz_branch_type == "t_learner":
            self.riesz_branch = TBranch(
                representation_size=shared_hidden_layers[-1],
                hidden_sizes=not_shared_hidden_layers,
                activation=activation,
                dropout_prob=dropout_prob,
            )
        elif riesz_branch_type == "s_learner":
            self.riesz_branch = SBranch(
                representation_size=shared_hidden_layers[-1],
                hidden_sizes=not_shared_hidden_layers,
                activation=activation,
                dropout_prob=dropout_prob,
            )
        else:
            raise ValueError("Invalid branch type. Must be 't_learner' or 's_learner'.")

    def outcome_forward(self, covariates, treatment):
        device = next(self.parameters()).device
        covariates = covariates.to(device)
        treatment = treatment.to(device)
        return self.outcome_branch(self.shared_trunk(covariates), treatment)

    def riesz_forward(self, covariates, treatment):
        device = next(self.parameters()).device
        covariates = covariates.to(device)
        treatment = treatment.to(device)
        return self.riesz_branch(self.shared_trunk(covariates), treatment)

    def freeze_shared_trunk(self):
        for param in self.shared_trunk.parameters():
            param.requires_grad = False

    def get_outcome_mse_loss(self, batch):
        covariates, treatment, outcome = batch
        return nn.functional.mse_loss(self.outcome_forward(covariates, treatment), outcome)

    def get_riesz_loss(self, batch):
        covariates, treatment, _ = batch
        return (
            self.riesz_forward(covariates, treatment) ** 2
            - 2 * self.moment_functional(self.riesz_forward, covariates, treatment)
        ).mean()

    def fit_riesz_branch(self, data, batch_size=1000, epochs=1000, lr=1e-3, patience=30, weight_decay=1e-3, lambda_lasso=0):
        self._fit(
            data=data,
            loss_fn=self.get_riesz_loss,
            trunk=self.shared_trunk,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            lambda_lasso=lambda_lasso,
        )

    def fit_outcome_branch(self, data, batch_size=64, epochs=1000, lr=1e-3, patience=30, weight_decay=1e-3, lambda_lasso=0):
        self._fit(
            data=data,
            loss_fn=self.get_outcome_mse_loss,
            trunk=self.shared_trunk,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            lambda_lasso=lambda_lasso,
        )
